"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import _ctypes
import ctypes
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import llvmlite.binding as llvmlite

import dgen
from dgen import Type
from dgen.asm.formatting import SlotTracker, format_float
from dgen.dialects import builtin, control_flow, function, goto, llvm
from dgen.module import ConstantOp, Module, PackOp, pack, string_value
from dgen.layout import Layout
from dgen.compiler import Compiler, IdentityPass
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.type import Constant, Memory, Value

# ---------------------------------------------------------------------------
# Layout → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double", "B": "i1"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double, "B": ctypes.c_bool}


def _llvm_type(layout: Layout) -> str:
    """Derive LLVM IR type string from a layout."""
    if not layout.register_passable:
        return "ptr"
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_LLVM.get(fmt, "ptr")


def _ctype(layout: Layout) -> type[ctypes._CData]:
    """Derive ctypes type from a layout."""
    if not layout.register_passable:
        return ctypes.c_void_p
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_CTYPE.get(fmt, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# Runtime: print_memref callback
# ---------------------------------------------------------------------------


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
def _print_memref(ptr: int, size: int) -> None:
    arr = (ctypes.c_double * size).from_address(ptr)
    print(", ".join(f"{v:g}" for v in arr))


# ---------------------------------------------------------------------------
# LLVM initialization
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        llvmlite.initialize_native_target()
        llvmlite.initialize_native_asmprinter()
        llvmlite.add_symbol(
            "print_memref", ctypes.cast(_print_memref, ctypes.c_void_p).value
        )
        _initialized = True


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def emit_llvm_ir(module: Module, *, externs: Sequence[str] = ()) -> tuple[str, list]:
    """Emit valid LLVM IR text that llvmlite can parse.

    Returns (ir_text, host_buffers) where host_buffers keeps Memory objects
    alive for the lifetime of the JIT.
    """
    host_buffers: list = []

    def _lines() -> Iterator[str]:
        yield "declare void @print_memref(ptr, i64)"
        yield "declare ptr @malloc(i64)"
        yield from externs
        yield ""
        for func in module.functions:
            yield from _emit_func(func, host_buffers)

    return "\n".join(_lines()), host_buffers


def _result_type_str(ty: Value[dgen.TypeType]) -> str | None:
    """Derive LLVM IR type string from an op's type, or None for void ops."""
    if isinstance(ty, builtin.Nil):
        return None
    resolved = dgen.type.type_constant(ty)
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    return _llvm_type(resolved.__layout__)


def _emit_func(f: function.FunctionOp, host_buffers: list) -> Iterator[str]:
    """Emit a function as LLVM IR by recursively walking block.ops."""
    tracker = SlotTracker()
    constants: dict[dgen.Value, str] = {}
    types: dict[dgen.Value, str] = {}
    param_to_label: dict[dgen.Value, goto.LabelOp] = {}
    predecessors: dict[int, list[tuple[dgen.Value, list[dgen.Value]]]] = {}
    # IfOp expansion state: cond_i1 placeholder → (IfOp, then_name, else_name)
    if_blocks: dict[int, tuple[control_flow.IfOp, str, str]] = {}
    # merge block name → (IfOp, then_result, then_exit_name, else_result, else_exit_name)
    if_phis: dict[str, tuple[control_flow.IfOp, dgen.Value, str, dgen.Value, str]] = {}
    # block name → merge block name (for then/else blocks that need br to merge)
    if_merge_targets: dict[str, str] = {}
    entry_sentinel = dgen.Value(name="entry", type=goto.Label())

    def unpack(val: Value) -> list[Value]:
        return list(val) if isinstance(val, PackOp) else [val]

    def resolve_target(target: dgen.Value) -> dgen.Value:
        if isinstance(target, goto.LabelOp):
            return target
        if target in param_to_label:
            return param_to_label[target]
        # BlockParameter like %exit — used directly as a label name
        return target

    def typed_ref(val: dgen.Value) -> str:
        return (
            constants[val]
            if val in constants
            else f"{types.get(val, 'i64')} %{tracker.track_name(val)}"
        )

    def bare_ref(val: dgen.Value) -> str:
        return (
            constants[val].split(" ", 1)[1]
            if val in constants
            else f"%{tracker.track_name(val)}"
        )

    def _register_constant(val: dgen.Value, mem: Memory) -> None:
        """Register a constant value's LLVM representation."""
        if not mem.layout.register_passable:
            host_buffers.append(mem)
            constants[val] = f"ptr inttoptr (i64 {mem.address} to ptr)"
            types[val] = "ptr"
        else:
            lt = _llvm_type(mem.layout)
            raw = mem.unpack()[0]
            constants[val] = (
                f"{lt} {format_float(raw) if isinstance(raw, float) else raw}"
            )
            types[val] = lt

    def _register(val: dgen.Value) -> None:
        """Register a value's SSA name, type, and constants. Idempotent."""
        if val in types or val in constants:
            return
        tracker.track_name(val)
        if isinstance(val, Constant):
            _register_constant(val, val.__constant__)
        elif isinstance(val, builtin.ChainOp):
            _register(val.lhs)
            types[val] = types.get(val.lhs, "i64")
            constants[val] = (
                constants[val.lhs]
                if val.lhs in constants
                else f"{types.get(val.lhs, 'i64')} %{tracker.track_name(val.lhs)}"
            )
        elif isinstance(val, dgen.Op):
            rt = _result_type_str(val.type)
            if rt is not None:
                types[val] = rt
            for _, operand in val.operands:
                if isinstance(operand, Constant):
                    _register(operand)

    def _register_block_args(block: dgen.Block) -> None:
        for arg in block.args:
            tracker.track_name(arg)
            types[arg] = _llvm_type(dgen.type.type_constant(arg.type).__layout__)

    # ===================================================================
    # Phase 1: Separate label ops from non-label ops.
    # Mixed blocks get their non-label ops wrapped into new label blocks.
    # After this, each block either only has label ops or only non-label ops.
    # ===================================================================

    _synth_counter = 0

    def _label_deps(op: dgen.Op, block_ops: list[dgen.Op]) -> frozenset[goto.LabelOp]:
        """Which label ops in this block does op transitively depend on?

        This is intentionally NOT walk_ops — we only follow operand edges,
        not parameter edges. A BranchOp's target is a parameter; it's a
        control-flow reference, not a data dependency. For scheduling the
        non-label ops into LLVM blocks, we only care about data deps.
        """
        labels_in_block: set[goto.LabelOp] = {
            o for o in block_ops if isinstance(o, goto.LabelOp)
        }
        deps: set[goto.LabelOp] = set()
        visited: set[dgen.Value] = set()

        def walk(v: dgen.Value) -> None:
            if not isinstance(v, dgen.Op) or v in visited:
                return
            visited.add(v)
            if v in labels_in_block:
                assert isinstance(v, goto.LabelOp)
                deps.add(v)
                return
            for _, operand in v.operands:
                walk(operand)

        walk(op)
        return frozenset(deps)

    LabelBlock = tuple[goto.LabelOp | None, list[dgen.Op]]
    # None means "entry" or "anonymous" block (synthetic label)

    def _separate(block: dgen.Block) -> list[LabelBlock]:
        """Split a block's ops into groups by label dependency.

        Returns a list of (label_or_None, ops) pairs. Each group becomes
        an LLVM basic block. Groups are ordered: {} first, then by the
        topo position of their label dependency.
        """
        nonlocal _synth_counter

        ops = block.ops
        labels_in_block = [op for op in ops if isinstance(op, goto.LabelOp)]
        if not labels_in_block:
            # Pure non-label block — no separation needed.
            return [(None, ops)]

        non_label_ops = [op for op in ops if not isinstance(op, goto.LabelOp)]
        if not non_label_ops:
            # Pure label block — no separation needed.
            return [(label, []) for label in labels_in_block]

        # Group non-label ops by label dependency set.
        groups: dict[frozenset[goto.LabelOp], list[dgen.Op]] = {}
        for op in non_label_ops:
            deps = _label_deps(op, ops)
            groups.setdefault(deps, []).append(op)

        def _make_synthetic_label(ops_list: list[dgen.Op]) -> goto.LabelOp:
            nonlocal _synth_counter
            synth = goto.LabelOp(
                name=f"_blk{_synth_counter}",
                initial_arguments=pack(),
                body=dgen.Block(
                    result=ops_list[-1] if ops_list else dgen.Value(type=builtin.Nil())
                ),
            )
            _synth_counter += 1
            return synth

        # Assemble result: no-dep group first, then each label followed by
        # its dependent ops group, then multi-label groups last.
        result: list[LabelBlock] = []
        emitted_labels: set[goto.LabelOp] = set()

        no_dep = groups.pop(frozenset(), None)
        if no_dep:
            result.append((_make_synthetic_label(no_dep), no_dep))

        for label in labels_in_block:
            dep_key = frozenset({label})
            dep_ops = groups.pop(dep_key, None)
            if dep_ops is not None:
                result.append((label, []))
                result.append((_make_synthetic_label(dep_ops), dep_ops))
                emitted_labels.add(label)

        for label in labels_in_block:
            if label not in emitted_labels:
                result.append((label, []))

        for dep_ops in groups.values():
            result.append((_make_synthetic_label(dep_ops), dep_ops))

        return result

    # ===================================================================
    # Phase 2: Linearize — flatten the tree into a list of LLVM blocks.
    # Each LabelBlock becomes one LLVM basic block.
    # ===================================================================

    @dataclass
    class LinearBlock:
        name: str
        ops: list[dgen.Op]
        label: goto.LabelOp | None = None

    _visited_labels: set[goto.LabelOp] = set()
    _if_counter = 0

    def _linearize_ops(ops: list[dgen.Op], start_name: str) -> list[LinearBlock]:
        """Linearize a flat list of ops, expanding IfOps inline.

        When an IfOp is encountered, the current block is split:
          current_block → br i1 %cond, %then_N, %else_N
          then_N: <then_body ops> → br %merge_N
          else_N: <else_body ops> → br %merge_N
          merge_N: phi [then_result, %then_N], [else_result, %else_N]
                   <remaining ops, with IfOp aliased to phi result>
        """
        nonlocal _if_counter

        result: list[LinearBlock] = []
        current_ops: list[dgen.Op] = []

        for op in ops:
            if not isinstance(op, control_flow.IfOp):
                _register(op)
                current_ops.append(op)
                continue

            # --- IfOp: split the block ---
            if_id = _if_counter
            _if_counter += 1
            then_name = f"then_{if_id}"
            else_name = f"else_{if_id}"
            merge_name = f"merge_{if_id}"

            # Register the IfOp itself (for its result type).
            _register(op)

            # Current block ends with a conditional branch.
            blk_name = result[-1].name if result and not current_ops else start_name
            if current_ops:
                if result:
                    result[-1].ops.extend(current_ops)
                else:
                    result.append(LinearBlock(start_name, current_ops))
                current_ops = []

            # Ensure we have a block to append the cond_br to.
            if not result:
                result.append(LinearBlock(start_name, []))

            # The condition needs icmp ne 0 to convert i64 → i1.
            cond_i1_name = f"_if_cond_{if_id}"
            cond_i1 = dgen.Value(
                name=cond_i1_name, type=llvm.Int(bits=builtin.Nil().constant(1))
            )
            tracker.track_name(cond_i1)
            types[cond_i1] = "i1"
            result[-1].ops.append(cond_i1)  # placeholder for emit

            # Record the IfOp for phase 3 to emit cond_br + icmp.
            if_blocks[id(cond_i1)] = (op, then_name, else_name)

            # Then block: linearize then_body ops.
            _register_block_args(op.then_body)
            then_body_ops = op.then_body.ops
            then_blocks = _linearize_ops(then_body_ops, then_name)
            if not then_blocks:
                then_blocks = [LinearBlock(then_name, [])]
            else:
                then_blocks[0].name = then_name
            result.extend(then_blocks)

            # Else block: linearize else_body ops.
            _register_block_args(op.else_body)
            else_body_ops = op.else_body.ops
            else_blocks = _linearize_ops(else_body_ops, else_name)
            if not else_blocks:
                else_blocks = [LinearBlock(else_name, [])]
            else:
                else_blocks[0].name = else_name
            result.extend(else_blocks)

            # Merge block: phi collects results from both branches.
            merge_block = LinearBlock(merge_name, [])
            result.append(merge_block)

            # Record the phi info for emit.
            then_result = op.then_body.result
            else_result = op.else_body.result
            # The last then/else block names (in case they expanded further).
            then_exit = then_blocks[-1].name
            else_exit = else_blocks[-1].name
            if_phis[merge_name] = (op, then_result, then_exit, else_result, else_exit)
            if_merge_targets[then_exit] = merge_name
            if_merge_targets[else_exit] = merge_name

            # Alias IfOp result to the phi value (registered in types).
            rt = _result_type_str(op.type)
            if rt is not None:
                types[op] = rt

            # Start_name for remaining ops is the merge block.
            start_name = merge_name

        # Remaining ops after last IfOp (or all ops if no IfOps).
        if current_ops:
            if result:
                result[-1].ops.extend(current_ops)
            else:
                result.append(LinearBlock(start_name, current_ops))

        return result

    def _linearize(block: dgen.Block, name: str) -> list[LinearBlock]:
        """Recursively flatten a block and its label children."""
        separated = _separate(block)
        result: list[LinearBlock] = []

        for label, ops in separated:
            if label is None:
                # Pure non-label block — expand IfOps inline.
                cur_name = result[-1].name if result else name
                expanded = _linearize_ops(ops, cur_name)
                if expanded and result and expanded[0].name == cur_name:
                    # Merge first expanded block into existing.
                    result[-1].ops.extend(expanded[0].ops)
                    result.extend(expanded[1:])
                else:
                    result.extend(expanded)
                continue

            label_name = tracker.track_name(label)
            _register(label)
            _register_block_args(label.body)
            for param in label.body.parameters:
                tracker.track_name(param)

            if label in _visited_labels:
                continue
            _visited_labels.add(label)

            if ops:
                # Synthetic block with actual ops — expand IfOps.
                expanded = _linearize_ops(ops, label_name)
                result.extend(expanded)
            else:
                # Real label — emit phis, then recurse into body.
                label_block = LinearBlock(label_name, [], label)
                result.append(label_block)
                children = _linearize(label.body, label_name)
                # If the first child is a pure-ops block with the same name,
                # merge its ops into the label block.
                if (
                    children
                    and children[0].label is None
                    and children[0].name == label_name
                ):
                    label_block.ops.extend(children[0].ops)
                    children = children[1:]
                result.extend(children)
                for param in label.body.parameters:
                    if param.name and param.name.startswith("exit"):
                        result.append(LinearBlock(tracker.track_name(param), []))

        return result

    # Register %self → label mappings (needed for resolve_target).
    def _scan_self_params(block: dgen.Block) -> None:
        for op in block.ops:
            if isinstance(op, goto.LabelOp):
                for param in op.body.parameters:
                    if param.name == "self":
                        param_to_label[param] = op
                _scan_self_params(op.body)

    _scan_self_params(f.body)

    # Register function args and linearize.
    _register_block_args(f.body)

    llvm_ret = (
        "void"
        if isinstance(f.result, builtin.Nil)
        else _llvm_type(dgen.type.type_constant(f.result).__layout__)
    )
    params = ", ".join(
        f"{types.get(a, 'i64')} %{tracker.track_name(a)}" for a in f.body.args
    )
    blocks = _linearize(f.body, "entry")
    if not blocks:
        blocks = [LinearBlock("entry", [])]

    # Build predecessor map from linearized blocks.
    # Each block's branches use the block's name as source.
    for blk in blocks:
        src = dgen.Value(name=blk.name, type=goto.Label())
        for op in blk.ops:
            if isinstance(op, goto.BranchOp):
                predecessors.setdefault(id(resolve_target(op.target)), []).append(
                    (src, unpack(op.arguments))
                )
            elif isinstance(op, goto.ConditionalBranchOp):
                for t, a in [
                    (op.true_target, op.true_arguments),
                    (op.false_target, op.false_arguments),
                ]:
                    predecessors.setdefault(id(resolve_target(t)), []).append(
                        (src, unpack(a))
                    )
    # Fall-through predecessors: if block i has no terminator, it falls through to block i+1.
    for i, blk in enumerate(blocks):
        has_term = any(
            isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)) for op in blk.ops
        )
        if not has_term and i + 1 < len(blocks) and blocks[i + 1].label is not None:
            src = dgen.Value(name=blk.name, type=goto.Label())
            label_op = blocks[i + 1].label
            init_args = unpack(label_op.initial_arguments)
            predecessors.setdefault(id(label_op), []).append((src, init_args))

    # ===================================================================
    # Phase 3: Emit — walk the linear list and emit LLVM IR.
    # ===================================================================

    def _emit_op(op: dgen.Op) -> Iterator[str]:
        name = tracker.track_name(op)
        # IfOp condition placeholder: emit cond_br (condition is i1 Boolean).
        if id(op) in if_blocks:
            if_op, then_name, else_name = if_blocks[id(op)]
            cond_ref = bare_ref(if_op.condition)
            cond_type = types.get(if_op.condition, "i1")
            if cond_type != "i1":
                yield f"  %{name} = icmp ne {cond_type} {cond_ref}, 0"
                cond_ref = f"%{name}"
            yield f"  br i1 {cond_ref}, label %{then_name}, label %{else_name}"
            return
        if isinstance(op, control_flow.IfOp):
            # IfOp itself is a no-op — result aliased to merge phi.
            return
        if isinstance(op, goto.BranchOp):
            yield f"  br label %{tracker.track_name(resolve_target(op.target))}"
        elif isinstance(op, goto.ConditionalBranchOp):
            yield f"  br i1 %{tracker.track_name(op.condition)}, label %{tracker.track_name(resolve_target(op.true_target))}, label %{tracker.track_name(resolve_target(op.false_target))}"
        elif isinstance(op, (ConstantOp, PackOp, builtin.ChainOp)):
            pass
        elif isinstance(op, function.CallOp):
            a = ", ".join(typed_ref(v) for v in unpack(op.arguments))
            callee = op.callee.name
            yield (
                f"  call void @{callee}({a})"
                if isinstance(op.type, builtin.Nil)
                else f"  %{name} = call {types[op]} @{callee}({a})"
            )
        elif isinstance(op, llvm.CallOp):
            a = ", ".join(typed_ref(v) for v in unpack(op.args))
            callee = string_value(op.callee)
            yield (
                f"  call void @{callee}({a})"
                if isinstance(op.type, builtin.Nil)
                else f"  %{name} = call {types[op]} @{callee}({a})"
            )
        elif isinstance(op, llvm.StoreOp):
            yield f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.AllocaOp):
            yield f"  %{name} = alloca double, i64 {op.elem_count.__constant__.to_json()}"
        elif isinstance(op, llvm.GepOp):
            yield f"  %{name} = getelementptr double, ptr {bare_ref(op.base)}, {typed_ref(op.index)}"
        elif isinstance(op, llvm.LoadOp):
            yield f"  %{name} = load {_llvm_type(dgen.type.type_constant(op.type).__layout__)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.ZextOp):
            yield f"  %{name} = zext i1 {bare_ref(op.input)} to i64"
        elif isinstance(op, llvm.FaddOp):
            yield f"  %{name} = fadd double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FsubOp):
            yield f"  %{name} = fsub double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FmulOp):
            yield f"  %{name} = fmul double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FdivOp):
            yield f"  %{name} = fdiv double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FnegOp):
            yield f"  %{name} = fneg double {bare_ref(op.input)}"
        elif isinstance(op, llvm.AddOp):
            yield f"  %{name} = add i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.SubOp):
            yield f"  %{name} = sub i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.MulOp):
            yield f"  %{name} = mul i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.SdivOp):
            yield f"  %{name} = sdiv i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.AndOp):
            yield f"  %{name} = and i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.OrOp):
            yield f"  %{name} = or i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.XorOp):
            yield f"  %{name} = xor i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.IcmpOp):
            yield f"  %{name} = icmp {string_value(op.pred)} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FcmpOp):
            yield f"  %{name} = fcmp {string_value(op.pred)} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        else:
            raise ValueError(
                f"codegen: unhandled op {type(op).__name__} "
                f"(dialect={op.dialect.name}, asm_name={op.asm_name})"
            )

    yield f"define {llvm_ret} @{tracker.track_name(f)}({params}) {{"

    for i, blk in enumerate(blocks):
        yield f"{blk.name}:"
        # Phi nodes for real labels.
        if blk.label is not None:
            preds = predecessors.get(id(blk.label), [])
            for arg_idx, arg in enumerate(blk.label.body.args):
                ty = types.get(arg, "i64")
                phi_parts = []
                for pred_src, pred_args in preds:
                    if arg_idx < len(pred_args):
                        phi_parts.append(
                            f"[ {bare_ref(pred_args[arg_idx])}, %{pred_src.name} ]"
                        )
                if phi_parts:
                    yield f"  %{tracker.track_name(arg)} = phi {ty} {', '.join(phi_parts)}"
        # Phi for if/else merge blocks.
        if blk.name in if_phis:
            if_op, then_result, then_exit, else_result, else_exit = if_phis[blk.name]
            rt = types.get(if_op, "i64")
            if not isinstance(if_op.type, builtin.Nil):
                yield (
                    f"  %{tracker.track_name(if_op)} = phi {rt}"
                    f" [ {bare_ref(then_result)}, %{then_exit} ],"
                    f" [ {bare_ref(else_result)}, %{else_exit} ]"
                )
        # Emit ops.
        has_terminator = False
        for op in blk.ops:
            yield from _emit_op(op)
            if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
                has_terminator = True
            if id(op) in if_blocks:
                has_terminator = True  # cond_br was emitted
        # If no terminator, branch to merge (if then/else), next block, or ret.
        if not has_terminator:
            if blk.name in if_merge_targets:
                yield f"  br label %{if_merge_targets[blk.name]}"
            elif i + 1 < len(blocks):
                yield f"  br label %{blocks[i + 1].name}"
            else:
                yield (
                    "  ret void"
                    if llvm_ret == "void"
                    else f"  ret {typed_ref(f.body.result)}"
                )

    yield "}"


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------


@dataclass
class Executable:
    """Compiled LLVM IR ready for JIT execution."""

    ir: str
    input_types: list[Type]
    result_type: Type
    main_name: str
    host_refs: list = field(default_factory=list)

    @property
    def ctype(self) -> type[_ctypes.CFuncPtr]:
        """ctypes function pointer type for this executable."""
        param_ctypes = [_ctype(t.__layout__) for t in self.input_types]
        result_ctype = _ctype(self.result_type.__layout__)
        return ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    def run(self, *args: Memory | object) -> Memory:
        """JIT and execute, returning the function's result as a Memory.

        Args can be Memory objects or raw Python values (str, int, float).
        Raw values are converted to Memory via Memory.from_value.
        """
        memories: list[Memory] = [
            arg if isinstance(arg, Memory) else Memory.from_value(ty, arg)
            for arg, ty in zip(args, self.input_types)
        ]

        engine = _jit_engine(self)
        func_ptr = engine.get_function_address(self.main_name)
        cfunc = self.ctype(func_ptr)
        ctypes_args = [
            m.unpack()[0] if t.__layout__.register_passable else m.address
            for m, t in zip(memories, self.input_types)
        ]
        result = cfunc(*ctypes_args)
        if self.result_type.__layout__.register_passable:
            return Memory.from_value(self.result_type, result)
        return Memory.from_raw(self.result_type, result)


def compile(module: Module, *, externs: Sequence[str] = ()) -> Executable:
    """Emit LLVM IR and bundle with execution metadata."""
    _dummy = Compiler([], IdentityPass())
    module = AlgebraToLLVM().run(module, _dummy)
    ir, host_buffers = emit_llvm_ir(module, externs=externs)
    main = module.functions[0]
    assert main.name is not None
    result = dgen.type.type_constant(main.result)
    return Executable(
        ir=ir,
        input_types=[dgen.type.type_constant(arg.type) for arg in main.body.args],
        main_name=main.name,
        result_type=result,
        host_refs=host_buffers,
    )


def _jit_engine(exe: Executable) -> Any:  # noqa: ANN401
    """Parse, verify, and create MCJIT engine from an Executable."""
    _ensure_initialized()
    mod = llvmlite.parse_assembly(exe.ir)
    mod.verify()
    target = llvmlite.Target.from_default_triple()
    tm = target.create_target_machine()
    return llvmlite.create_mcjit_compiler(mod, tm)


# ---------------------------------------------------------------------------
# LLVMCodegen exit pass
# ---------------------------------------------------------------------------


class LLVMCodegen:
    """Exit pass: lower builtin ops to LLVM, emit IR, bundle as Executable."""

    def run(self, module: Module) -> Executable:
        return compile(module)
