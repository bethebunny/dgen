"""Emit valid LLVM IR text from a Module and JIT-compile via llvmlite."""

from __future__ import annotations

import _ctypes
import ctypes
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import llvmlite.binding as llvmlite

import dgen
from dgen import Type
from dgen.asm.formatting import SlotTracker
from dgen.compiler import Compiler, IdentityPass
from dgen.dialects import (
    algebra,
    builtin,
    control_flow,
    function,
    goto,
    index,
    llvm,
    memory,
    number,
)
from dgen.graph import all_values
from dgen.layout import Layout
from dgen.module import ConstantOp, Module, PackOp, pack, string_value
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.type import Constant, Memory, TypeType, Value, type_constant
from dgen.type import _format_float as format_float

# ---------------------------------------------------------------------------
# Layout → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double", "B": "i1"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double, "B": ctypes.c_bool}


def _llvm_type(layout: Layout) -> str:
    if not layout.register_passable:
        return "ptr"
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_LLVM.get(fmt, "ptr")


def _ctype(layout: Layout) -> type[ctypes._CData]:
    if not layout.register_passable:
        return ctypes.c_void_p
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_CTYPE.get(fmt, ctypes.c_void_p)


def llvm_type(t: dgen.Value[TypeType]) -> str:
    match type_constant(t):
        case memory.Reference():
            return "ptr"
        case index.Index():
            return "i64"
        case number.SignedInteger(bits):
            return f"i{bits}"
        case number.UnsignedInteger(bits):
            return f"u{bits}"
        case number.Boolean():
            return "i1"
        case number.Float64():
            return "double"
        case builtin.Nil():
            return "void"
        case goto.Label():
            return "label"
        case _:
            raise TypeError(f"Unhandled type lowered to llvm: {t.asm}")


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
# Linearized IR: the bridge between phases 2 and 3
# ---------------------------------------------------------------------------


@dataclass
class LinearBlock:
    """One LLVM basic block.

    label       — set for real goto.RegionOp/LabelOp blocks; drives phi-node emission.
    cond_branch — set when the block ends with an IfOp expansion:
                  (if_op, then_label_name, else_label_name).
    """

    name: str
    ops: list[dgen.Op] = field(default_factory=list)
    label: goto.RegionOp | goto.LabelOp | None = None
    cond_branch: tuple[control_flow.IfOp, str, str] | None = None


@dataclass
class IfMerge:
    """Phi-node info for the merge block produced by lowering an IfOp."""

    if_op: control_flow.IfOp
    then_result: dgen.Value
    then_exit: str
    else_result: dgen.Value
    else_exit: str


# Binary ops: op type → full LLVM instruction string. All entries have .lhs/.rhs.
_BINOP: dict[type[dgen.Op], str] = {
    llvm.FaddOp: "fadd double",
    llvm.FsubOp: "fsub double",
    llvm.FmulOp: "fmul double",
    llvm.FdivOp: "fdiv double",
    llvm.AddOp: "add i64",
    llvm.SubOp: "sub i64",
    llvm.MulOp: "mul i64",
    llvm.SdivOp: "sdiv i64",
    llvm.AndOp: "and i64",
    llvm.OrOp: "or i64",
    llvm.XorOp: "xor i64",
}


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def _externs(module: Module) -> list[builtin.ExternOp]:
    """Discover extern declarations from ExternOps and call sites in the module.

    Scans all functions for builtin.ExternOp instances. Returns LLVM ``declare`` strings.
    """
    externs: dict[builtin.ExternOp, None] = {
        value: None
        for top_level in module.ops
        for value in all_values(top_level)
        if isinstance(value, builtin.ExternOp)
    }
    return list(externs)


def emit_llvm_ir(module: Module) -> tuple[str, list]:
    """Emit LLVM IR text for a module.

    Returns (ir_text, host_buffers) where host_buffers keeps Memory objects
    alive for the lifetime of the JIT engine.
    """
    host_buffers: list = []

    def _lines() -> Iterator[str]:
        for extern in _externs(module):
            yield from emit_extern(extern)
        yield ""
        for func in module.functions:
            yield from _emit_func(func, host_buffers)

    return "\n".join(_lines()), host_buffers


def _result_type_str(ty: Value[dgen.TypeType]) -> str:
    """Map an op's result type to an LLVM type string, or None for void."""
    if isinstance(ty, builtin.Nil):
        return "void"
    resolved = dgen.type.type_constant(ty)
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    return _llvm_type(resolved.__layout__)


def unpack(val: Value) -> list[Value]:
    return list(val) if isinstance(val, PackOp) else [val]


# runtime_dependencies follows operands and block captures — NOT types
# or parameters. This means branch targets (which are parameters) are
# not followed, so emit_linearized won't descend into label bodies
# through branch ops. That's correct:
# - RegionOps (inline regions) appear as runtime deps of their consumers
#   and emit themselves via fall-through.
# - LabelOps (jump targets) are emitted by the region/function that
#   contains them, not by the branch that targets them.
def runtime_dependencies(value: dgen.Value) -> Iterator[dgen.Value]:
    seen = set()

    def visit(v: dgen.Value) -> Iterator[dgen.Value]:
        dependencies = chain(
            (operand for _, operand in v.operands),
            (capture for _, block in v.blocks for capture in block.captures),
        )
        for dependency in dependencies:
            if dependency not in seen:
                seen.add(dependency)
                yield from visit(dependency)
                yield dependency

    yield from visit(value)


def new_synthetic_label() -> str:
    return "synthetic:"


def emit_linearized(block: dgen.Block) -> Iterator[str]:
    for op in block.ops:
        yield from emit(op)
        if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
            return True


EMITTERS = {}


def emitter_for(ValueType: type[dgen.Value]):
    def decorator(f):
        EMITTERS[ValueType] = f
        return f

    return decorator


def emit(value: dgen.Value):
    if not isinstance(value, dgen.Op):
        return
    return (yield from EMITTERS[type(value)](value))


@emitter_for(goto.RegionOp)
def emit_region_op(op: goto.RegionOp) -> Iterator[str]:
    """Region: executes inline in use-def order (fall-through entry).

    Emits: fall-through into region block, body ops, then fall-through to exit.
    """
    yield f"  br label %{op.name}"
    yield f"{op.name}:"
    # XXX: need to emit phi ops
    terminated = yield from emit_linearized(op.body)
    if not terminated:
        # Find the exit parameter and fall through to it.
        for param in op.body.parameters:
            if param.name and param.name.startswith("exit"):
                yield f"  br label %{param.name}"
                yield f"{param.name}:"
                return
    return True


@emitter_for(goto.LabelOp)
def emit_label_op(op: goto.LabelOp) -> Iterator[str]:
    """Label: jump target only, not reachable by fall-through.

    Terminates the current basic block with a skip branch, emits the
    label body, then resumes with an exit label.
    """
    exit_name = f"{op.name}_exit"
    # Skip over the label body in the current block's flow.
    yield f"  br label %{exit_name}"
    # Emit the label body as a separate basic block.
    yield f"{op.name}:"
    # XXX: need to emit phi ops
    yield from emit_linearized(op.body)
    # Resume the enclosing block's flow.
    yield f"{exit_name}:"


@emitter_for(function.FunctionOp)
def emit_function_op(op: function.FunctionOp) -> Iterator[str]:
    ret_type = llvm_type(op.result_type)
    arguments = ", ".join(f"{llvm_type(arg.type)} %{arg.name}" for arg in op.body.args)
    yield f"define {ret_type} @{op.name}({arguments}) {{"
    yield from emit_linearized(op.body)
    # XXX: body result might be a label for instance
    yield f"  ret {typed_reference(op.body.result)}"
    yield "}"


@emitter_for(PackOp)
@emitter_for(ConstantOp)
@emitter_for(builtin.ChainOp)
@emitter_for(builtin.ExternOp)
def noop(op: dgen.Op) -> Iterator[str]:
    return ()


def emit_extern(extern: builtin.ExternOp) -> Iterator[str]:
    sym = string_value(extern.symbol)
    if isinstance(extern.type, function.Function):
        result_type = llvm_type(extern.type.result_type)
        args = ", ".join(
            f"{llvm_type(arg.type)} %{arg.name}" for arg in extern.type.arguments
        )
        yield f"declare {result_type} @{sym}({args})"
    else:
        result_type = llvm_type(extern.type)
        yield f"declare {result_type} @{sym}"


def value_reference(v: dgen.Value) -> str:
    if isinstance(v, Constant):
        return f"{v.__constant__.to_json()}"
    return f"%{v.name}"


vr = value_reference


def typed_reference(*vs: dgen.Value) -> str:
    first, *_ = vs
    vrs = ", ".join(map(value_reference, vs))
    return f"{llvm_type(first.type)} {vrs}"


def typed_references(*vs: dgen.Value) -> str:
    return ", ".join(typed_reference(v) for v in vs)


@emitter_for(algebra.AddOp)
def emit_add(op: algebra.AddOp) -> Iterator[str]:
    # XXX: types aren't comparable yet
    # if op.left.type != op.right.type:
    #     raise TypeError("codegen algebra must have the same type")
    vtype = op.left.type
    llvm_op = {
        number.Float64: "fadd",
        number.SignedInteger: "add",
        index.Index: "add",
        number.UnsignedInteger: "add",
    }[type(vtype)]
    yield f"  {llvm_op} {typed_reference(op.left, op.right)}"


@emitter_for(algebra.LessThanOp)
def emit_less_than(op: algebra.LessThanOp) -> Iterator[str]:
    # XXX: types aren't comparable yet
    # if op.left.type != op.right.type:
    #     raise TypeError("codegen algebra must have the same type")
    vtype = op.left.type
    llvm_op, optype = {
        number.Float64: ("fcmp", "olt"),
        number.SignedInteger: ("icmp", "slt"),
        index.Index: ("icmp", "slt"),
        number.UnsignedInteger: ("icmp", "ult"),
    }[type(vtype)]
    yield f"  {llvm_op} {optype} {typed_reference(op.left, op.right)}"


@emitter_for(goto.ConditionalBranchOp)
def emit_conditional_branch(op: goto.ConditionalBranchOp) -> Iterator[str]:
    yield f"  br {typed_references(op.condition, op.true_target, op.false_target)}"


@emitter_for(goto.BranchOp)
def emit_branch(op: goto.BranchOp) -> Iterator[str]:
    yield f"  br {typed_references(op.target)}"


def is_block_terminator(value: dgen.Value) -> bool:
    return isinstance(value.type, (goto.BranchOp, goto.ConditionalBranchOp))


def _emit_func(f: function.FunctionOp, host_buffers: list) -> Iterator[str]:
    """Three-phase LLVM IR emission for one function.

    Phase 1 (separate):  split mixed dgen blocks into pure label / non-label groups.
    Phase 2 (linearize): flatten the label tree into a list of LinearBlocks.
    Phase 3 (emit):      walk the list, emitting phis, ops, and terminators.
    """
    tracker: SlotTracker = SlotTracker()
    constants: dict[dgen.Value, str] = {}
    types: dict[dgen.Value, str] = {}
    param_to_label: dict[dgen.Value, goto.LabelOp] = {}
    predecessors: dict[int, list[tuple[dgen.Value, list[dgen.Value]]]] = {}
    if_merges: dict[str, IfMerge] = {}
    if_merge_targets: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Value registration helpers
    # -----------------------------------------------------------------------

    def unpack(val: Value) -> list[Value]:
        return list(val) if isinstance(val, PackOp) else [val]

    def resolve_target(target: dgen.Value) -> dgen.Value:
        """Resolve a branch target to its canonical value (label or %exit param)."""
        if isinstance(target, (goto.RegionOp, goto.LabelOp)):
            return target
        return param_to_label.get(target, target)

    def _register_constant(val: dgen.Value, mem: Memory) -> None:
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

    # -----------------------------------------------------------------------
    # Phase 1: Separate — split mixed blocks into label / non-label groups.
    #
    # A dgen block may contain both LabelOps (which become LLVM basic blocks)
    # and regular ops (which belong inside those blocks). _separate assigns
    # each non-label op to the label it data-depends on, or to a synthetic
    # block if it has no label dependency.
    # -----------------------------------------------------------------------

    _synth_n = 0

    def _label_deps(op: dgen.Op, block_ops: list[dgen.Op]) -> frozenset[goto.LabelOp]:
        """Label ops this op transitively depends on (operand edges only).

        Branch targets are parameters, not operands — they don't appear here.
        That's why cond_br groups with the no-dep ops: its target is a parameter.
        """
        labels: set[goto.LabelOp] = {
            o for o in block_ops if isinstance(o, (goto.RegionOp, goto.LabelOp))
        }
        deps: set[goto.LabelOp] = set()
        seen: set[dgen.Value] = set()

        def walk(v: dgen.Value) -> None:
            if not isinstance(v, dgen.Op) or v in seen:
                return
            seen.add(v)
            if v in labels:
                assert isinstance(v, (goto.RegionOp, goto.LabelOp))
                deps.add(v)
                return
            for _, operand in v.operands:
                walk(operand)

        walk(op)
        return frozenset(deps)

    @dataclass
    class _Seg:
        """One separated segment for _linearize to consume.

        label:      real LabelOp to recurse into (ops must be empty).
        ops:        inline ops for synthetic or anonymous segments.
        synth_name: LLVM block name when label is None and ops are non-empty.
                    If both label and synth_name are None, fold ops into the
                    current block (anonymous segment).
        """

        label: goto.LabelOp | None
        ops: list[dgen.Op]
        synth_name: str | None = None

    def _separate(block: dgen.Block) -> list[_Seg]:
        """Split a block's ops into segments by label-dependency.

        Ordering: no-dep group first, then each label followed by its
        single-dep ops, then any remaining labels, then multi-dep groups.
        """
        nonlocal _synth_n
        ops = block.ops
        labels = [op for op in ops if isinstance(op, (goto.RegionOp, goto.LabelOp))]
        if not labels:
            return [_Seg(None, ops)]

        non_label = [
            op for op in ops if not isinstance(op, (goto.RegionOp, goto.LabelOp))
        ]
        if not non_label:
            return [_Seg(label, []) for label in labels]

        groups: dict[frozenset[goto.LabelOp], list[dgen.Op]] = {}
        for op in non_label:
            groups.setdefault(_label_deps(op, ops), []).append(op)

        def synth(ops_list: list[dgen.Op]) -> _Seg:
            nonlocal _synth_n
            name, _synth_n = f"_blk{_synth_n}", _synth_n + 1
            return _Seg(None, ops_list, synth_name=name)

        result: list[_Seg] = []
        emitted: set[goto.LabelOp] = set()

        no_dep = groups.pop(frozenset(), None)
        if no_dep:
            result.append(synth(no_dep))

        for label in labels:
            dep_ops = groups.pop(frozenset({label}), None)
            if dep_ops is not None:
                result.append(_Seg(label, []))
                result.append(synth(dep_ops))
                emitted.add(label)

        for label in labels:
            if label not in emitted:
                result.append(_Seg(label, []))

        for dep_ops in groups.values():
            result.append(synth(dep_ops))

        return result

    # -----------------------------------------------------------------------
    # Phase 2: Linearize — flatten the label tree into a list of LinearBlocks.
    # -----------------------------------------------------------------------

    _if_n: int = 0
    _visited_labels: set[goto.LabelOp] = set()

    def _linearize_ops(ops: list[dgen.Op], start_name: str) -> list[LinearBlock]:
        """Linearize a flat op list, expanding IfOps into then/else/merge triples.

        IfOp expansion layout:
          <current block>  →  icmp ne + br i1 %cond, %then_N, %else_N
          then_N:  <then body ops>  →  br %merge_N
          else_N:  <else body ops>  →  br %merge_N
          merge_N: phi [then_result, %then_N], [else_result, %else_N]
                   <remaining ops after the IfOp>
        """
        nonlocal _if_n
        result: list[LinearBlock] = []

        def current() -> LinearBlock:
            if not result:
                result.append(LinearBlock(start_name))
            return result[-1]

        for op in ops:
            if not isinstance(op, control_flow.IfOp):
                _register(op)
                current().ops.append(op)
                continue

            if_id, _if_n = _if_n, _if_n + 1
            then_name = f"then_{if_id}"
            else_name = f"else_{if_id}"
            merge_name = f"merge_{if_id}"

            _register(op)
            current().cond_branch = (op, then_name, else_name)

            _register_block_args(op.then_body)
            _register_block_args(op.else_body)
            then_blocks = _linearize_ops(op.then_body.ops, then_name) or [
                LinearBlock(then_name)
            ]
            else_blocks = _linearize_ops(op.else_body.ops, else_name) or [
                LinearBlock(else_name)
            ]
            then_blocks[0].name = then_name
            else_blocks[0].name = else_name
            result.extend(then_blocks)
            result.extend(else_blocks)
            result.append(LinearBlock(merge_name))

            if_merges[merge_name] = IfMerge(
                op,
                op.then_body.result,
                then_blocks[-1].name,
                op.else_body.result,
                else_blocks[-1].name,
            )
            if_merge_targets[then_blocks[-1].name] = merge_name
            if_merge_targets[else_blocks[-1].name] = merge_name

            rt = _result_type_str(op.type)
            if rt is not None:
                types[op] = rt

        return result

    def _linearize(block: dgen.Block, name: str) -> list[LinearBlock]:
        """Recursively flatten a block and its label subtree into LinearBlocks."""
        result: list[LinearBlock] = []

        for seg in _separate(block):
            if seg.label is not None:
                # Real label: emit a phi block, then recurse into its body.
                label_name = tracker.track_name(seg.label)
                _register(seg.label)
                _register_block_args(seg.label.body)
                for param in seg.label.body.parameters:
                    tracker.track_name(param)

                if seg.label in _visited_labels:
                    continue
                _visited_labels.add(seg.label)

                label_block = LinearBlock(label_name, [], seg.label)
                result.append(label_block)
                children = _linearize(seg.label.body, label_name)
                # Merge an immediately following anonymous block of the same name
                # into the label block, avoiding a spurious block split.
                if (
                    children
                    and children[0].label is None
                    and children[0].name == label_name
                ):
                    label_block.ops.extend(children[0].ops)
                    children = children[1:]
                result.extend(children)
                for param in seg.label.body.parameters:
                    if param.name and param.name.startswith("exit"):
                        result.append(LinearBlock(tracker.track_name(param), []))

            elif seg.synth_name is not None:
                # Synthetic block: expand inline with IfOp handling.
                result.extend(_linearize_ops(seg.ops, seg.synth_name))

            else:
                # Anonymous ops: fold into the current block (or start a new one).
                cur_name = result[-1].name if result else name
                expanded = _linearize_ops(seg.ops, cur_name)
                if expanded and result and expanded[0].name == cur_name:
                    result[-1].ops.extend(expanded[0].ops)
                    result.extend(expanded[1:])
                else:
                    result.extend(expanded)

        return result

    # Populate param_to_label so resolve_target can map %self → its LabelOp.
    def _scan_self_params(block: dgen.Block) -> None:
        for op in block.ops:
            if isinstance(op, (goto.RegionOp, goto.LabelOp)):
                for param in op.body.parameters:
                    if param.name == "self":
                        param_to_label[param] = op
                _scan_self_params(op.body)

    _scan_self_params(f.body)
    _register_block_args(f.body)

    llvm_ret = (
        "void"
        if isinstance(f.result_type, builtin.Nil)
        else _llvm_type(dgen.type.type_constant(f.result_type).__layout__)
    )
    params = ", ".join(
        f"{types.get(a, 'i64')} %{tracker.track_name(a)}" for a in f.body.args
    )
    blocks = _linearize(f.body, "entry") or [LinearBlock("entry")]

    # Build predecessor map: branch target id → [(source block value, arg values)].
    for blk in blocks:
        src = dgen.Value(name=blk.name, type=goto.Label())
        for op in blk.ops:
            if isinstance(op, goto.BranchOp):
                predecessors.setdefault(id(resolve_target(op.target)), []).append(
                    (src, unpack(op.arguments))
                )
            elif isinstance(op, goto.ConditionalBranchOp):
                for target, args in [
                    (op.true_target, op.true_arguments),
                    (op.false_target, op.false_arguments),
                ]:
                    predecessors.setdefault(id(resolve_target(target)), []).append(
                        (src, unpack(args))
                    )
    # A block without a terminator falls through to the next real label block.
    for i, blk in enumerate(blocks):
        terminated = (
            any(
                isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp))
                for op in blk.ops
            )
            or blk.cond_branch is not None
        )
        if not terminated and i + 1 < len(blocks) and blocks[i + 1].label is not None:
            src = dgen.Value(name=blk.name, type=goto.Label())
            next_lbl = blocks[i + 1].label
            predecessors.setdefault(id(next_lbl), []).append(
                (src, unpack(next_lbl.initial_arguments))
            )

    # -----------------------------------------------------------------------
    # Phase 3: Emit — walk the LinearBlocks and produce LLVM IR text.
    # -----------------------------------------------------------------------

    def _emit_op(op: dgen.Op) -> Iterator[str]:
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

        name = tracker.track_name(op)
        if isinstance(
            op,
            (ConstantOp, PackOp, builtin.ChainOp, builtin.ExternOp, control_flow.IfOp),
        ):
            return
        if isinstance(op, goto.BranchOp):
            yield f"  br label %{tracker.track_name(resolve_target(op.target))}"
        elif isinstance(op, goto.ConditionalBranchOp):
            true_lbl = tracker.track_name(resolve_target(op.true_target))
            false_lbl = tracker.track_name(resolve_target(op.false_target))
            yield f"  br i1 %{tracker.track_name(op.condition)}, label %{true_lbl}, label %{false_lbl}"
        elif isinstance(op, function.CallOp):
            args = ", ".join(typed_ref(v) for v in unpack(op.arguments))
            callee = op.callee.name
            yield (
                f"  call void @{callee}({args})"
                if isinstance(op.type, builtin.Nil)
                else f"  %{name} = call {types[op]} @{callee}({args})"
            )
        elif isinstance(op, llvm.CallOp):
            args = ", ".join(typed_ref(v) for v in unpack(op.args))
            callee = string_value(op.callee)
            yield (
                f"  call void @{callee}({args})"
                if isinstance(op.type, builtin.Nil)
                else f"  %{name} = call {types[op]} @{callee}({args})"
            )
        elif isinstance(op, memory.StoreOp):
            yield f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.AllocaOp):
            yield f"  %{name} = alloca double, i64 {op.elem_count.__constant__.to_json()}"
        elif isinstance(op, llvm.GepOp):
            yield f"  %{name} = getelementptr double, ptr {bare_ref(op.base)}, {typed_ref(op.index)}"
        elif isinstance(op, memory.LoadOp):
            yield f"  %{name} = load {_llvm_type(dgen.type.type_constant(op.type).__layout__)}, {typed_ref(op.ptr)}"
        elif isinstance(op, llvm.ZextOp):
            yield f"  %{name} = zext i1 {bare_ref(op.input)} to i64"
        elif isinstance(op, llvm.FnegOp):
            yield f"  %{name} = fneg double {bare_ref(op.input)}"
        elif isinstance(op, llvm.IcmpOp):
            yield f"  %{name} = icmp {string_value(op.pred)} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(op, llvm.FcmpOp):
            yield f"  %{name} = fcmp {string_value(op.pred)} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        elif isinstance(
            op,
            (
                llvm.FaddOp,
                llvm.FsubOp,
                llvm.FmulOp,
                llvm.FdivOp,
                llvm.AddOp,
                llvm.SubOp,
                llvm.MulOp,
                llvm.SdivOp,
                llvm.AndOp,
                llvm.OrOp,
                llvm.XorOp,
            ),
        ):
            yield f"  %{name} = {_BINOP[type(op)]} {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
        else:
            raise ValueError(
                f"codegen: unhandled op {type(op).__name__} "
                f"(dialect={op.dialect.name}, asm_name={op.asm_name})"
            )

    yield f"define {llvm_ret} @{tracker.track_name(f)}({params}) {{"

    for i, blk in enumerate(blocks):
        yield f"{blk.name}:"

        # Phi nodes for real label blocks.
        if blk.label is not None:
            preds = predecessors.get(id(blk.label), [])
            for arg_idx, arg in enumerate(blk.label.body.args):
                ty = types.get(arg, "i64")
                phi_parts = [
                    f"[ {bare_ref(pred_args[arg_idx])}, %{pred_src.name} ]"
                    for pred_src, pred_args in preds
                    if arg_idx < len(pred_args)
                ]
                if phi_parts:
                    yield f"  %{tracker.track_name(arg)} = phi {ty} {', '.join(phi_parts)}"

        # Phi for if/else merge blocks.
        if blk.name in if_merges:
            m = if_merges[blk.name]
            rt = types.get(m.if_op, "i64")
            if not isinstance(m.if_op.type, builtin.Nil):
                yield (
                    f"  %{tracker.track_name(m.if_op)} = phi {rt}"
                    f" [ {bare_ref(m.then_result)}, %{m.then_exit} ],"
                    f" [ {bare_ref(m.else_result)}, %{m.else_exit} ]"
                )

        # Ops.
        has_term = False
        for op in blk.ops:
            yield from _emit_op(op)
            if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
                has_term = True

        # Conditional branch from IfOp expansion (emitted after the block's ops).
        if blk.cond_branch is not None:
            if_op, then_lbl, else_lbl = blk.cond_branch
            cond = bare_ref(if_op.condition)
            cond_type = types.get(if_op.condition, "i64")
            if cond_type != "i1":
                tmp = f"_cond_{then_lbl}"
                zero = "null" if cond_type == "ptr" else "0"
                yield f"  %{tmp} = icmp ne {cond_type} {cond}, {zero}"
                cond = f"%{tmp}"
            yield f"  br i1 {cond}, label %{then_lbl}, label %{else_lbl}"
            has_term = True

        # Fall-through terminator.
        if not has_term:
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
        """ctypes function pointer type matching this executable's signature."""
        return ctypes.CFUNCTYPE(
            _ctype(self.result_type.__layout__),
            *(_ctype(t.__layout__) for t in self.input_types),
        )

    def run(self, *args: Memory | object) -> Memory:
        """JIT and execute, returning the result as a Memory object.

        Args may be Memory objects or raw Python values (int, float, etc.),
        which are converted via Memory.from_value.
        """
        memories: list[Memory] = [
            arg if isinstance(arg, Memory) else Memory.from_value(ty, arg)
            for arg, ty in zip(args, self.input_types)
        ]
        engine = _jit_engine(self)  # must stay alive until cfunc returns
        func_ptr = engine.get_function_address(self.main_name)
        cfunc = self.ctype(func_ptr)
        raw_args = [
            m.unpack()[0] if t.__layout__.register_passable else m.address
            for m, t in zip(memories, self.input_types)
        ]
        result = cfunc(*raw_args)
        if self.result_type.__layout__.register_passable:
            return Memory.from_value(self.result_type, result)
        return Memory.from_raw(self.result_type, result)


def compile(module: Module) -> Executable:
    """Lower a Module to LLVM IR and bundle with execution metadata."""
    _dummy = Compiler([], IdentityPass())
    module = BuiltinToLLVM().run(module, _dummy)
    module = AlgebraToLLVM().run(module, _dummy)
    ir, host_buffers = emit_llvm_ir(module)
    main = module.functions[0]
    assert main.name is not None
    return Executable(
        ir=ir,
        input_types=[dgen.type.type_constant(arg.type) for arg in main.body.args],
        main_name=main.name,
        result_type=dgen.type.type_constant(main.result_type),
        host_refs=host_buffers,
    )


def _jit_engine(exe: Executable) -> Any:  # noqa: ANN401
    """Parse, verify, and create an MCJIT engine from an Executable."""
    _ensure_initialized()
    mod = llvmlite.parse_assembly(exe.ir)
    mod.verify()
    target = llvmlite.Target.from_default_triple()
    tm = target.create_target_machine()
    return llvmlite.create_mcjit_compiler(mod, tm)


def register_executable(exe: Executable) -> list[object]:
    """JIT-compile an Executable and register its main function as a global symbol.

    Returns a list of objects that must be kept alive (the JIT engine).
    """
    engine = _jit_engine(exe)
    func_ptr = engine.get_function_address(exe.main_name)
    llvmlite.add_symbol(exe.main_name, func_ptr)
    return [engine]


def _raw_to_json(raw: object, ty: dgen.Type) -> object:
    """Convert a raw ctypes callback value to a Python value.

    Scalars (int, float) pass through. Pointer types are read from memory
    via Memory.from_raw().to_json().
    """
    if not ty.__layout__.register_passable:
        assert isinstance(raw, int)
        return Memory.from_raw(ty, raw).to_json()
    return raw


def build_callback_thunk(
    func_op: function.FunctionOp,
    on_call: Callable[..., Memory],
) -> Executable:
    """Build a stage-1 thunk that forwards all args to a host callback.

    The thunk passes all function arguments to ``on_call`` via ctypes.
    ``on_call`` receives Python values (converted from raw ctypes) and
    must return a Memory object with the result.

    Handles: ctypes callback construction, LLVM thunk IR generation,
    symbol registration with llvmlite, and compilation.
    """
    assert func_op.name is not None
    callback_name = f"_stage2_{func_op.name}"
    orig_types = [dgen.type.type_constant(arg.type) for arg in func_op.body.args]
    result_type = dgen.type.type_constant(func_op.result_type)
    result_ctype: type[ctypes._CData] | None = (
        None if isinstance(result_type, builtin.Nil) else _ctype(result_type.__layout__)
    )
    param_ctypes = [_ctype(t.__layout__) for t in orig_types]
    cb_type = ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    def _callback(*raw_args: object) -> object:
        python_args = [
            _raw_to_json(raw_args[i], orig_types[i]) for i in range(len(orig_types))
        ]
        mem = on_call(*python_args)
        if not mem.type.__layout__.register_passable:
            return mem.address
        return mem.to_json()

    callback_func = cb_type(_callback)

    # Register callback symbol with llvmlite
    _ensure_initialized()
    llvmlite.add_symbol(
        callback_name,
        ctypes.cast(callback_func, ctypes.c_void_p).value,
    )

    # Build thunk: call callback with all original params, return result
    from dgen.block import BlockArgument
    from dgen.dialects.builtin import String
    from dgen.dialects.function import Function

    thunk_args = [
        BlockArgument(name=arg.name, type=arg.type) for arg in func_op.body.args
    ]
    call_op = llvm.CallOp(
        callee=String().constant(callback_name),
        args=pack(thunk_args),
        type=result_type,
    )
    thunk_func = function.FunctionOp(
        name=func_op.name,
        body=dgen.Block(result=call_op, args=thunk_args),
        result_type=result_type,
        type=Function(
            arguments=pack(arg.type for arg in thunk_args), result_type=result_type
        ),
    )
    thunk_module = Module(ops=[thunk_func])

    exe = compile(thunk_module)
    exe.host_refs.append(callback_func)  # prevent GC
    return exe


# ---------------------------------------------------------------------------
# Exit pass
# ---------------------------------------------------------------------------


class LLVMCodegen:
    """Exit pass: lower to LLVM, emit IR, bundle as Executable."""

    def run(self, module: Module) -> Executable:
        return compile(module)
