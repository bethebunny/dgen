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
from dgen.dialects import builtin, function, goto, llvm
from dgen.module import ConstantOp, Module, PackOp, string_value
from dgen.layout import Layout
from dgen.compiler import Compiler, IdentityPass
from dgen.passes.builtin_to_llvm import BuiltinToLLVMLowering
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.type import Constant, Memory, Value

# ---------------------------------------------------------------------------
# Layout → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double}


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
    print(", ".join(f"{arr[i]:g}" for i in range(size)))


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
    lines: list[str] = [
        "declare void @print_memref(ptr, i64)",
        "declare ptr @malloc(i64)",
    ]
    lines.extend(externs)
    lines.append("")
    for func in module.functions:
        lines.extend(_emit_func(func, host_buffers))
    return "\n".join(lines), host_buffers


def _result_type_str(ty: Value[dgen.TypeType]) -> str | None:
    """Derive LLVM IR type string from an op's type, or None for void ops."""
    if isinstance(ty, builtin.Nil):
        return None
    resolved = dgen.type.type_constant(ty)
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    return _llvm_type(resolved.__layout__)


def _compute_label_depths(
    all_ops: list[dgen.Op],
) -> dict[int, int]:
    """Compute nesting depth for each label from the branch structure.

    A label targeted by a branch that depends on label A gets depth = A's depth + 1.
    Labels targeted from entry code (no label dependency) get depth 0.
    Uses a simple fixed-point iteration since there are few labels.
    """
    labels = [op for op in all_ops if isinstance(op, goto.LabelOp)]
    depths: dict[int, int] = {id(l): 0 for l in labels}

    # First: figure out which label each branch op directly depends on,
    # using only ArgOp operand tracing (no recursion through depths).
    def _direct_label(op: dgen.Op) -> goto.LabelOp | None:
        """Find the label this op directly depends on via ArgOp."""
        visited: set[int] = set()

        def find(v: dgen.Value) -> goto.LabelOp | None:
            if not isinstance(v, dgen.Op) or id(v) in visited:
                return None
            visited.add(id(v))
            if isinstance(v, goto.ArgOp):
                assert isinstance(v.label, goto.LabelOp)
                return v.label
            if isinstance(v, goto.LabelOp):
                return v
            for _, operand in v.operands:
                found = find(operand)
                if found is not None:
                    return found
            return None

        return find(op)

    # Collect branch edges: (source_label_or_None, target_label)
    edges: list[tuple[goto.LabelOp | None, goto.LabelOp]] = []
    for op in all_ops:
        if isinstance(op, goto.BranchOp) and isinstance(op.target, goto.LabelOp):
            edges.append((_direct_label(op), op.target))
        elif isinstance(op, goto.ConditionalBranchOp):
            src = _direct_label(op)
            if isinstance(op.true_target, goto.LabelOp):
                edges.append((src, op.true_target))
            if isinstance(op.false_target, goto.LabelOp):
                edges.append((src, op.false_target))

    # DFS-based back-edge detection: edges to in-progress nodes are back-edges.
    adj: dict[int, list[tuple[int, int]]] = {}  # src_id → [(target_id, edge_idx)]
    ENTRY = 0
    for i, (src, target) in enumerate(edges):
        src_id = id(src) if src is not None else ENTRY
        adj.setdefault(src_id, []).append((id(target), i))

    back_edges: set[int] = set()  # edge indices
    visited: set[int] = set()
    in_progress: set[int] = set()

    def dfs(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        in_progress.add(node)
        for target_id, edge_idx in adj.get(node, []):
            if target_id in in_progress:
                back_edges.add(edge_idx)
            else:
                dfs(target_id)
        in_progress.discard(node)

    dfs(ENTRY)

    # Propagate depths along forward edges only.
    forward_edges = [
        (src, target)
        for i, (src, target) in enumerate(edges)
        if i not in back_edges
    ]
    for _ in range(len(labels) + 1):
        for src, target in forward_edges:
            src_depth = depths[id(src)] if src is not None else -1
            new_depth = src_depth + 1
            if new_depth > depths[id(target)]:
                depths[id(target)] = new_depth

    return depths


def _find_deepest_label(
    op: dgen.Op,
    cache: dict[int, goto.LabelOp | None],
    depths: dict[int, int],
) -> goto.LabelOp | None:
    """Find the deepest (innermost) label an op depends on via operands.

    When an op depends on multiple labels (e.g. load using indices from two
    different loop bodies), returns the one with the greatest depth.
    """
    if id(op) in cache:
        return cache[id(op)]
    cache[id(op)] = None  # prevent cycles

    if isinstance(op, goto.LabelOp):
        cache[id(op)] = op
        return op
    if isinstance(op, goto.ArgOp):
        assert isinstance(op.label, goto.LabelOp)
        cache[id(op)] = op.label
        return op.label

    best: goto.LabelOp | None = None
    best_depth = -1
    for _, operand in op.operands:
        if isinstance(operand, dgen.Op):
            found = _find_deepest_label(operand, cache, depths)
            if found is not None:
                d = depths.get(id(found), 0)
                if d > best_depth:
                    best = found
                    best_depth = d

    cache[id(op)] = best
    return best


def _emit_func(f: function.FunctionOp, host_buffers: list) -> list[str]:
    # Get all ops from the flat walk_ops graph.
    all_ops: list[dgen.Op] = f.body.ops

    # Read label nesting depths (annotated by ControlFlowToGoto pass).
    label_ops: list[goto.LabelOp] = [
        op for op in all_ops if isinstance(op, goto.LabelOp)
    ]
    depths: dict[int, int] = {
        id(l): getattr(l, "_depth", 0) for l in label_ops
    }
    label_cache: dict[int, goto.LabelOp | None] = {}
    # Sort labels by depth so they emit in nesting order.
    label_ops.sort(key=lambda l: depths.get(id(l), 0))
    entry_ops: list[dgen.Op] = []
    label_body_ops: dict[int, list[dgen.Op]] = {id(l): [] for l in label_ops}

    for op in all_ops:
        if isinstance(op, goto.LabelOp):
            continue
        owner = _find_deepest_label(op, label_cache, depths)
        if owner is None:
            entry_ops.append(op)
        else:
            label_body_ops[id(owner)].append(op)

    # Collect ArgOps per label (these become phi nodes), preserving walk_ops order.
    label_arg_ops: dict[int, list[goto.ArgOp]] = {id(l): [] for l in label_ops}
    for op in all_ops:
        if isinstance(op, goto.ArgOp):
            assert isinstance(op.label, goto.LabelOp)
            label_arg_ops[id(op.label)].append(op)

    # Register everything in emission order so SSA numbers are sequential.
    # LLVM requires unnamed values (%0, %1, ...) to be numbered in order.
    tracker = SlotTracker()
    for arg in f.body.args:
        tracker.track_name(arg)
    tracker.register(entry_ops)
    for label_op in label_ops:
        tracker.track_name(label_op)
        for arg_op in label_arg_ops[id(label_op)]:
            tracker.track_name(arg_op)
        tracker.register(label_body_ops[id(label_op)])

    def _unpack_args(val: Value) -> list[Value]:
        return val.values if isinstance(val, PackOp) else [val]

    def _resolve_target(target: dgen.Value) -> goto.LabelOp:
        """Resolve a branch target parameter to its LabelOp."""
        assert isinstance(target, goto.LabelOp)
        return target

    def _branch_edges(
        op: dgen.Op,
    ) -> Iterator[tuple[goto.LabelOp, list[dgen.Value]]]:
        if isinstance(op, goto.BranchOp):
            yield _resolve_target(op.target), _unpack_args(op.arguments)
        elif isinstance(op, goto.ConditionalBranchOp):
            yield _resolve_target(op.true_target), _unpack_args(op.true_arguments)
            yield _resolve_target(op.false_target), _unpack_args(op.false_arguments)

    # Build predecessor map: label → [(source_label_or_sentinel, passed_arg_values)]
    entry_sentinel = dgen.Value(name="entry", type=goto.Label())
    predecessors: dict[int, list[tuple[dgen.Value, list[dgen.Value]]]] = {
        id(l): [] for l in label_ops
    }
    for op in all_ops:
        for target_label, args in _branch_edges(op):
            src_label = _find_deepest_label(op, label_cache, depths)
            src: dgen.Value = src_label if src_label is not None else entry_sentinel
            predecessors[id(target_label)].append((src, args))

    # Pre-scan: build constants and types maps
    constants: dict[dgen.Value, str] = {}
    types: dict[dgen.Value, str] = {}

    def _register_constant(c: Constant) -> None:
        if c in constants:
            return
        mem = c.__constant__
        layout = mem.layout
        if not layout.register_passable:
            # Pointer-passed layout: emit struct address (Span: [data_ptr, len])
            host_buffers.append(mem)
            constants[c] = f"ptr inttoptr (i64 {mem.address} to ptr)"
            types[c] = "ptr"
        else:
            lt = _llvm_type(layout)
            raw = mem.unpack()[0]
            val_str = format_float(raw) if isinstance(raw, float) else str(raw)
            constants[c] = f"{lt} {val_str}"
            types[c] = lt

    # Register block arg types (function parameters)
    for arg in f.body.args:
        types[arg] = _llvm_type(dgen.type.type_constant(arg.type).__layout__)

    # Register ArgOp types (they become phi nodes).
    for label_op in label_ops:
        for arg_op in label_arg_ops[id(label_op)]:
            types[arg_op] = _llvm_type(dgen.type.type_constant(arg_op.type).__layout__)

    for op in all_ops:
        if isinstance(op, ConstantOp):
            _register_constant(op)
        elif isinstance(op, PackOp):
            continue
        elif isinstance(op, builtin.ChainOp):
            # Chain is transparent: alias to lhs at runtime
            types[op] = types.get(op.lhs, "i64")
        elif isinstance(op, goto.ArgOp):
            # ArgOps are emitted as phi nodes, not regular ops
            continue
        else:
            if (rt := _result_type_str(op.type)) is not None:
                types[op] = rt
            # Register inline Constant operands (e.g. literal args)
            for operand_name, _ in op.__operands__:
                operand = getattr(op, operand_name)
                if isinstance(operand, Constant):
                    _register_constant(operand)

    # Resolve chain aliases: ChainOp is transparent, maps to its lhs
    for op in all_ops:
        if isinstance(op, builtin.ChainOp):
            if op.lhs in constants:
                constants[op] = constants[op.lhs]
            else:
                # Point to same slot as lhs
                lhs_name = tracker.track_name(op.lhs)
                lhs_ty = types.get(op.lhs, "i64")
                constants[op] = f"{lhs_ty} %{lhs_name}"

    def typed_ref(val: dgen.Value) -> str:
        """'type value' — e.g. 'double 1.0' or 'ptr %v3'."""
        if val in constants:
            return constants[val]
        return f"{types.get(val, 'i64')} %{tracker.track_name(val)}"

    def bare_ref(val: dgen.Value) -> str:
        """Just the value — e.g. '1.0' or '%v3'."""
        if val in constants:
            return constants[val].split(" ", 1)[1]
        return f"%{tracker.track_name(val)}"

    def emit_op(op: dgen.Op, lines: list[str]) -> None:
        """Emit a single op as LLVM IR."""
        if isinstance(
            op, (ConstantOp, PackOp, builtin.ChainOp, goto.LabelOp, goto.ArgOp)
        ):
            return

        name = tracker.track_name(op)

        if isinstance(op, llvm.AllocaOp):
            lines.append(
                f"  %{name} = alloca double, i64 {op.elem_count.__constant__.to_json()}"
            )
        elif isinstance(op, llvm.GepOp):
            lines.append(
                f"  %{name} = getelementptr double, ptr {bare_ref(op.base)},"
                f" {typed_ref(op.index)}"
            )
        elif isinstance(op, llvm.LoadOp):
            lt = _llvm_type(dgen.type.type_constant(op.type).__layout__)
            lines.append(f"  %{name} = load {lt}, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.StoreOp):
            lines.append(f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.FaddOp):
            lines.append(
                f"  %{name} = fadd double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.FmulOp):
            lines.append(
                f"  %{name} = fmul double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.AddOp):
            lines.append(f"  %{name} = add i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.SubOp):
            lines.append(f"  %{name} = sub i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.MulOp):
            lines.append(f"  %{name} = mul i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.FcmpOp):
            lines.append(
                f"  %{name} = fcmp {string_value(op.pred)} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.ZextOp):
            lines.append(f"  %{name} = zext i1 {bare_ref(op.input)} to i64")
        elif isinstance(op, llvm.IcmpOp):
            lines.append(
                f"  %{name} = icmp {string_value(op.pred)} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, goto.BranchOp):
            lines.append(
                f"  br label %{tracker.track_name(_resolve_target(op.target))}"
            )
        elif isinstance(op, goto.ConditionalBranchOp):
            lines.append(
                f"  br i1 %{tracker.track_name(op.condition)}, label %{tracker.track_name(_resolve_target(op.true_target))}, label %{tracker.track_name(_resolve_target(op.false_target))}"
            )
        elif isinstance(op, llvm.CallOp):
            callee = string_value(op.callee)
            call_args = _unpack_args(op.args)
            a = ", ".join(typed_ref(arg) for arg in call_args)
            if isinstance(op.type, builtin.Nil):
                lines.append(f"  call void @{callee}({a})")
            else:
                ret_ty = types[op]
                lines.append(f"  %{name} = call {ret_ty} @{callee}({a})")

    def _needs_ret(ops: list[dgen.Op]) -> bool:
        """Return True if this block needs a ret terminator (doesn't end with br)."""
        for op in reversed(ops):
            if isinstance(
                op, (ConstantOp, PackOp, builtin.ChainOp, goto.LabelOp, goto.ArgOp)
            ):
                continue
            return not isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp))
        return True

    def emit_ret(lines: list[str], block_result: dgen.Value) -> None:
        if llvm_ret == "void":
            lines.append("  ret void")
        else:
            lines.append(f"  ret {typed_ref(block_result)}")

    assert f.name is not None
    func_name = tracker.track_name(f)
    # Derive LLVM return type from function signature
    result_type = f.result
    if isinstance(result_type, builtin.Nil):
        llvm_ret = "void"
    else:
        llvm_ret = _llvm_type(dgen.type.type_constant(result_type).__layout__)
    # Build parameter list from block args
    params = []
    for arg in f.body.args:
        name = tracker.track_name(arg)
        ty = types.get(arg, "i64")
        params.append(f"{ty} %{name}")
    param_str = ", ".join(params)
    lines = [f"define {llvm_ret} @{func_name}({param_str}) {{", "entry:"]

    # Emit entry block ops
    for op in entry_ops:
        emit_op(op, lines)
    if _needs_ret(entry_ops):
        emit_ret(lines, f.body.result)

    # Emit each label's block (with phi instructions from ArgOps)
    for label_op in label_ops:
        label_name = tracker.track_name(label_op)
        lines.append(f"{label_name}:")
        # Emit phi instructions: each ArgOp for this label becomes a phi node.
        arg_ops = label_arg_ops[id(label_op)]
        preds = predecessors[id(label_op)]
        for arg_idx, arg_op in enumerate(arg_ops):
            arg_name = tracker.track_name(arg_op)
            ty = types.get(arg_op, "i64")
            phi_parts = []
            for pred_label, pred_args in preds:
                if arg_idx < len(pred_args):
                    pred_label_name = tracker.track_name(pred_label)
                    phi_parts.append(
                        f"[ {bare_ref(pred_args[arg_idx])}, %{pred_label_name} ]"
                    )
            if phi_parts:
                lines.append(f"  %{arg_name} = phi {ty} {', '.join(phi_parts)}")
        body_ops = label_body_ops[id(label_op)]
        for body_op in body_ops:
            emit_op(body_op, lines)
        if _needs_ret(body_ops):
            emit_ret(lines, f.body.result)

    lines.append("}")
    return lines


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
    module = BuiltinToLLVMLowering().run(module, _dummy)
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
