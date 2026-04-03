"""Emit valid LLVM IR text from a Module and JIT-compile via llvmlite."""

from __future__ import annotations

import _ctypes
import contextvars
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
    resolved = type_constant(t)
    match resolved:
        case memory.Reference():
            return "ptr"
        case llvm.Ptr():
            return "ptr"
        case llvm.Float():
            return "double"
        case llvm.Void():
            return "void"
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
            pass
    # Handle llvm.Int which has a Value parameter (not directly matchable)
    if isinstance(resolved, llvm.Int):
        bits = resolved.bits.__constant__.to_json()
        return f"i{bits}"
    # Fallback: use the type's layout to determine the LLVM type
    return _llvm_type(resolved.__layout__)


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
    """Discover extern declarations from ExternOps in the module."""
    externs: dict[builtin.ExternOp, None] = {
        value: None
        for top_level in module.ops
        for value in all_values(top_level)
        if isinstance(value, builtin.ExternOp)
    }
    return list(externs)


def _call_externs(module: Module) -> Iterator[str]:
    """Discover and emit declare statements for external function calls.

    Scans for llvm.CallOp and function.CallOp references to functions
    not defined in the module, and emits LLVM ``declare`` statements.
    """
    defined = {func.name for func in module.functions}
    seen: set[str] = set()

    for top_level in module.ops:
        for value in all_values(top_level):
            if isinstance(value, llvm.CallOp):
                callee = string_value(value.callee)
                if callee not in defined and callee not in seen:
                    seen.add(callee)
                    ret = llvm_type(value.type)
                    args = ", ".join(typed_reference(v) for v in unpack(value.args))
                    # Emit with arg types only (no names needed for declare).
                    arg_types = ", ".join(llvm_type(v.type) for v in unpack(value.args))
                    yield f"declare {ret} @{callee}({arg_types})"
            elif isinstance(value, function.CallOp):
                callee = value.callee.name
                if callee and callee not in defined and callee not in seen:
                    seen.add(callee)
                    ret = llvm_type(value.type)
                    arg_types = ", ".join(
                        llvm_type(v.type) for v in unpack(value.arguments)
                    )
                    yield f"declare {ret} @{callee}({arg_types})"


def emit_llvm_ir(module: Module) -> tuple[str, list[Memory]]:
    """Emit LLVM IR text for a module.

    Returns (ir_text, host_buffers) where host_buffers keeps Memory objects
    alive for the lifetime of the JIT engine.
    """
    ctx = EmitContext()
    token = _emit_ctx.set(ctx)

    try:

        def _lines() -> Iterator[str]:
            for extern in _externs(module):
                yield from emit_extern(extern)
            yield from _call_externs(module)
            yield ""
            for func in module.functions:
                build_predecessors(func, ctx)
                yield from emit(func)

        return "\n".join(_lines()), ctx.host_buffers
    finally:
        _emit_ctx.reset(token)


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


# ---------------------------------------------------------------------------
# New emitter dispatch
# ---------------------------------------------------------------------------


class CodegenSlotTracker:
    """Like SlotTracker but uses _N names for unnamed values.

    LLVM IR requires unnamed numeric values (%0, %1, ...) to appear in
    sequential textual order. Since the new emitter pre-registers all names
    before emitting, the registration order may differ from emission order.
    Using %_0, %_1, ... sidesteps this requirement.
    """

    def __init__(self) -> None:
        self._slots: dict[dgen.Value, str] = {}
        self._used: set[str] = set()
        self._counter = 0

    def track_name(self, value: dgen.Value) -> str:
        if value in self._slots:
            return self._slots[value]
        if value.name is not None and value.name not in self._used:
            name = value.name
        else:
            name = f"_{self._counter}"
            self._counter += 1
        self._slots[value] = name
        self._used.add(name)
        return name

    def __getitem__(self, value: dgen.Value) -> str:
        return self._slots[value]


@dataclass
class Predecessor:
    """One incoming edge to a label/region block."""

    source_name: str
    args: list[dgen.Value]


@dataclass
class EmitContext:
    """Shared state for the new emitter path."""

    tracker: CodegenSlotTracker = field(default_factory=CodegenSlotTracker)
    host_buffers: list[Memory] = field(default_factory=list)
    # Map from id(RegionOp/LabelOp) → list of predecessors.
    predecessors: dict[int, list[Predecessor]] = field(default_factory=dict)
    # Map from id(BlockParameter) → owning RegionOp/LabelOp.
    param_to_owner: dict[int, goto.RegionOp | goto.LabelOp] = field(
        default_factory=dict
    )


def build_predecessors(func: function.FunctionOp, ctx: EmitContext) -> None:
    """Walk the function body and record branch predecessors for each target.

    For each BranchOp/ConditionalBranchOp, records the source block name and
    the argument values carried to the target. Resolves BlockParameter targets
    (like %self, %exit) to their owning RegionOp/LabelOp via param_to_owner.

    Also records initial_arguments as a predecessor from the enclosing scope
    (the fall-through entry into a region).
    """
    from dgen.block import BlockParameter

    def _resolve(target: dgen.Value) -> dgen.Value:
        """Resolve %self parameters to owning label. Other params stay as-is."""
        if isinstance(target, (goto.RegionOp, goto.LabelOp)):
            return target
        if isinstance(target, BlockParameter) and target.name == "self":
            owner = ctx.param_to_owner.get(id(target))
            if owner is not None:
                return owner
        return target

    def _walk_block(block: dgen.Block, enclosing_name: str) -> None:
        # Pre-register all op names so phi nodes can reference values
        # from blocks that haven't been emitted yet.
        for op in block.ops:
            ctx.tracker.track_name(op)

        # Track the current LLVM basic block name. Labels create skip
        # blocks ({name}_exit), so ops after a label are in the exit block.
        current_block = enclosing_name

        for op in block.ops:
            if isinstance(op, goto.BranchOp):
                resolved = _resolve(op.target)
                pred_args = unpack(op.arguments)
                ctx.predecessors.setdefault(id(resolved), []).append(
                    Predecessor(source_name=current_block, args=pred_args)
                )
            elif isinstance(op, goto.ConditionalBranchOp):
                for target, args in [
                    (op.true_target, op.true_arguments),
                    (op.false_target, op.false_arguments),
                ]:
                    resolved = _resolve(target)
                    pred_args = unpack(args)
                    ctx.predecessors.setdefault(id(resolved), []).append(
                        Predecessor(source_name=current_block, args=pred_args)
                    )
            # Recurse into nested blocks.
            if isinstance(op, (goto.RegionOp, goto.LabelOp)):
                label_name = ctx.tracker.track_name(op)
                for arg in op.body.args:
                    ctx.tracker.track_name(arg)
                for param in op.body.parameters:
                    ctx.tracker.track_name(param)
                    ctx.param_to_owner[id(param)] = op
                # Record initial_arguments as the fall-through entry predecessor.
                # Only add if there are actual values (empty init_args means
                # the region has no entry values, e.g. if-merge regions).
                init_args = unpack(op.initial_arguments)
                if op.body.args and init_args:
                    ctx.predecessors.setdefault(id(op), []).append(
                        Predecessor(source_name=current_block, args=init_args)
                    )
                _walk_block(op.body, label_name)
                # After a LabelOp, remaining ops are in the skip-exit block.
                if isinstance(op, goto.LabelOp):
                    current_block = f"{label_name}_exit"
                # After a RegionOp with an exit parameter, remaining ops
                # are in the exit block.
                elif isinstance(op, goto.RegionOp):
                    for param in op.body.parameters:
                        if param.name and (
                            param.name.startswith("exit")
                            or param.name.startswith("if_exit")
                        ):
                            current_block = param.name
                            break

    # Pre-register function args.
    for arg in func.body.args:
        ctx.tracker.track_name(arg)
    _walk_block(func.body, "entry")


_emit_ctx: contextvars.ContextVar[EmitContext] = contextvars.ContextVar("_emit_ctx")


def _ctx() -> EmitContext:
    try:
        return _emit_ctx.get()
    except LookupError:
        ctx = EmitContext()
        _emit_ctx.set(ctx)
        return ctx


# Ops whose results are structural / have no LLVM SSA value.
_NO_ASSIGN_OPS: tuple[type[dgen.Op], ...] = (
    goto.RegionOp,
    goto.LabelOp,
    function.FunctionOp,
    goto.BranchOp,
    goto.ConditionalBranchOp,
    memory.StoreOp,
    PackOp,
    ConstantOp,
    builtin.ChainOp,
    builtin.ExternOp,
)


def emit_linearized(block: dgen.Block) -> Iterator[str]:
    for op in block.ops:
        yield from emit(op)
        if isinstance(op, (goto.BranchOp, goto.ConditionalBranchOp)):
            return True


EMITTERS: dict[type[dgen.Value], Callable[..., Iterator[str]]] = {}


def emitter_for(
    ValueType: type[dgen.Value],
) -> Callable[[Callable[..., Iterator[str]]], Callable[..., Iterator[str]]]:
    def decorator(
        f: Callable[..., Iterator[str]],
    ) -> Callable[..., Iterator[str]]:
        EMITTERS[ValueType] = f
        return f

    return decorator


def emit(value: dgen.Value) -> Iterator[str]:
    if not isinstance(value, dgen.Op):
        return
    emitter = EMITTERS.get(type(value))
    if emitter is None:
        raise ValueError(
            f"codegen: unhandled op {type(value).__name__} "
            f"(dialect={value.dialect.name}, asm_name={value.asm_name})"
        )
    lines = emitter(value)
    # For ops that produce an SSA value, prepend %name = to the first line.
    if isinstance(value, _NO_ASSIGN_OPS):
        yield from lines
    elif isinstance(value.type, builtin.Nil):
        # Void-typed ops (e.g. void calls) — emit without assignment.
        yield from lines
    else:
        ctx = _ctx()
        name = ctx.tracker.track_name(value)
        first = True
        for line in lines:
            if first:
                # Prepend assignment to the first instruction line.
                # Lines are indented with 2 spaces: "  fadd double %a, %b"
                yield f"  %{name} = {line.lstrip(' ')}"
                first = False
            else:
                yield line
    return


def _emit_phi_nodes(
    op: goto.RegionOp | goto.LabelOp,
) -> Iterator[str]:
    """Emit phi nodes for block args based on predecessor branches."""
    ctx = _ctx()
    preds = ctx.predecessors.get(id(op), [])
    if not preds:
        return
    for arg_idx, arg in enumerate(op.body.args):
        ty = llvm_type(arg.type)
        name = ctx.tracker.track_name(arg)
        phi_parts = [
            f"[ {value_reference(pred.args[arg_idx])}, %{pred.source_name} ]"
            for pred in preds
            if arg_idx < len(pred.args)
        ]
        if phi_parts:
            yield f"  %{name} = phi {ty} {', '.join(phi_parts)}"


@emitter_for(goto.RegionOp)
def emit_region_op(op: goto.RegionOp) -> Iterator[str]:
    """Region: executes inline in use-def order (fall-through entry).

    For regions with block args AND initial_arguments (loops): single block
    with phi nodes at entry.

    For regions with block args but NO initial_arguments (if-merge): the
    initial entry dispatches (no phi), and back-edges (from then/else) target
    the region name which has the phi. This emits as two blocks:
      {name}_entry: <body ops> (fall-through entry, dispatches)
      {name}: phi ... (merge point, entered only by branches)
    """
    ctx = _ctx()
    ctx.tracker.track_name(op)
    for arg in op.body.args:
        ctx.tracker.track_name(arg)
    for param in op.body.parameters:
        ctx.tracker.track_name(param)

    has_initial_args = bool(unpack(op.initial_arguments))
    has_merge_args = bool(op.body.args) and not has_initial_args

    if has_merge_args:
        # If-merge pattern: dispatch first, then merge block for phi.
        entry_name = f"{op.name}_entry"
        yield f"  br label %{entry_name}"
        yield f"{entry_name}:"
        terminated = yield from emit_linearized(op.body)
        # After dispatch (cond_br terminates), emit the merge block.
        yield f"{op.name}:"
        yield from _emit_phi_nodes(op)
    else:
        # Standard region (loop header, etc.): single block with phi.
        yield f"  br label %{op.name}"
        yield f"{op.name}:"
        yield from _emit_phi_nodes(op)
        terminated = yield from emit_linearized(op.body)

    # Emit exit labels for any "exit*" or "if_exit*" parameters.
    for param in op.body.parameters:
        if param.name and (
            param.name.startswith("exit") or param.name.startswith("if_exit")
        ):
            if not terminated:
                yield f"  br label %{param.name}"
            yield f"{param.name}:"
            yield from _emit_exit_phi_nodes(param)
            return
    return True


def _emit_exit_phi_nodes(
    param: dgen.Value,
) -> Iterator[str]:
    """Emit phi nodes for an exit parameter (branches target the param directly)."""
    ctx = _ctx()
    preds = ctx.predecessors.get(id(param), [])
    if not preds:
        return
    # Exit parameters carry values — emit phi for each arg position.
    # The first predecessor determines how many values to expect.
    n_args = len(preds[0].args) if preds else 0
    for arg_idx in range(n_args):
        # Create a synthetic name for the phi result.
        phi_name = f"{param.name}_phi{arg_idx}"
        ty = llvm_type(preds[0].args[arg_idx].type)
        phi_parts = [
            f"[ {value_reference(pred.args[arg_idx])}, %{pred.source_name} ]"
            for pred in preds
        ]
        if phi_parts:
            yield f"  %{phi_name} = phi {ty} {', '.join(phi_parts)}"


@emitter_for(goto.LabelOp)
def emit_label_op(op: goto.LabelOp) -> Iterator[str]:
    """Label: jump target only, not reachable by fall-through.

    Terminates the current basic block with a skip branch, emits the
    label body, then resumes with an exit label.
    """
    ctx = _ctx()
    ctx.tracker.track_name(op)
    for arg in op.body.args:
        ctx.tracker.track_name(arg)
    for param in op.body.parameters:
        ctx.tracker.track_name(param)
    exit_name = f"{op.name}_exit"
    # Skip over the label body in the current block's flow.
    yield f"  br label %{exit_name}"
    # Emit the label body as a separate basic block.
    yield f"{op.name}:"
    yield from _emit_phi_nodes(op)
    yield from emit_linearized(op.body)
    # Resume the enclosing block's flow.
    yield f"{exit_name}:"


@emitter_for(function.FunctionOp)
def emit_function_op(op: function.FunctionOp) -> Iterator[str]:
    ctx = _ctx()
    ctx.tracker.track_name(op)
    for arg in op.body.args:
        ctx.tracker.track_name(arg)
    ret_type = llvm_type(op.result_type)
    arguments = ", ".join(
        f"{llvm_type(arg.type)} %{ctx.tracker.track_name(arg)}" for arg in op.body.args
    )
    yield f"define {ret_type} @{op.name}({arguments}) {{"
    yield "entry:"
    terminated = yield from emit_linearized(op.body)
    if not terminated:
        if isinstance(op.result_type, builtin.Nil):
            yield "  ret void"
        else:
            result = op.body.result
            # If the result is a ChainOp, unwrap to get the actual value.
            if isinstance(result, builtin.ChainOp):
                result = result.lhs
            # If the result is a RegionOp with block args (e.g. if/else merge),
            # the value is the first block arg (the phi result), not the region.
            if isinstance(result, goto.RegionOp) and result.body.args:
                result = result.body.args[0]
            yield f"  ret {typed_reference(result)}"
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
    from dgen.block import BlockParameter

    if isinstance(v, Constant):
        mem = v.__constant__
        if not mem.layout.register_passable:
            ctx = _ctx()
            ctx.host_buffers.append(mem)
            return f"inttoptr (i64 {mem.address} to ptr)"
        raw = mem.unpack()[0]
        return f"{format_float(raw) if isinstance(raw, float) else raw}"
    if isinstance(v, builtin.ChainOp):
        return value_reference(v.lhs)
    # Resolve %self parameters to their owning RegionOp/LabelOp.
    # Only %self represents the label itself (for back-edges).
    # Exit parameters keep their own name (they're separate jump targets).
    if isinstance(v, BlockParameter) and v.name == "self":
        ctx = _ctx()
        owner = ctx.param_to_owner.get(id(v))
        if owner is not None:
            return f"%{ctx.tracker.track_name(owner)}"
    ctx = _ctx()
    name = ctx.tracker.track_name(v)
    return f"%{name}"


vr = value_reference


def typed_reference(*vs: dgen.Value) -> str:
    first, *_ = vs
    vrs = ", ".join(map(value_reference, vs))
    return f"{llvm_type(first.type)} {vrs}"


def typed_references(*vs: dgen.Value) -> str:
    return ", ".join(typed_reference(v) for v in vs)


# ---------------------------------------------------------------------------
# LLVM binary ops
# ---------------------------------------------------------------------------

_BINOP_EMITTERS: dict[type[dgen.Op], str] = {
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


def _emit_binop(op: dgen.Op) -> Iterator[str]:
    yield f"  {_BINOP_EMITTERS[type(op)]} {vr(op.lhs)}, {vr(op.rhs)}"


for _op_type in _BINOP_EMITTERS:
    EMITTERS[_op_type] = _emit_binop


# ---------------------------------------------------------------------------
# LLVM unary / cast ops
# ---------------------------------------------------------------------------


@emitter_for(llvm.FnegOp)
def emit_fneg(op: llvm.FnegOp) -> Iterator[str]:
    yield f"  fneg double {vr(op.input)}"


@emitter_for(llvm.ZextOp)
def emit_zext(op: llvm.ZextOp) -> Iterator[str]:
    yield f"  zext i1 {vr(op.input)} to i64"


# ---------------------------------------------------------------------------
# LLVM comparison ops
# ---------------------------------------------------------------------------


@emitter_for(llvm.IcmpOp)
def emit_icmp(op: llvm.IcmpOp) -> Iterator[str]:
    pred = string_value(op.pred)
    yield f"  icmp {pred} i64 {vr(op.lhs)}, {vr(op.rhs)}"


@emitter_for(llvm.FcmpOp)
def emit_fcmp(op: llvm.FcmpOp) -> Iterator[str]:
    pred = string_value(op.pred)
    yield f"  fcmp {pred} double {vr(op.lhs)}, {vr(op.rhs)}"


# ---------------------------------------------------------------------------
# LLVM memory ops
# ---------------------------------------------------------------------------


@emitter_for(llvm.AllocaOp)
def emit_alloca(op: llvm.AllocaOp) -> Iterator[str]:
    yield f"  alloca double, i64 {op.elem_count.__constant__.to_json()}"


@emitter_for(llvm.GepOp)
def emit_gep(op: llvm.GepOp) -> Iterator[str]:
    yield f"  getelementptr double, ptr {vr(op.base)}, {typed_reference(op.index)}"


@emitter_for(memory.StoreOp)
def emit_store(op: memory.StoreOp) -> Iterator[str]:
    yield f"  store {typed_reference(op.value)}, {typed_reference(op.ptr)}"


@emitter_for(memory.LoadOp)
def emit_load(op: memory.LoadOp) -> Iterator[str]:
    yield f"  load {llvm_type(op.type)}, {typed_reference(op.ptr)}"


# ---------------------------------------------------------------------------
# Call ops
# ---------------------------------------------------------------------------


@emitter_for(function.CallOp)
def emit_function_call(op: function.CallOp) -> Iterator[str]:
    args = ", ".join(typed_reference(v) for v in unpack(op.arguments))
    callee = op.callee.name
    ret = llvm_type(op.type)
    yield f"  call {ret} @{callee}({args})"


@emitter_for(llvm.CallOp)
def emit_llvm_call(op: llvm.CallOp) -> Iterator[str]:
    args = ", ".join(typed_reference(v) for v in unpack(op.args))
    callee = string_value(op.callee)
    ret = llvm_type(op.type)
    yield f"  call {ret} @{callee}({args})"


# ---------------------------------------------------------------------------
# Branch ops
# ---------------------------------------------------------------------------


@emitter_for(goto.ConditionalBranchOp)
def emit_conditional_branch(op: goto.ConditionalBranchOp) -> Iterator[str]:
    cond_type = llvm_type(op.condition.type)
    cond = vr(op.condition)
    if cond_type != "i1":
        # Convert non-i1 condition to i1 via icmp ne.
        zero = "null" if cond_type == "ptr" else "0"
        yield f"  %_cond = icmp ne {cond_type} {cond}, {zero}"
        cond = "%_cond"
    yield f"  br i1 {cond}, {typed_references(op.true_target, op.false_target)}"


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

    def _emit_op(op: dgen.Op) -> Iterator[str]:
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
    from dgen.passes.control_flow_to_goto import ControlFlowToGoto

    _dummy = Compiler([], IdentityPass())
    module = ControlFlowToGoto().run(module, _dummy)
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
