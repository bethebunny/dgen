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
from dgen.compiler import Compiler, IdentityPass
from dgen.dialects import (
    builtin,
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


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def _externs(module: Module) -> list[builtin.ExternOp]:
    """Discover extern declarations from ExternOps in the module, deduped by symbol."""
    seen: dict[str, builtin.ExternOp] = {}
    for top_level in module.ops:
        for value in all_values(top_level):
            if isinstance(value, builtin.ExternOp):
                sym = string_value(value.symbol)
                if sym not in seen:
                    seen[sym] = value
    return list(seen.values())


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
            yield ""
            for func in module.functions:
                build_predecessors(func, ctx)
                yield from emit(func)

        return "\n".join(_lines()), ctx.host_buffers
    finally:
        _emit_ctx.reset(token)


def unpack(val: Value) -> list[Value]:
    if isinstance(val, builtin.ChainOp):
        return unpack(val.lhs)
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
    # Structural and noop ops handle their own output.
    if isinstance(value, _NO_ASSIGN_OPS):
        yield from lines
    else:
        # Prepend %name = for value-producing ops.
        ctx = _ctx()
        name = ctx.tracker.track_name(value)
        first_line = next(lines, None)
        if first_line is None:
            return
        # If the line already contains an assignment or is a void instruction
        # (e.g. "  call void @..." or "  store ..."), emit as-is.
        stripped = first_line.lstrip()
        if stripped.startswith("call void") or stripped.startswith("store"):
            yield first_line
        else:
            yield f"  %{name} = {stripped}"
        yield from lines
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
        if ty == "void":
            continue
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
            yield f"  ret {ret_type} {value_reference(result)}"
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
        arg_types = ", ".join(llvm_type(arg) for arg in extern.type.arguments)
        yield f"declare {result_type} @{sym}({arg_types})"
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
    callee = op.callee
    if isinstance(callee, builtin.ExternOp):
        callee_name = string_value(callee.symbol)
    else:
        callee_name = callee.name
    ret = llvm_type(op.type)
    yield f"  call {ret} @{callee_name}({args})"


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
    callback_extern = builtin.ExternOp(
        symbol=String().constant(callback_name),
        type=Function(
            arguments=pack(arg.type for arg in thunk_args), result_type=result_type
        ),
    )
    call_op = function.CallOp(
        callee=callback_extern,
        arguments=pack(thunk_args),
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
