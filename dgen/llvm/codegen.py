"""Emit valid LLVM IR text from a Value and JIT-compile via llvmlite."""

from __future__ import annotations

import contextvars
import ctypes
import functools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import llvmlite.binding as llvmlite

import dgen
from dgen import Type
from dgen.block import BlockArgument
from dgen.dialects import (
    builtin,
    function,
    goto,
    llvm,
    memory,
    number,
)
from dgen.ir.traversal import all_values
from dgen.layout import Layout
from dgen.builtins import ConstantOp, PackOp, _aggregate_elements, pack
from dgen.dialects.builtin import String
from dgen.dialects.function import Function
from dgen.memory import Memory
from dgen.type import Constant, TypeType, Value, constant

# ---------------------------------------------------------------------------
from dgen.llvm import ffi


def _ffi_ctype(layout: Layout) -> type[ctypes._CData]:
    """ctypes type for a layout, normalizing non-register-passable to c_void_p."""
    return ctypes.c_void_p if not layout.register_passable else ffi.ctype(layout)


def llvm_type(t: dgen.Value[TypeType]) -> str:
    resolved = constant(t)
    # Types whose declared bit width differs from their layout width.
    match resolved:
        case number.SignedInteger(bits) | number.UnsignedInteger(bits):
            declared = bits.__constant__.to_json()
            layout_bits = resolved.__layout__.struct.size * 8
            return f"i{max(declared, layout_bits)}"
        case goto.Label():
            return "label"
        case _:
            pass
    if isinstance(resolved, llvm.Int):
        return f"i{resolved.bits.__constant__.to_json()}"
    layout = resolved.__layout__
    return "ptr" if not layout.register_passable else ffi.llvm_type(layout)


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


def _externs(root: dgen.Value) -> dict[str, builtin.ExternOp]:
    """Discover extern declarations reachable from root, deduped by symbol."""
    return {
        constant(v.symbol): v
        for v in all_values(root)
        if isinstance(v, builtin.ExternOp)
    }


def emit_llvm_ir(root: dgen.Value) -> tuple[str, list[Memory]]:
    """Emit LLVM IR text for root and every function reachable from it.

    Returns (ir_text, host_buffers) where host_buffers keeps Memory objects
    alive for the lifetime of the JIT engine.
    """

    def _lines() -> Iterator[str]:
        funcs = [v for v in all_values(root) if isinstance(v, function.FunctionOp)]
        defined = {f.name for f in funcs}
        for extern in _externs(root).values():
            sym = constant(extern.symbol)
            # Skip ExternOp if a FunctionOp with the same name
            # provides the definition (avoids duplicate symbols).
            if sym in defined:
                continue
            yield from emit_extern(extern)
        yield ""
        for func in funcs:
            prepare_function(func, ctx)
            yield from emit(func)

    ctx = EmitContext()
    token = _emit_ctx.set(ctx)
    try:
        return "\n".join(_lines()), ctx.host_buffers
    finally:
        _emit_ctx.reset(token)


# runtime_dependencies follows operands and block captures — NOT types
# or parameters. This means branch targets (which are parameters) are
# not followed, so emit_linearized won't descend into label bodies
# through branch ops. That's correct:
# - RegionOps (inline regions) appear as runtime deps of their consumers
#   and emit themselves via fall-through.
# - LabelOps (jump targets) are emitted by the region/function that
#   contains them, not by the branch that targets them.
def runtime_dependencies(
    value: dgen.Value, *, stop: Iterable[dgen.Value] = ()
) -> Iterator[dgen.Value]:
    """Walk operand- and capture-edge dependencies of *value* in topo order.

    Doesn't follow type / parameter edges — those are compile-time
    metadata. ``stop`` is a set of values to halt at (not yielded, not
    descended into); callers use this to honour block boundaries (e.g.
    ``block.captures``: captures live in outer scope and are emitted
    there, not inside the block).
    """
    seen: set[dgen.Value] = set(stop)

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
    sequential textual order. Using %_0, %_1, ... sidesteps that, so names
    can be allocated lazily during emission rather than pre-registered.
    """

    def __init__(self) -> None:
        self._slots: dict[dgen.Value, str] = {}
        self._used: set[str] = set()
        self._counter = 0

    def name(self, value: dgen.Value) -> str:
        """Stable LLVM SSA name for *value*. Idempotent: same value → same name."""
        if value in self._slots:
            return self._slots[value]
        name = self.fresh(prefix=value.name)
        self._slots[value] = name
        return name

    def fresh(self, prefix: str | None = None) -> str:
        """A unique SSA name. If *prefix* is given and free, returns it as-is;
        otherwise returns ``{prefix}{counter}`` (with ``_`` as the fallback
        base when no prefix is given, since LLVM requires bare-numeric names
        to appear in sequential textual order)."""
        if prefix and prefix not in self._used:
            self._used.add(prefix)
            return prefix
        base = prefix or "_"
        while True:
            name = f"{base}{self._counter}"
            self._counter += 1
            if name not in self._used:
                self._used.add(name)
                return name


@dataclass
class Predecessor:
    """One incoming edge to a label/region block.

    ``tuple`` is the single value flowing into the target — either a
    ``PackOp`` constructed inline (``[%a, %b]``, which emits as an
    ``insertvalue`` chain producing an aggregate) or any other value
    whose runtime shape matches the consumer's expected aggregate
    (e.g. a ``Tuple`` returned by a function call). Codegen materialises
    the target's body args via aggregate ``phi`` + ``extractvalue`` per
    arg.
    """

    source_name: str  # LLVM basic block name where the branch originates
    tuple: dgen.Value


@dataclass
class EmitContext:
    """Shared state for the emitter path."""

    tracker: CodegenSlotTracker = field(default_factory=CodegenSlotTracker)
    host_buffers: list[Memory] = field(default_factory=list)
    predecessors: dict[dgen.Value, list[Predecessor]] = field(default_factory=dict)
    param_to_owner: dict[dgen.Value, goto.RegionOp | goto.LabelOp] = field(
        default_factory=dict
    )
    self_params: set[dgen.Value] = field(default_factory=set)


def _llvm_fields(arg_types: list) -> list[tuple[int, int, str]]:
    """Filter a tuple's element types down to its LLVM-visible fields.

    Returns ``[(orig_i, llvm_i, llvm_ty), ...]`` — one entry per non-void
    element, with ``orig_i`` the position in the original list and
    ``llvm_i`` the position in the resulting LLVM struct. ``Nil``-typed
    elements drop out (LLVM has no ``void`` aggregate field), so a
    ``pack([%nil, %int])`` becomes a single-field struct with ``llvm_i``
    0 pointing at ``orig_i`` 1.
    """
    fields: list[tuple[int, int, str]] = []
    for orig_i, t in enumerate(arg_types):
        ty = llvm_type(t)
        if ty == "void":
            continue
        fields.append((orig_i, len(fields), ty))
    return fields


def _tuple_llvm_type(arg_types: list) -> str:
    """Internal aggregate LLVM type ``{ T1, ..., Tn }`` for any n ≥ 1
    LLVM-visible fields, ``{}`` when every field is ``void``.

    Codegen treats *every* tuple-shaped value (PackOp, runtime aggregate,
    aggregate Constant) uniformly through this helper, so insertvalue /
    extractvalue / phi all see the same ``{ ... }`` shape regardless of
    arity. The FFI/ABI collapse for single-field layouts (``ffi.llvm_type``,
    ``ffi.llvm_constant``) lives only at function boundaries and is
    bridged by the few callers that emit ``define`` / ``declare`` /
    ``call`` / ``ret``.
    """
    fields = [llvm_type(t) for t in arg_types if llvm_type(t) != "void"]
    if not fields:
        return "{}"
    return "{ " + ", ".join(fields) + " }"


def _resolve_target(target: dgen.Value, ctx: EmitContext) -> dgen.Value:
    """Resolve a branch target to the label/region it refers to.

    Self parameters map to their owning RegionOp/LabelOp.
    Direct RegionOp/LabelOp references pass through.
    """
    owner = ctx.param_to_owner.get(target)
    return owner if owner is not None else target


def prepare_function(func: function.FunctionOp, ctx: EmitContext) -> None:
    """Record branch predecessors and target ownership for a function.

    Single walk over the block tree that:
    - Records param_to_owner for branch target resolution
    - Populates self_params for self-parameter identification
    - Tracks the current LLVM basic block name through label/region nesting
    - Records branch predecessors with their source block names

    SSA names are not pre-registered: ``tracker.name`` is idempotent and
    allocated lazily during emission.
    """

    def _record_branch(target: dgen.Value, t: dgen.Value, current_block: str) -> None:
        resolved = _resolve_target(target, ctx)
        ctx.predecessors.setdefault(resolved, []).append(
            Predecessor(source_name=current_block, tuple=t)
        )

    def _walk(block: dgen.Block, current_block: str) -> None:
        for op in block.ops:
            if isinstance(op, goto.BranchOp):
                _record_branch(op.target, op.arguments, current_block)
            elif isinstance(op, goto.ConditionalBranchOp):
                _record_branch(op.true_target, op.true_arguments, current_block)
                _record_branch(op.false_target, op.false_arguments, current_block)

            if isinstance(op, (goto.RegionOp, goto.LabelOp)):
                # Fall-through entry as a predecessor — but only for
                # RegionOps. Regions execute inline (their body is
                # entered by falling through from the surrounding
                # block); ``initial_arguments`` is the tuple of values
                # that feed into the body's args via phi at body entry.
                # LabelOps are jump-only (the surrounding code emits a
                # ``br label %{exit}`` to skip them), so there's no
                # fall-through tuple — every value reaching a label's
                # body args comes from an explicit branch.
                if isinstance(op, goto.RegionOp) and op.body.args:
                    ctx.predecessors.setdefault(op, []).append(
                        Predecessor(
                            source_name=current_block, tuple=op.initial_arguments
                        )
                    )

                if isinstance(op, goto.RegionOp):
                    # Regions always declare [%self, %exit]. %self maps to
                    # the owner (back-edges resolve to the body block);
                    # %exit stays out of param_to_owner so branch<%exit>
                    # records its predecessors against exit_param itself —
                    # that's where the merge phi lives.
                    self_param, exit_param = op.body.parameters
                    ctx.param_to_owner[self_param] = op
                    ctx.self_params.add(self_param)
                    _walk(op.body, op.name)
                    current_block = ctx.tracker.name(exit_param)
                else:
                    # Labels declare no parameters; they capture the
                    # enclosing region's %self/%exit when needed.
                    _walk(op.body, op.name)
                    current_block = f"{op.name}_exit"

    _walk(func.body, "entry")


_emit_ctx: contextvars.ContextVar[EmitContext] = contextvars.ContextVar("_emit_ctx")


def _ctx() -> EmitContext:
    try:
        return _emit_ctx.get()
    except LookupError:
        ctx = EmitContext()
        _emit_ctx.set(ctx)
        return ctx


# Ops whose results are structural / have no LLVM SSA value, or whose
# emitter manages its own ``%name = `` assignment (because the emission
# spans multiple lines starting with a prologue, e.g. ``extractvalue``
# ops materialising args from a bundle before the actual call).
_NO_ASSIGN_OPS: tuple[type[dgen.Op], ...] = (
    goto.RegionOp,
    goto.LabelOp,
    function.FunctionOp,
    function.CallOp,
    goto.BranchOp,
    goto.ConditionalBranchOp,
    memory.StoreOp,
    PackOp,
    ConstantOp,
    builtin.ChainOp,
    builtin.ExternOp,
    llvm.CallOp,
)


def emit_linearized(block: dgen.Block) -> Iterator[str]:
    """Emit LLVM IR for the runtime ops of a block, in dependency order."""
    for op in block.ops:
        # FunctionOps and ExternOps reachable as operands (e.g. a call's
        # callee) are top-level declarations, not block instructions.
        if isinstance(op, (function.FunctionOp, builtin.ExternOp)):
            continue
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
        name = ctx.tracker.name(value)
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


def _pred_block_name(pred: Predecessor) -> str:
    """Return the LLVM basic block name where this predecessor branch lives."""
    return pred.source_name


def _emit_tuple_phi(target: dgen.Value, name: str, tuple_ty: str) -> Iterator[str]:
    """Emit a single aggregate phi at *target* merging predecessor tuples.

    Every predecessor contributes its tuple value (a PackOp's
    ``insertvalue`` chain result, or a runtime aggregate like a
    ``Tuple`` returned by another call). The phi merges those aggregates;
    consumers ``extractvalue`` individual fields downstream.
    """
    preds = _ctx().predecessors.get(target, [])
    phi_parts = [
        f"[ {value_reference(pred.tuple)}, %{_pred_block_name(pred)} ]"
        for pred in preds
    ]
    if phi_parts:
        yield f"  %{name} = phi {tuple_ty} {', '.join(phi_parts)}"


def _emit_block_arg_phis(op: goto.RegionOp | goto.LabelOp) -> Iterator[str]:
    """Materialise ``op.body.args`` from predecessor tuples.

    Emits an aggregate phi merging the tuple from each predecessor,
    followed by one ``extractvalue`` per non-void body arg. ``Nil``-typed
    body args have no LLVM representation and are skipped (consumers
    treat them as memory tokens, not real values). LLVM's
    ``mem2reg``/``sroa`` collapses the insertvalue/phi/extractvalue
    chain back into scalar phis after opt, so the optimised IR matches
    the pre-aggregate shape.
    """
    args = op.body.args
    fields = _llvm_fields([arg.type for arg in args])
    if not fields:
        return
    tracker = _ctx().tracker
    tuple_ty = _tuple_llvm_type([arg.type for arg in args])
    tuple_name = tracker.fresh()
    yield from _emit_tuple_phi(op, tuple_name, tuple_ty)
    for orig_i, llvm_i, _ in fields:
        yield (
            f"  %{tracker.name(args[orig_i])} = extractvalue "
            f"{tuple_ty} %{tuple_name}, {llvm_i}"
        )


def _emit_exit_phi(op: goto.RegionOp, name: str, ty: str) -> Iterator[str]:
    """Exit phi: merge predecessor tuples (each a 1-element pack of the
    region's result) into the region's named result.

    The predecessor tuples have aggregate LLVM type ``{ T }`` (per
    ``_tuple_llvm_type``), so we phi an aggregate then ``extractvalue``
    field 0 to bind the named scalar/inner value that consumers see via
    ``value_reference(op)``. The FFI/ABI scalar-collapse for single-field
    layouts is bridged here.
    """
    if ty == "void":
        return
    exit_param = op.body.parameters[1]
    tuple_ty = _tuple_llvm_type([op.type])
    tracker = _ctx().tracker
    tuple_name = tracker.fresh()
    yield from _emit_tuple_phi(exit_param, tuple_name, tuple_ty)
    yield f"  %{name} = extractvalue {tuple_ty} %{tuple_name}, 0"


def _exit_phi_name(op: goto.RegionOp) -> str:
    """The LLVM SSA name for the exit phi of *op* — what
    ``value_reference(op)`` resolves to. Stable, deterministic; lets
    consumers reference the region's value without inspecting block args.
    """
    return f"{op.name}_result"


@emitter_for(goto.RegionOp)
def emit_region_op(op: goto.RegionOp) -> Iterator[str]:
    """Region: executes inline in use-def order. One uniform shape.

    The body lives at ``%{op.name}:``. Its entry has a phi if and only
    if the body has block args (and predecessors carrying values for
    them). After the body, ``%{exit_param.name}:`` opens; that block
    has a phi if branches to %exit carried values, named after
    :func:`_exit_phi_name`. Codegen has no notion of "is this a loop" —
    each label gets the phi nodes its predecessors call for.

    The "is region.type Nil" question lives entirely here: emit the
    exit phi only when there's a non-Nil result type to phi for. LLVM
    has no void phi.
    """
    _self_param, exit_param = op.body.parameters
    exit_name = _ctx().tracker.name(exit_param)
    yield f"  br label %{op.name}"
    yield f"{op.name}:"
    yield from _emit_block_arg_phis(op)
    terminated = yield from emit_linearized(op.body)
    if not terminated:
        yield f"  br label %{exit_name}"
    yield f"{exit_name}:"
    if not isinstance(constant(op.type), builtin.Nil):
        yield from _emit_exit_phi(op, _exit_phi_name(op), llvm_type(op.type))


@emitter_for(goto.LabelOp)
def emit_label_op(op: goto.LabelOp) -> Iterator[str]:
    """Label: jump target only, not reachable by fall-through.

    Terminates the current basic block with a skip branch, emits the
    label body, then resumes with an exit label.
    """
    exit_name = f"{op.name}_exit"
    yield f"  br label %{exit_name}"
    yield f"{op.name}:"
    yield from _emit_block_arg_phis(op)
    yield from emit_linearized(op.body)
    yield f"{exit_name}:"


@emitter_for(function.FunctionOp)
def emit_function_op(op: function.FunctionOp) -> Iterator[str]:
    ctx = _ctx()
    ret_type = llvm_type(op.result_type)
    arguments = ", ".join(
        f"{llvm_type(arg.type)} %{ctx.tracker.name(arg)}" for arg in op.body.args
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
            yield f"  ret {ret_type} {value_reference(result)}"
    yield "}"


@emitter_for(ConstantOp)
@emitter_for(builtin.ChainOp)
@emitter_for(builtin.ExternOp)
def noop(op: dgen.Op) -> Iterator[str]:
    return ()


@emitter_for(PackOp)
def emit_pack_op(op: PackOp) -> Iterator[str]:
    """Build the tuple's LLVM aggregate via an ``insertvalue`` chain.

    Skips ``Nil``-typed elements (LLVM has no ``void`` aggregate field).
    A pack with no LLVM-visible fields is a no-op (no aggregate value
    to produce). All other arities — including n=1 — emit a uniform
    ``insertvalue`` chain producing a ``{ T1, ..., Tn }`` aggregate.
    """
    fields = _llvm_fields([v.type for v in op.values])
    if not fields:
        return
    ctx = _ctx()
    tracker = ctx.tracker
    tuple_ty = _tuple_llvm_type([v.type for v in op.values])
    op_name = tracker.name(op)
    cur = "undef"
    last = len(fields) - 1
    for k, (orig_i, llvm_i, _) in enumerate(fields):
        target = op_name if k == last else tracker.fresh()
        yield (
            f"  %{target} = insertvalue {tuple_ty} {cur}, "
            f"{typed_reference(op.values[orig_i])}, {llvm_i}"
        )
        cur = f"%{target}"


def emit_extern(extern: builtin.ExternOp) -> Iterator[str]:
    sym = constant(extern.symbol)
    if isinstance(extern.type, function.Function):
        result_type = llvm_type(extern.type.result_type)
        arg_type_list = constant(extern.type.arguments)
        assert isinstance(arg_type_list, list)
        arg_types = ", ".join(llvm_type(t) for t in arg_type_list)
        yield f"declare {result_type} @{sym}({arg_types})"
    else:
        # Global variable: `@name = external global <type>`. LLVM treats
        # the symbol itself as a pointer; the value must be loaded.
        ty = llvm_type(extern.type)
        yield f"@{sym} = external global {ty}"


def _aggregate_constant_literal(c: Constant) -> str:
    """LLVM literal ``{ T1 v1, T2 v2, ... }`` for an aggregate Constant.

    Walks each element via ``_aggregate_elements`` (each yielded element
    is itself a ``Value`` of its field type — Constants for scalars,
    nested aggregate Constants for nested tuples/arrays) and recursively
    builds the typed reference. ``Nil``-typed fields drop out (LLVM has
    no ``void`` aggregate field), matching ``_llvm_fields``. An aggregate
    with no LLVM-visible fields formats as ``{}``.
    """
    parts = [
        typed_reference(e)
        for e in _aggregate_elements(c)
        if llvm_type(e.type) != "void"
    ]
    if not parts:
        return "{}"
    return "{ " + ", ".join(parts) + " }"


def value_reference(v: dgen.Value) -> str:
    if isinstance(v, Constant):
        mem = v.__constant__
        ctx = _ctx()
        # Track the Memory whenever it owns heap-allocated descriptor
        # bytearrays (``mem.origins``) — even register-passable layouts
        # like TypeValue (``"P"``) bake a raw pointer into the LLVM IR
        # whose target is a Python-managed ``bytearray`` referenced only
        # via ``mem.origins``. Without keeping ``mem`` alive past IR-text
        # generation, that bytearray is GC'd and the pointer dangles.
        if not mem.layout.register_passable or mem.origins:
            ctx.host_buffers.append(mem)
        if not mem.layout.register_passable:
            return f"inttoptr (i64 {mem.address} to ptr)"
        # Aggregate Constants build their LLVM literal element-by-element
        # so the form matches ``_tuple_llvm_type`` (uniform ``{ T1, ... }``
        # shape — no scalar collapse). ``ffi.llvm_constant`` would emit
        # the FFI/ABI-collapsed scalar form for single-field layouts,
        # which mismatches the internal aggregate type used elsewhere.
        if isinstance(v.type, (builtin.Array, builtin.Tuple)):
            return _aggregate_constant_literal(v)
        return ffi.llvm_constant(bytes(mem.buffer), mem.layout)
    if isinstance(v, builtin.ChainOp):
        return value_reference(v.lhs)
    if isinstance(v, builtin.ExternOp):
        sym = constant(v.symbol)
        if isinstance(v.type, function.Function):
            return f"@{sym}"
        # Global variable: the LLVM symbol is a pointer to the value.
        # Emit an inline load to dereference it.
        ty = llvm_type(v.type)
        return f"load {ty}, ptr @{sym}"
    if isinstance(v, function.FunctionOp) and v.name:
        return f"@{v.name}"
    # PackOp emits as an ``insertvalue`` chain whose final SSA name is
    # the PackOp's tracker name (regardless of arity). An empty pack has
    # no LLVM value at all — referencing one is a usage error.
    if isinstance(v, PackOp) and not v.values:
        raise ValueError("value_reference: empty PackOp has no LLVM value to reference")
    # A region's value is the exit phi (named by ``_exit_phi_name``) when
    # its type is non-Nil. Nil-typed regions don't produce a value at the
    # LLVM level — referencing one is a usage error.
    if isinstance(v, goto.RegionOp):
        assert not isinstance(constant(v.type), builtin.Nil), (
            f"value_reference: region {v.name!r} has Nil type and produces "
            f"no LLVM value"
        )
        return f"%{_exit_phi_name(v)}"
    # Self parameters resolve to their owning RegionOp/LabelOp name
    # (the label target for back-edges). Exit parameters keep their own
    # name — they are separate LLVM basic blocks.
    ctx = _ctx()
    if v in ctx.self_params:
        owner = ctx.param_to_owner[v]
        return f"%{ctx.tracker.name(owner)}"
    name = ctx.tracker.name(v)
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

_FLOAT_BINOPS: dict[type[dgen.Op], str] = {
    llvm.FaddOp: "fadd double",
    llvm.FsubOp: "fsub double",
    llvm.FmulOp: "fmul double",
    llvm.FdivOp: "fdiv double",
}

_INT_BINOPS: dict[type[dgen.Op], str] = {
    llvm.AddOp: "add",
    llvm.SubOp: "sub",
    llvm.MulOp: "mul",
    llvm.SdivOp: "sdiv",
    llvm.AndOp: "and",
    llvm.OrOp: "or",
    llvm.XorOp: "xor",
}

_BINOP_EMITTERS: dict[type[dgen.Op], str] = {**_FLOAT_BINOPS, **_INT_BINOPS}


def _emit_binop(op: dgen.Op) -> Iterator[str]:
    if type(op) in _FLOAT_BINOPS:
        yield f"  {_FLOAT_BINOPS[type(op)]} {vr(op.lhs)}, {vr(op.rhs)}"
    else:
        ty = llvm_type(op.type)
        yield f"  {_INT_BINOPS[type(op)]} {ty} {vr(op.lhs)}, {vr(op.rhs)}"


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
    ty = llvm_type(op.type)
    yield f"  zext i1 {vr(op.input)} to {ty}"


@emitter_for(llvm.PtrtointOp)
def emit_ptrtoint(op: llvm.PtrtointOp) -> Iterator[str]:
    ty = llvm_type(op.type)
    yield f"  ptrtoint ptr {vr(op.input)} to {ty}"


@emitter_for(llvm.InttoptrOp)
def emit_inttoptr(op: llvm.InttoptrOp) -> Iterator[str]:
    src_ty = llvm_type(op.input.type)
    yield f"  inttoptr {src_ty} {vr(op.input)} to ptr"


# ---------------------------------------------------------------------------
# LLVM comparison ops
# ---------------------------------------------------------------------------


@emitter_for(llvm.IcmpOp)
def emit_icmp(op: llvm.IcmpOp) -> Iterator[str]:
    pred = constant(op.pred)
    ty = llvm_type(op.lhs.type)
    yield f"  icmp {pred} {ty} {vr(op.lhs)}, {vr(op.rhs)}"


@emitter_for(llvm.FcmpOp)
def emit_fcmp(op: llvm.FcmpOp) -> Iterator[str]:
    pred = constant(op.pred)
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
    yield f"  store {typed_reference(op.value)}, ptr {vr(op.ptr)}"


@emitter_for(memory.LoadOp)
def emit_load(op: memory.LoadOp) -> Iterator[str]:
    yield f"  load {llvm_type(op.type)}, ptr {vr(op.ptr)}"


@emitter_for(llvm.ExtractValueOp)
def emit_extract_value(op: llvm.ExtractValueOp) -> Iterator[str]:
    index = constant(op.index)
    yield f"  extractvalue {typed_reference(op.aggregate)}, {index}"


# ---------------------------------------------------------------------------
# Call ops
# ---------------------------------------------------------------------------


def _extract_call_args(tup: dgen.Value, arg_types: list) -> tuple[list[str], str]:
    """Materialise positional call args from a tuple value.

    Always extracts via ``extractvalue``: the tuple may be an inline
    PackOp (whose ``insertvalue`` chain produces the aggregate) or a
    runtime aggregate (a ``Tuple`` returned by another call). LLVM
    folds the round-trip in either case. ``Nil``-typed arg slots are
    skipped (no LLVM value to extract or pass).

    Returns ``(prologue, args_str)`` — prologue is the ``extractvalue``
    ops emitted before the call, ``args_str`` is the comma-joined typed
    argument list ready to slot into ``call ret @f(...)``.
    """
    fields = _llvm_fields(arg_types)
    if not fields:
        return [], ""
    tracker = _ctx().tracker
    tuple_ty = _tuple_llvm_type(arg_types)
    tuple_ref = value_reference(tup)
    prologue: list[str] = []
    typed_args: list[str] = []
    for _, llvm_i, ty in fields:
        name = tracker.fresh()
        prologue.append(f"  %{name} = extractvalue {tuple_ty} {tuple_ref}, {llvm_i}")
        typed_args.append(f"{ty} %{name}")
    return prologue, ", ".join(typed_args)


@emitter_for(function.CallOp)
def emit_function_call(op: function.CallOp) -> Iterator[str]:
    callee_type = op.callee.type
    assert isinstance(callee_type, function.Function)
    arg_types = constant(callee_type.arguments)
    assert isinstance(arg_types, list)
    prologue, args = _extract_call_args(op.arguments, arg_types)
    yield from prologue
    callee = op.callee
    ret = llvm_type(op.type)
    if isinstance(callee, builtin.ExternOp):
        callee_ref = f"@{constant(callee.symbol)}"
    elif isinstance(callee, function.FunctionOp):
        callee_ref = f"@{callee.name}"
    else:
        # Function-pointer call — callee is an arbitrary SSA value.
        callee_ref = value_reference(callee)
    if ret == "void":
        yield f"  call void {callee_ref}({args})"
    else:
        name = _ctx().tracker.name(op)
        yield f"  %{name} = call {ret} {callee_ref}({args})"


@emitter_for(llvm.CallOp)
def emit_llvm_call(op: llvm.CallOp) -> Iterator[str]:
    # llvm.call carries no callee signature in its IR; derive arg types
    # from the tuple itself. PackOp args read each ``.values[i].type``;
    # a runtime aggregate uses the tuple's type parameters.
    arg_types = _tuple_arg_types(op.args)
    prologue, args = _extract_call_args(op.args, arg_types)
    yield from prologue
    callee = constant(op.callee)
    ret = llvm_type(op.type)
    if ret == "void":
        yield f"  call void @{callee}({args})"
    else:
        name = _ctx().tracker.name(op)
        yield f"  %{name} = call {ret} @{callee}({args})"


def _tuple_arg_types(tup: dgen.Value) -> list:
    """Per-element types for a tuple value. PackOp reads each
    ``.values[i].type``; a runtime aggregate reads its type's parameters."""
    if isinstance(tup, PackOp):
        return [v.type for v in tup.values]
    tup_type = constant(tup.type)
    if isinstance(tup_type, builtin.Tuple):
        types_param = constant(tup_type.types)
        assert isinstance(types_param, list)
        return types_param
    if isinstance(tup_type, builtin.Array):
        n = constant(tup_type.n)
        assert isinstance(n, int)
        return [tup_type.element_type] * n
    raise TypeError(
        f"Cannot derive tuple element types from {tup_type!r}; expected "
        f"PackOp / Tuple / Array shape"
    )


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
    """Compiled LLVM IR ready for JIT execution.

    Thin wrapper around a ConstantOp[Function] (the JIT'd function pointer).
    `run()` delegates to `call()`. Kept for backwards compat with callsites
    that want an Executable-shaped object; new code should prefer passing
    a ConstantOp[Function] around directly and invoking via `call()`.
    """

    ir: str
    input_types: list[Type]
    result_type: Type
    main_name: str
    host_refs: list[object] = field(default_factory=list)

    @functools.cached_property
    def func_constant(self) -> ConstantOp:
        """JIT-compile and return a ConstantOp[Function] for this executable."""
        func_type = function.Function(
            arguments=pack(self.input_types),
            result_type=self.result_type,
        )
        return jit_function(func_type, self.main_name, self.ir, self.host_refs)

    def run(self, *args: Memory | object) -> Memory:
        """JIT and execute, returning the result as a Memory object.

        Args may be Memory objects or raw Python values (int, float, etc.),
        which are converted via Memory.from_value.
        """
        return call(self.func_constant, *args)


def _jit_engine(exe: Executable) -> Any:  # noqa: ANN401
    """Parse, verify, and create an MCJIT engine from an Executable."""
    _ensure_initialized()
    mod = llvmlite.parse_assembly(exe.ir)
    mod.verify()
    target = llvmlite.Target.from_default_triple()
    tm = target.create_target_machine()
    return llvmlite.create_mcjit_compiler(mod, tm)


# ---------------------------------------------------------------------------
# Function values: Constant[Function] + call()
# ---------------------------------------------------------------------------


def _function_arg_types(func_type: function.Function) -> list[Type]:
    """Extract the runtime Type instances for each argument of a Function type."""
    args = constant(func_type.arguments)
    assert isinstance(args, list)
    return args


def jit_function(
    func_type: function.Function, symbol: str, ir: str, host_refs: list
) -> ConstantOp:
    """JIT-compile `ir` and return a ConstantOp[Function] holding the function pointer.

    `symbol` is the entry function's name in the IR. The returned constant's
    Memory buffer contains an 8-byte function pointer; its origins keep the
    JIT engine (and any other supplied refs) alive.
    """
    _ensure_initialized()
    llvm_mod = llvmlite.parse_assembly(ir)
    llvm_mod.verify()
    target = llvmlite.Target.from_default_triple()
    tm = target.create_target_machine()
    engine = llvmlite.create_mcjit_compiler(llvm_mod, tm)
    func_ptr = engine.get_function_address(symbol)
    mem: Memory = Memory(func_type)
    # Pack the raw pointer into the 8-byte buffer (Pointer layout = "P").
    mem.layout.struct.pack_into(mem.buffer, 0, func_ptr)
    mem.origins = [engine, *host_refs]
    return ConstantOp(type=func_type, value=mem)


def call(func_constant: Value, *args: Memory | object) -> Memory:
    """Invoke a JIT-compiled function value.

    `func_constant` is a Value whose type is a Function (typically a
    ConstantOp[Function] produced by `jit_function`). Its Memory holds the
    raw function pointer. Args may be Memory objects or raw Python values.
    """
    func_type = func_constant.type
    assert isinstance(func_type, function.Function)
    input_types = _function_arg_types(func_type)
    result_type = constant(func_type.result_type)
    func_mem = func_constant.__constant__
    (func_ptr,) = func_mem.layout.struct.unpack(func_mem.buffer)

    memories: list[Memory] = [
        arg if isinstance(arg, Memory) else Memory.from_value(ty, arg)
        for arg, ty in zip(args, input_types)
    ]
    cfunc_type = ctypes.CFUNCTYPE(
        _ffi_ctype(result_type.__layout__),
        *(_ffi_ctype(t.__layout__) for t in input_types),
    )
    cfunc = cfunc_type(func_ptr)
    raw_args = [
        ffi.to_ffi(t.__layout__, m.buffer)
        if t.__layout__.register_passable
        else m.address
        for m, t in zip(memories, input_types)
    ]
    raw_result = cfunc(*raw_args)
    result_layout = result_type.__layout__
    if result_layout.register_passable:
        result = Memory(result_type, ffi.from_ffi(result_layout, raw_result))
    else:
        result = Memory.from_raw(result_type, raw_result)
    # Keep the JIT engine, input memories, and any callback closures alive
    # for as long as the result lives (non-register-passable results embed
    # pointers into those buffers).
    result.origins = [*func_mem.origins, *memories]
    return result


def register_executable(exe: Executable) -> list[object]:
    """JIT-compile an Executable and register its main function as a global symbol.

    Returns a list of objects that must be kept alive (the JIT engine).
    """
    engine = _jit_engine(exe)
    func_ptr = engine.get_function_address(exe.main_name)
    llvmlite.add_symbol(exe.main_name, func_ptr)
    return [engine]


def _raw_to_json(raw: object, ty: dgen.Type) -> object:
    """Convert a raw ctypes callback arg to a Python value."""
    layout = ty.__layout__
    if not layout.register_passable:
        return Memory.from_raw(ty, raw).to_json()
    if isinstance(raw, ctypes.Structure):
        return Memory(ty, ffi.from_ffi(layout, raw)).to_json()
    return raw  # scalar int/float passes through


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
    orig_types = [dgen.type.constant(arg.type) for arg in func_op.body.args]
    result_type = dgen.type.constant(func_op.result_type)
    result_ctype: type[ctypes._CData] | None = (
        None
        if isinstance(result_type, builtin.Nil)
        else _ffi_ctype(result_type.__layout__)
    )
    param_ctypes = [_ffi_ctype(t.__layout__) for t in orig_types]
    cb_type = ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    def _callback(*raw_args: object) -> object:
        python_args = [_raw_to_json(raw, typ) for raw, typ in zip(raw_args, orig_types)]
        mem = on_call(*python_args)
        layout = mem.type.__layout__
        if layout.register_passable:
            return ffi.to_ffi(layout, mem.buffer)
        return mem.address

    callback_func = cb_type(_callback)

    # Register callback symbol with llvmlite
    _ensure_initialized()
    llvmlite.add_symbol(
        callback_name,
        ctypes.cast(callback_func, ctypes.c_void_p).value,
    )

    # Build thunk: call callback with all original params, return result
    thunk_args = [
        BlockArgument(name=arg.name, type=arg.type) for arg in func_op.body.args
    ]
    callback_extern = builtin.ExternOp(
        symbol=String().constant(callback_name),
        type=Function(
            arguments=pack(arg.type for arg in thunk_args),
            result_type=result_type,
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
            arguments=pack(arg.type for arg in thunk_args),
            result_type=result_type,
        ),
    )
    ir, host_buffers = emit_llvm_ir(thunk_func)
    exe = Executable(
        ir=ir,
        input_types=orig_types,
        main_name=func_op.name,
        result_type=result_type,
        host_refs=host_buffers,
    )
    exe.host_refs.append(callback_func)  # prevent GC
    return exe


# ---------------------------------------------------------------------------
# Exit pass
# ---------------------------------------------------------------------------


class LLVMCodegen:
    """Exit pass: emit LLVM IR and bundle as Executable.

    The entry is always a ``FunctionOp`` with an emitted body. Non-FunctionOp
    inputs are wrapped:
      - ``Value[Function]`` (e.g. ``ExternOp``, ``Constant[Function]``) is
        wrapped in a trampoline that forwards its args to the callable.
      - Any other value is wrapped in a nil-arg function that returns it.

    Signature (input/result types) comes from the entry FunctionOp's
    ``Function`` type, not from its body — which keeps wrapping uniform.
    """

    def run(self, value: dgen.Value) -> Executable:
        if isinstance(value, function.FunctionOp):
            entry = value
        elif isinstance(value.type, function.Function):
            entry = _wrap_callable(value)
        else:
            entry = function.FunctionOp(
                name="main",
                body=dgen.Block(result=value),
                result_type=value.type,
                type=function.Function(arguments=pack([]), result_type=value.type),
            )

        func_type = entry.type
        assert isinstance(func_type, function.Function)
        input_types = constant(func_type.arguments)
        result_type = constant(func_type.result_type)
        assert entry.name is not None

        ir, host_buffers = emit_llvm_ir(entry)
        return Executable(
            ir=ir,
            input_types=input_types,
            main_name=entry.name,
            result_type=result_type,
            host_refs=host_buffers,
        )


def _wrap_callable(callable_value: dgen.Value) -> function.FunctionOp:
    """Wrap a ``Value[Function]`` (e.g. ExternOp) in a trampoline function
    that forwards its args to the callable."""
    func_type = callable_value.type
    assert isinstance(func_type, function.Function)
    arg_types = constant(func_type.arguments)
    result_type_value = func_type.result_type
    result_type = constant(result_type_value)
    args = [BlockArgument(name=f"a{i}", type=t) for i, t in enumerate(arg_types)]
    call_op = function.CallOp(
        callee=callable_value,
        arguments=pack(args),
        type=result_type,
    )
    return function.FunctionOp(
        name="main",
        body=dgen.Block(result=call_op, args=args, captures=[callable_value]),
        result_type=result_type_value,
        type=func_type,
    )
