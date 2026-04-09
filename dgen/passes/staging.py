"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from itertools import chain as iterchain
from typing import TYPE_CHECKING, TypeVar

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.llvm.codegen import Executable, build_callback_thunk, register_executable
from dgen.dialects.function import Function, FunctionOp
from dgen.ir.traversal import all_values, interior_values
from dgen.builtins import ConstantOp, pack
from dgen.passes.pass_ import Pass
from dgen.memory import Memory
from dgen.type import Constant

if TYPE_CHECKING:
    from dgen.passes.compiler import Compiler

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stage computation
# ---------------------------------------------------------------------------


def _is_unresolved(value: dgen.Value) -> bool:
    """True if value is an unresolved parameter (not yet a constant/type/function)."""
    return not isinstance(value, (Constant, dgen.Type, FunctionOp))


def _field_values(op: dgen.Op, fields: dgen.type.Fields) -> list[dgen.Value]:
    """Get all Value inputs from a set of fields, flattening list-valued ones."""
    return [getattr(op, name) for name, _ in fields]


def compute_stages(root: dgen.Value) -> dict[dgen.Value, int]:
    """Assign a stage number to every Value reachable from root.

    Stage 0 means the value can be evaluated at compile time (no runtime
    dependencies). Stage 1+ means it depends on runtime values.

    Base cases:
      - Constant / ConstantOp / Type / FunctionOp / BlockParameter: stage 0
      - BlockArgument: stage 1

    For ops, params bump the stage only when they depend on runtime::

        stage = max((
            *((1 + stage(p) if stage(p) > 0 else stage(p)) for p in __params__),
            *(stage(v) for v in __operands__),
        ))

    Returns a dict mapping ``value → stage_number``.
    """
    stages: dict[dgen.Value, int] = {}

    def _stage(value: dgen.Value) -> int:
        if value in stages:
            return stages[value]
        if isinstance(value, (Constant, dgen.Type, FunctionOp, BlockParameter)):
            stages[value] = 0
            return 0
        if isinstance(value, BlockArgument):
            stages[value] = 1
            return 1
        if not isinstance(value, dgen.Op):
            stages[value] = 0
            return 0
        parts: list[int] = []
        for pv in _field_values(value, value.__params__):
            s = _stage(pv)
            # +1 only when the param depends on runtime values (stage > 0).
            parts.append(1 + s if s > 0 else s)
        for ov in _field_values(value, value.__operands__):
            parts.append(_stage(ov))
        if _is_unresolved(value.type):
            s = _stage(value.type)
            parts.append(1 + s if s > 0 else s)
        result = max(parts, default=0)
        stages[value] = result
        return result

    for value in all_values(root):
        _stage(value)
    return stages


def _unresolved_boundaries(
    root: dgen.Value,
    stages: dict[dgen.Value, int],
) -> list[tuple[int, dgen.Op, str, dgen.Value]]:
    """Find unresolved __params__ / type boundaries sorted by stage.

    Walks each reachable function's top-level body ops; does not descend into
    nested blocks (loop bodies etc.) — those boundaries are resolved by
    recursive compilation once a parent staging resolution lands.
    """
    boundaries: list[tuple[int, dgen.Op, str, dgen.Value]] = []
    for func in [v for v in all_values(root) if isinstance(v, FunctionOp)]:
        for op in func.body.ops:
            if not isinstance(op, dgen.Op):
                continue
            for field_name, param in op.parameters:
                if _is_unresolved(param):
                    boundaries.append((stages.get(op, 0), op, field_name, param))
            if _is_unresolved(op.type):
                boundaries.append((stages.get(op, 0), op, "type", op.type))
    boundaries.sort(key=lambda t: t[0])
    return boundaries


# ---------------------------------------------------------------------------
# Compile-time subgraph evaluation
# ---------------------------------------------------------------------------


def _jit_evaluate(target: dgen.Value, compiler: Compiler[object]) -> ConstantOp:
    """JIT-compile and execute `target`, return its result as a ConstantOp.

    Target must be stage-0 evaluable (no BlockArgument dependencies).
    """
    func = FunctionOp(
        name="main",
        body=dgen.Block(result=target),
        result_type=target.type,
        type=Function(arguments=pack([]), result_type=target.type),
    )
    exe = compiler.run(func)
    result = exe.run()  # type: ignore[attr-defined]
    return ConstantOp.from_constant(target.type.constant(result.to_json()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConstantFold(Pass):
    """Pass that resolves all stage-0 comptime boundaries.

    Stage-0 resolution is performed by ``compile_value`` before the
    pass pipeline runs, so this pass is a no-op in the pipeline.
    It exists so callers can include it in a pass list for clarity.
    """

    allow_unregistered_ops = True


def resolve_stage0(
    root: dgen.Value,
    compiler: Compiler[object],
) -> dgen.Value:
    """Deepcopy + resolve all stage-0 comptime boundaries.

    Iteratively resolves the lowest-stage boundary whose value can be
    evaluated without runtime values (no BlockArgument dependencies).
    Stops when only runtime-dependent boundaries remain.
    """
    root = deepcopy(root)
    while True:
        stages = compute_stages(root)
        boundaries = [
            (op, field, value)
            for (_, op, field, value) in _unresolved_boundaries(root, stages)
            if stages.get(value, 0) == 0
        ]
        if not boundaries:
            break
        op, field_name, value = boundaries[0]
        setattr(op, field_name, _jit_evaluate(value, compiler))
    return root


def _resolve_with_runtime_args(
    func: FunctionOp,
    compiler: Compiler[object],
    python_args: Sequence[object],
) -> None:
    """Substitute func.body.args with runtime-value constants, then
    resolve any remaining stage-0 boundaries in func."""
    for block_arg, py_val in zip(func.body.args, python_args):
        const = ConstantOp.from_constant(block_arg.type.constant(py_val))
        func.body.replace_uses_of(block_arg, const)

    while True:
        stages = compute_stages(func)
        boundaries = _unresolved_boundaries(func, stages)
        if not boundaries:
            break
        _stage, op, field_name, value = boundaries[0]
        setattr(op, field_name, _jit_evaluate(value, compiler))


def _build_callback_thunk_for(
    resolved_root: dgen.Value,
    func: FunctionOp,
    compiler: Compiler[T],
) -> Executable:
    """Build a stage-1 thunk that calls a host callback for stage-2 JIT.

    The compiled function passes all its arguments to a host callback.
    The callback resolves remaining __params__ using the runtime args,
    then JIT-compiles and executes stage-2. ``func`` is the target
    FunctionOp inside ``resolved_root``'s graph.
    """
    callback_host_refs: list[object] = []

    def _on_call(*python_args: object) -> Memory:
        # deepcopy (root, func) together so the copied func is still
        # the same node inside the copied root.
        _, s2_func = deepcopy((resolved_root, func))
        _resolve_with_runtime_args(s2_func, compiler, python_args)
        exe = compiler.compile(s2_func)
        assert isinstance(exe, Executable)
        callback_host_refs.extend(exe.host_refs)
        mem = exe.run(*python_args)
        callback_host_refs.append(mem)
        return mem

    exe = build_callback_thunk(func, _on_call)
    exe.host_refs.append(callback_host_refs)
    return exe


def compile_value(root: dgen.Value, compiler: Compiler[T]) -> T:
    """Full staged compilation: resolve stage-0, run passes, exit.

    If all __params__ are resolvable at compile time (stage-0), resolves them
    and proceeds through the normal pass pipeline.

    If some __params__ depend on runtime values (stage-1+), builds callback-
    based thunks that JIT-compile when runtime values become available.

    For multi-function IR where multiple functions have unresolved
    boundaries, each is compiled as a callback thunk and registered as a
    global symbol so cross-function calls (including recursion) work.
    """
    resolved = resolve_stage0(root, compiler)

    # Remaining boundaries depend on runtime (BlockArgument) values. Each
    # reachable function that contains such a boundary needs its own
    # callback thunk — the runtime values are that function's block args.
    # Non-entry callbacks are registered as global symbols first so the
    # entry callback can call them. The root itself is the entry.
    entry = resolved if isinstance(resolved, FunctionOp) else None
    needs_callback = [
        f for f in [v for v in all_values(resolved) if isinstance(v, FunctionOp)] if _function_has_boundaries(f)
    ]
    if not needs_callback:
        return compiler.run(resolved)
    ordered = [f for f in needs_callback if f is not entry]
    if entry is not None and entry in needs_callback:
        ordered.append(entry)

    all_host_refs: list[object] = []
    entry_exe: Executable | None = None
    for func in ordered:
        exe = _build_callback_thunk_for(resolved, func, compiler)
        all_host_refs.extend(exe.host_refs)
        all_host_refs.extend(register_executable(exe))
        if func is entry:
            entry_exe = exe

    if entry_exe is not None:
        entry_exe.host_refs = all_host_refs
        return entry_exe  # type: ignore[return-value]

    # Entry has no unresolved boundaries — compile normally but keep
    # callback thunks alive.
    result = compiler.run(resolved)
    assert isinstance(result, Executable)
    result.host_refs.extend(all_host_refs)
    return result


def _has_unresolved(op: dgen.Op) -> bool:
    """True if op has any unresolved params, type, or block arg/param types."""
    if any(_is_unresolved(p) for _, p in op.parameters) or _is_unresolved(op.type):
        return True
    for _, block in op.blocks:
        if any(_is_unresolved(arg.type) for arg in block.args):
            return True
        if any(_is_unresolved(param.type) for param in block.parameters):
            return True
    return False


def _function_has_boundaries(func: FunctionOp) -> bool:
    """True if any op in ``func``'s own body has unresolved __params__ or
    an unresolved type. Does not cross into captured callees.
    """
    all_ops = iterchain(
        func.body.values, *(interior_values(v) for v in func.body.values)
    )
    return any(_has_unresolved(v) for v in all_ops if isinstance(v, dgen.Op))
