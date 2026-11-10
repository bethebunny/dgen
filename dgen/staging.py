"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.codegen import Executable, build_callback_thunk, register_executable
from dgen.dialects.function import Function, FunctionOp
from dgen.graph import all_values, transitive_dependencies
from dgen.module import ConstantOp, pack
from dgen.passes.pass_ import Pass
from dgen.type import Constant, Memory

if TYPE_CHECKING:
    from dgen.compiler import Compiler

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stage computation
# ---------------------------------------------------------------------------


def _is_unresolved(value: dgen.Value) -> bool:
    """True if value is an unresolved parameter (not yet a constant/type/function).

    A BlockParameter is resolved: the op owning its block binds it, eg. a
    function's %self.
    """
    return not isinstance(value, (Constant, dgen.Type, FunctionOp, BlockParameter))


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


class StagingError(Exception):
    """A boundary that staging can't resolve."""


type Boundary = tuple[dgen.Op, str, dgen.Value]


def _body_ops(block: dgen.Block) -> Iterator[dgen.Op]:
    """Ops of block and its nested blocks in topological order, except nested functions' bodies."""
    for value in block.values:
        if not isinstance(value, dgen.Op):
            continue
        if not isinstance(value, FunctionOp):
            for _, inner in value.blocks:
                yield from _body_ops(inner)
        yield value


def _function_boundaries(func: FunctionOp) -> Iterator[Boundary]:
    """Unresolved fields of the ops in func's body, nested blocks included."""
    for op in _body_ops(func.body):
        for field_name, param in op.parameters:
            if _is_unresolved(param):
                yield op, field_name, param
        if _is_unresolved(op.type):
            yield op, "type", op.type


def _unresolved_boundaries(
    root: dgen.Value, stages: dict[dgen.Value, int]
) -> list[Boundary]:
    """Boundaries of every reachable function, by the op's stage, then topologically."""
    boundaries = [
        b for f in _reachable_functions(root) for b in _function_boundaries(f)
    ]
    return sorted(boundaries, key=lambda b: stages.get(b[0], 0))


def _verify_specializable(func: FunctionOp) -> None:
    """Raise unless each of func's boundaries depends on no block argument but func's own.

    A callback specializes func on its arguments only, so a boundary on eg. a loop
    variable would stay unresolved in every specialization.
    """
    for op, field_name, value in _function_boundaries(func):
        for dep in transitive_dependencies(value):
            if isinstance(dep, BlockArgument) and dep not in func.body.args:
                raise StagingError(
                    f"%{op.name}.{field_name} in function %{func.name} depends on "
                    f"%{dep.name}, an argument of a nested block; staging "
                    f"specializes only on the function's arguments"
                )


def _reachable_functions(root: dgen.Value) -> list[FunctionOp]:
    """FunctionOps reachable from root, in topological order."""
    return [v for v in all_values(root) if isinstance(v, FunctionOp)]


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
    return ConstantOp(value=result.to_json(), type=target.type)


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
            for (op, field, value) in _unresolved_boundaries(root, stages)
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
        const = ConstantOp(value=py_val, type=block_arg.type)
        func.body.replace_uses_of(block_arg, const)

    while True:
        stages = compute_stages(func)
        boundaries = _unresolved_boundaries(func, stages)
        if not boundaries:
            break
        op, field_name, value = boundaries[0]
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
        f
        for f in _reachable_functions(resolved)
        if next(_function_boundaries(f), None) is not None
    ]
    for func in needs_callback:
        _verify_specializable(func)
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
