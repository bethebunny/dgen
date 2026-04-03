"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Iterator, TypeVar

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.codegen import Executable, build_callback_thunk, register_executable
from dgen.dialects import function
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, pack
from dgen.type import Constant, Memory

from dgen.passes.pass_ import Pass

if TYPE_CHECKING:
    from dgen.compiler import Compiler

T = TypeVar("T")


def _walk_inputs(op: dgen.Op) -> Iterator[dgen.Value]:
    """Yield all parameter and operand Values."""
    for _, val in itertools.chain(op.parameters, op.operands):
        yield val


def _trace_dependencies(target: dgen.Value, func: FunctionOp) -> list[dgen.Op]:
    """Backward-walk from target, return all needed ops in topological order."""
    needed: set[dgen.Value] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if val in needed:
            continue
        needed.add(val)
        if isinstance(val, dgen.Op):
            worklist.extend(_walk_inputs(val))
    return [op for op in func.body.ops if op in needed]


def _jit_evaluate(
    subgraph: list[dgen.Op],
    target: dgen.Value,
    compile: Callable[[Module], Executable],
    *,
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> object:
    """Build a mini-module from the subgraph, compile, and JIT-execute."""
    func = function.FunctionOp(
        name="main",
        body=dgen.Block(result=target, args=list(block_args)),
        result_type=target.type,
        type=Function(
            arguments=pack(arg.type for arg in block_args), result_type=target.type
        ),
    )
    module = Module(ops=[func])
    exe = compile(module)
    # LIFETIME BUG WORKAROUND: Executable.run() converts raw Python values
    # to Memory objects internally, but those temporaries can be GC'd before
    # the result is read. For non-register-passable types (e.g. TypeType),
    # the JIT function returns a pointer into the input Memory's buffer.
    # If that Memory is collected, the pointer dangles and to_json() reads
    # garbage. Creating Memory objects here keeps them alive through to_json().
    # The real fix is for Executable.run() to attach input memories to the
    # result's host_refs so they outlive the call.
    memories = [
        Memory.from_value(param.type, arg) for arg, param in zip(args, block_args)
    ]
    result = exe.run(*memories)
    return result.to_json()


def _resolve_comptime_field(
    func: FunctionOp,
    op: dgen.Op,
    field_name: str,
    value: dgen.Value,
    compile: Callable[[Module], Executable],
    *,
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> None:
    """Resolve a single Constant field: JIT the dependency subgraph, patch with ConstantOp."""
    subgraph = _trace_dependencies(value, func)
    result = _jit_evaluate(
        subgraph,
        value,
        compile,
        block_args=block_args,
        args=args,
    )
    const_type = value.type
    if isinstance(result, dict) and "tag" in result:
        const_type = dgen.type.TypeType()
    const_op = ConstantOp(value=result, type=const_type)
    setattr(op, field_name, const_op)


# ---------------------------------------------------------------------------
# Stage computation
# ---------------------------------------------------------------------------


def _field_values(op: dgen.Op, fields: dgen.type.Fields) -> list[dgen.Value]:
    """Get all Value inputs from a set of fields, flattening list-valued ones."""
    return [getattr(op, name) for name, _ in fields]


def compute_stages(func: FunctionOp) -> dict[dgen.Value, int]:
    """Assign a stage number to every Value in a function.

    Stage 0 means the value can be evaluated at compile time (no runtime
    dependencies). Stage 1+ means it depends on runtime values.

    Base cases:
      - Constant / ConstantOp: stage 0
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
            # Forward reference to a module-level entity (e.g. function name)
            stages[value] = 0
            return 0
        parts: list[int] = []
        for pv in _field_values(value, value.__params__):
            s = _stage(pv)
            # +1 only when the param depends on runtime values (stage > 0).
            # Stage-0 params (all-constant deps) don't bump the stage —
            # the op is still evaluable at compile time.
            parts.append(1 + s if s > 0 else s)
        for ov in _field_values(value, value.__operands__):
            parts.append(_stage(ov))
        # Unresolved type ref counts as a param boundary
        if isinstance(value.type, dgen.Value) and not isinstance(
            value.type, (Constant, dgen.Type)
        ):
            s = _stage(value.type)
            parts.append(1 + s if s > 0 else s)
        result = max(parts, default=0)
        stages[value] = result
        return result

    for arg in func.body.args:
        stages[arg] = 1
    for op in func.body.ops:
        _stage(op)
    return stages


def _unresolved_boundaries(
    func: FunctionOp,
    stages: dict[dgen.Value, int],
) -> list[tuple[int, dgen.Op, str, dgen.Value]]:
    """Find ops with unresolved __params__, sorted by stage number.

    Returns ``(stage, op, field_name, param_value)`` tuples for every
    ``__params__`` field that is a non-Constant Value.
    """
    boundaries: list[tuple[int, dgen.Op, str, dgen.Value]] = []
    for op in func.body.ops:
        for field_name, value in op.parameters:
            if isinstance(
                value, (dgen.Op, BlockArgument, BlockParameter)
            ) and not isinstance(value, (Constant, dgen.Type, FunctionOp)):
                boundaries.append((stages.get(op, 0), op, field_name, value))
        # Also check op.type — if it's an unresolved SSA ref (Value, not Type)
        if isinstance(op.type, dgen.Value) and not isinstance(
            op.type, (Constant, dgen.Type)
        ):
            boundaries.append((stages.get(op, 0), op, "type", op.type))
    boundaries.sort(key=lambda t: t[0])
    return boundaries


def _has_nested_boundaries(func: FunctionOp) -> bool:
    """True if any op in nested blocks has unresolved __params__."""
    for op in func.body.ops:
        for _, block in op.blocks:
            for nested_op in block.ops:
                for _, value in nested_op.parameters:
                    if isinstance(
                        value, (dgen.Op, BlockArgument, BlockParameter)
                    ) and not isinstance(value, (Constant, dgen.Type, FunctionOp)):
                        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConstantFold(Pass):
    """Pass that resolves all stage-0 comptime boundaries.

    Iteratively finds unresolved parameters whose dependency subgraphs
    consist entirely of constants, JIT-evaluates them, and patches the
    results back as ConstantOps. Uses the continuation compiler's run
    method to lower extracted subgraphs for JIT.
    """

    allow_unregistered_ops = True

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        return resolve_stage0(module, compiler.run)


def resolve_stage0(
    module: Module,
    compile: Callable[[Module], Executable],
) -> Module:
    """Deepcopy + resolve all stage-0 comptime boundaries.

    Iteratively resolves the lowest-stage boundary whose value can be
    evaluated without runtime values (no BlockArgument dependencies).
    Stops when only runtime-dependent boundaries remain.
    """
    module = deepcopy(module)
    while True:
        resolved_any = False
        for func in module.functions:
            stages = compute_stages(func)
            boundaries = _unresolved_boundaries(func, stages)
            if not boundaries:
                continue
            _stage_num, op, field_name, value = boundaries[0]
            if stages.get(value, 0) != 0:
                continue
            _resolve_comptime_field(func, op, field_name, value, compile)
            resolved_any = True
            break  # re-iterate from the start after each resolution
        if not resolved_any:
            break
    return module


def _resolve_with_runtime_args(
    func: FunctionOp,
    compile: Callable[[Module], Executable],
    block_args: Sequence[BlockArgument],
    python_args: Sequence,
) -> None:
    """Resolve all boundaries in a function using runtime argument values.

    Unlike ``resolve_stage0``, this resolves boundaries at any stage
    by passing runtime args to the JIT evaluator. Used inside callback
    thunks when runtime values become available.
    """
    while True:
        stages = compute_stages(func)
        boundaries = _unresolved_boundaries(func, stages)
        if not boundaries:
            break
        _stage_num, op, field_name, value = boundaries[0]
        _resolve_comptime_field(
            func,
            op,
            field_name,
            value,
            compile,
            block_args=block_args,
            args=python_args,
        )


def _build_callback_thunk(
    resolved: Module,
    func: FunctionOp,
    compiler: Compiler[T],
) -> Executable:
    """Build a stage-1 thunk that calls a host callback for stage-2 JIT.

    The compiled function passes all its arguments to a host callback.
    The callback resolves all remaining __params__ values (using the full
    stage-1 resolution loop), then JIT-compiles and executes stage-2.
    """
    stage2_template = resolved
    assert func.name is not None
    func_name = func.name
    callback_host_refs: list[object] = []

    def _on_call(*python_args: object) -> Memory:
        template = deepcopy(stage2_template)
        s2_func = next(f for f in template.functions if f.name == func_name)

        _resolve_with_runtime_args(
            s2_func, compiler.run, s2_func.body.args, python_args
        )

        func_module = Module(ops=[s2_func])
        result = compiler.compile(func_module)
        assert isinstance(result, Executable)
        callback_host_refs.extend(result.host_refs)
        mem = result.run(*python_args)
        callback_host_refs.append(mem)
        return mem

    exe = build_callback_thunk(func, _on_call)
    exe.host_refs.append(callback_host_refs)
    return exe


def compile_module(module: Module, compiler: Compiler[T]) -> T:
    """Full staged compilation: resolve stage-0, run passes, exit.

    If all __params__ are resolvable at compile time (stage-0), resolves them
    and proceeds through the normal pass pipeline.

    If some __params__ depend on runtime values (stage-1+), builds callback-
    based thunks that JIT-compile when runtime values become available.

    For multi-function modules where multiple functions have unresolved
    boundaries, each is compiled as a callback thunk and registered as a
    global symbol so cross-function calls (including recursion) work.
    """

    resolved = resolve_stage0(module, compiler.run)

    # Find all functions with unresolved boundaries (including nested blocks)
    unresolved_funcs: list[FunctionOp] = []
    for func in resolved.functions:
        stages = compute_stages(func)
        if _unresolved_boundaries(func, stages) or _has_nested_boundaries(func):
            unresolved_funcs.append(func)

    if not unresolved_funcs:
        return compiler.run(resolved)

    # Compile each unresolved function as a callback thunk
    # and register it as a global symbol for cross-function calls.
    # Process non-entry functions first so their symbols are available
    # when the entry function's callback fires.
    entry = resolved.functions[0]
    ordered = [f for f in unresolved_funcs if f is not entry] + [
        f for f in unresolved_funcs if f is entry
    ]

    all_host_refs: list[object] = []
    entry_exe: Executable | None = None

    for func in ordered:
        exe = _build_callback_thunk(resolved, func, compiler)
        all_host_refs.extend(exe.host_refs)
        all_host_refs.extend(register_executable(exe))

        if func is entry:
            entry_exe = exe

    if entry_exe is not None:
        entry_exe.host_refs = all_host_refs
        return entry_exe  # type: ignore[return-value]

    # Entry point has no unresolved boundaries — compile normally
    # but keep callback thunks alive
    result = compiler.run(resolved)
    assert isinstance(result, Executable)
    result.host_refs.extend(all_host_refs)
    return result
