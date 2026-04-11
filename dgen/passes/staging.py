"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.llvm.codegen import Executable, build_callback_thunk, register_executable
from dgen.dialects.function import Function, FunctionOp
from dgen.ir.traversal import all_values, transitive_dependencies
from dgen.builtins import ConstantOp, PackOp, pack
from dgen.passes.pass_ import Pass
from dgen.memory import Memory
from dgen.type import Constant

if TYPE_CHECKING:
    from dgen.passes.compiler import Compiler

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Resolution predicates and walkers
# ---------------------------------------------------------------------------


def _is_unresolved(value: dgen.Value) -> bool:
    """True if value is a bare SSA reference that needs JIT evaluation to
    become a Constant.

    The exclusion list — values that are *not* unresolved — covers everything
    that's already a complete value or a structural container the walker
    descends into separately rather than recursively here:

    - ``Constant``: a literal value, by definition.
    - ``dgen.Type``: a Type instance is the materialized form of a type.
      If its parameters reference unresolved SSA values, they surface
      naturally because the walker visits the Type as its own node in
      ``all_values`` and walks ``Type.compile_dependencies`` (= its params)
      from there. No recursive predicate needed.
    - ``FunctionOp``: a function value is materialized once you have a
      reference to it; you don't JIT-evaluate a function to "produce" it.
      Its body's compile dependencies are reached via the normal
      ``interior_values`` traversal as part of ``all_values``.
    - ``BlockParameter``: a structural placeholder bound at IR construction
      time (e.g. ``%self`` for the back-edge of a goto label). It's not a
      runtime SSA value and the staging engine has nothing to resolve.
    - ``PackOp``: ``[a, b, c]`` is shaped like a constant — a list literal
      whose elements are independently-walked SSA values. It only inherits
      ``Op`` for IR convenience (codegen treats it as a noop and inlines
      its elements at every use site). Like a Type, the walker descends
      into a PackOp's element values via the normal walk, not by recursing
      through the PackOp itself.
    """
    return not isinstance(
        value, (Constant, dgen.Type, FunctionOp, BlockParameter, PackOp)
    )


def _is_constant_foldable(value: dgen.Value) -> bool:
    """True if ``value``'s entire dependency subgraph reaches no
    ``BlockArgument`` — i.e. it can be JIT-evaluated at compile time
    without any runtime input."""
    return not any(isinstance(d, BlockArgument) for d in transitive_dependencies(value))


def _unresolved_compile_dependencies(root: dgen.Value) -> list[dgen.Value]:
    """The unresolved SSA references that appear as compile-time dependencies
    of some value reachable from ``root``, in topological (dataflow) order
    with duplicates removed.

    These are the staging engine's resolution targets: each one needs to be
    JIT-evaluated to a Constant (or, if it depends on runtime input, deferred
    to a callback thunk). Replacement happens via ``replace_uses_of`` so the
    caller doesn't need to track where each one is referenced.

    Topological order matters when targets are chained: JITing a value
    requires its own compile-time dependencies to already be Constants (so
    e.g. ``ShapeInference`` inside the JIT pipeline can finish). Walking via
    ``all_values`` yields dependencies before dependents; ``dict.fromkeys``
    deduplicates while preserving that order.
    """
    return list(
        dict.fromkeys(
            dep
            for v in all_values(root)
            for dep in v.compile_dependencies
            if _is_unresolved(dep)
        )
    )


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


def constant_fold_compile_dependencies(
    root: dgen.Value,
    compiler: Compiler[object],
) -> dgen.Value:
    """Fold every constant-foldable unresolved compile-time dependency in
    ``root`` to a Constant via JIT evaluation. Mutates ``root`` in place and
    returns it. Anything still unresolved after this depends on a runtime
    BlockArgument and must be deferred to a callback thunk.
    """
    for target in _unresolved_compile_dependencies(root):
        if _is_constant_foldable(target):
            root.replace_uses_of(target, _jit_evaluate(target, compiler))
    return root


def _resolve_with_runtime_args(
    func: FunctionOp,
    compiler: Compiler[object],
    python_args: Sequence[object],
) -> None:
    """Substitute ``func``'s block args with runtime-value constants, then
    fold every remaining unresolved compile-time dependency. After
    substitution every leaf is constant-foldable, so no filter is needed.
    """
    for block_arg, py_val in zip(func.body.args, python_args):
        const = ConstantOp.from_constant(block_arg.type.constant(py_val))
        func.body.replace_uses_of(block_arg, const)
    for target in _unresolved_compile_dependencies(func):
        func.body.replace_uses_of(target, _jit_evaluate(target, compiler))


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
    resolved = constant_fold_compile_dependencies(root, compiler)

    # Remaining boundaries depend on runtime (BlockArgument) values. Each
    # reachable function that contains such a boundary needs its own
    # callback thunk — the runtime values are that function's block args.
    # Non-entry callbacks are registered as global symbols first so the
    # entry callback can call them. The root itself is the entry.
    entry = resolved if isinstance(resolved, FunctionOp) else None
    needs_callback = [
        f
        for f in (v for v in all_values(resolved) if isinstance(v, FunctionOp))
        if _unresolved_compile_dependencies(f)
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
