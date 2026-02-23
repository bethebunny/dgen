"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import get_type_hints

import dgen
from dgen.block import BlockArgument
from dgen.codegen import compile_and_run, jit_eval
from dgen.dialects import builtin
from dgen.value import Comptime


def _comptime_fields(cls: type) -> list[str]:
    """Return field names whose type hint is Comptime."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        return []
    return [name for name, hint in hints.items() if hint is Comptime]


def _trace_dependencies(target: dgen.Value, func: builtin.FuncOp) -> list[dgen.Op]:
    """Backward-walk from target, return all needed ops in topological order."""
    needed: set[int] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if id(val) in needed:
            continue
        needed.add(id(val))
        if isinstance(val, dgen.Op):
            for _, operand in val.operands:
                worklist.append(operand)
    return [op for op in func.body.ops if id(op) in needed]


def _is_stage0_evaluable(target: dgen.Value) -> bool:
    """True if target's dependencies can be evaluated without runtime values.

    Returns False if any value in the dependency tree is a BlockArgument
    (function parameter), meaning it depends on runtime input.
    """
    visited: set[int] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if id(val) in visited:
            continue
        visited.add(id(val))
        if isinstance(val, BlockArgument):
            return False
        if isinstance(val, dgen.Op):
            for _, operand in val.operands:
                worklist.append(operand)
    return True


def _prepare_ctypes_args(
    block_args: list[BlockArgument], python_args: list
) -> tuple[list, list]:
    """Convert Python args to ctypes values based on block argument types."""
    ctypes_args: list = []
    host_bufs: list = []
    for arg, param in zip(python_args, block_args):
        ct_val, refs = param.type.__layout__.prepare_arg(arg, param.type)
        ctypes_args.append(ct_val)
        host_bufs.extend(refs)
    return ctypes_args, host_bufs


def _jit_evaluate(
    subgraph: list[dgen.Op],
    target: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
    *,
    block_args: list[BlockArgument] | None = None,
    args: list | None = None,
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=ops, args=list(block_args or [])),
        type=builtin.Function(result=target.type),
    )
    module = builtin.Module(functions=[func])
    lowered = lower(module)
    layout = target.type.__layout__
    ctypes_args: list = []
    if block_args and args:
        ctypes_args, _ = _prepare_ctypes_args(block_args, args)
    return jit_eval(lowered, layout, args=ctypes_args)


def _resolve_comptime_field(
    func: builtin.FuncOp,
    op: dgen.Op,
    field_name: str,
    value: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
) -> None:
    """Resolve a single Comptime field: JIT the dependency subgraph, patch with ConstantOp."""
    subgraph = _trace_dependencies(value, func)
    stage1 = not _is_stage0_evaluable(value)
    result = _jit_evaluate(
        subgraph,
        value,
        lower,
        block_args=func.body.args if stage1 else None,
        args=args,
    )
    if stage1:
        result = int(result)  # type: ignore[arg-type]
        subgraph_ids = {id(o) for o in subgraph}
        func.body.ops[:] = [o for o in func.body.ops if id(o) not in subgraph_ids]
    const_op = builtin.ConstantOp(value=result, type=value.type)
    idx = func.body.ops.index(op)
    func.body.ops.insert(idx, const_op)
    setattr(op, field_name, const_op)


def _resolve_constant_ops(func: builtin.FuncOp) -> None:
    """Replace ops that implement resolve_constant() with ConstantOps.

    After shape inference runs, ops like DimSizeOp may be resolvable to
    constants because their input types are now concrete. This function
    finds such ops and replaces them, patching any Comptime field references.
    """
    for i, op in enumerate(func.body.ops):
        resolver = getattr(op, "resolve_constant", None)
        if resolver is None:
            continue
        val = resolver()
        if val is None:
            continue
        const = builtin.ConstantOp(value=val, type=op.type)
        func.body.ops[i] = const
        for other in func.body.ops:
            for fn in _comptime_fields(type(other)):
                if getattr(other, fn) is op:
                    setattr(other, fn, const)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_and_run_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
    capture_output: bool = True,
) -> str | None:
    """Full staged compilation pipeline.

    Iteratively resolves all Comptime fields:
    - Stage-0 fields (constant dependencies) are JIT-evaluated in isolation.
    - Stage-1 fields (runtime dependencies) are JIT-evaluated with runtime args.
    After each resolution, shape inference is interleaved to propagate types,
    and constant-resolving ops (e.g. DimSizeOp) are replaced with ConstantOps.
    The loop restarts to handle subsequent Comptime fields.
    """
    module = deepcopy(module)

    changed = True
    while changed:
        changed = False
        func = module.functions[0]
        for op in list(func.body.ops):
            for field_name in _comptime_fields(type(op)):
                value = getattr(op, field_name)
                if isinstance(value, builtin.ConstantOp):
                    continue
                _resolve_comptime_field(func, op, field_name, value, lower, args)
                module = infer(module)
                _resolve_constant_ops(module.functions[0])
                changed = True
                break
            if changed:
                break

    # All comptime resolved, run full pipeline
    func = module.functions[0]
    typed = infer(module)
    lowered = lower(typed)
    ctypes_args: list = []
    if args and func.body.args:
        ctypes_args, _ = _prepare_ctypes_args(func.body.args, args)
    return compile_and_run(lowered, capture_output=capture_output, args=ctypes_args)
