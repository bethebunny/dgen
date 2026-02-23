"""Staging evaluator: JIT-compile stage-0 expressions to resolve Comptime fields."""

from __future__ import annotations

import ctypes
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


def _jit_evaluate(
    subgraph: list[dgen.Op],
    target: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=ops),
        type=builtin.Function(result=target.type),
    )
    module = builtin.Module(functions=[func])
    lowered = lower(module)
    layout = target.type.__layout__
    return jit_eval(lowered, layout)


def evaluate_stage0(
    module: builtin.Module,
    lower: Callable[[builtin.Module], builtin.Module],
) -> builtin.Module:
    """Evaluate Comptime fields by JIT-compiling their dependency subgraphs.

    ``lower`` is a callable that takes a Module containing stage-0 ops and
    returns a Module ready for codegen (LLVM dialect).  This is the same
    lowering pipeline the rest of the compiler uses, so any op with a
    lowering path works automatically.

    After this pass, all Comptime fields point to ConstantOps, so that
    shape inference can resolve them via _resolve_index_value.
    """
    module = deepcopy(module)
    for func in module.functions:
        for op in list(func.body.ops):
            for field_name in _comptime_fields(type(op)):
                value = getattr(op, field_name)
                if isinstance(value, builtin.ConstantOp):
                    continue
                subgraph = _trace_dependencies(value, func)
                result = _jit_evaluate(subgraph, value, lower)
                const_op = builtin.ConstantOp(value=result, type=value.type)
                idx = func.body.ops.index(op)
                func.body.ops.insert(idx, const_op)
                setattr(op, field_name, const_op)
    return module


# ---------------------------------------------------------------------------
# Stage-1 JIT: runtime-dependent comptime values
# ---------------------------------------------------------------------------


def _prepare_ctypes_args(
    block_args: list[BlockArgument], python_args: list
) -> tuple[list, list]:
    """Convert Python args to ctypes values based on block argument types."""
    ctypes_args: list = []
    host_bufs: list = []
    for arg, param in zip(python_args, block_args):
        if hasattr(param.type, "shape"):
            buf = (ctypes.c_double * len(arg))(*arg)
            host_bufs.append(buf)
            ctypes_args.append(ctypes.cast(buf, ctypes.c_void_p))
        else:
            ctypes_args.append(arg)
    return ctypes_args, host_bufs


def compile_and_run_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
    capture_output: bool = True,
) -> str | None:
    """Full staged compilation pipeline.

    1. Resolves stage-0 Comptime fields via isolated JIT.
    2. For stage-1 Comptime fields (runtime-dependent):
       - JITs stage 1 to compute the comptime value from runtime args
       - Instantiates and runs stage 2 with the resolved value
    """
    module = deepcopy(module)

    for func in module.functions:
        for op in list(func.body.ops):
            for field_name in _comptime_fields(type(op)):
                value = getattr(op, field_name)
                if isinstance(value, builtin.ConstantOp):
                    continue

                if _is_stage0_evaluable(value):
                    # Stage 0: evaluate in isolation
                    subgraph = _trace_dependencies(value, func)
                    result = _jit_evaluate(subgraph, value, lower)
                    const_op = builtin.ConstantOp(value=result, type=value.type)
                    idx = func.body.ops.index(op)
                    func.body.ops.insert(idx, const_op)
                    setattr(op, field_name, const_op)
                else:
                    # Stage 1: runtime-dependent comptime value
                    return _execute_stage1(
                        func, op, field_name, value,
                        infer=infer, lower=lower,
                        args=args, capture_output=capture_output,
                    )

    # All stage-0 resolved (or no comptime fields), run normally
    typed = infer(module)
    lowered = lower(typed)
    return compile_and_run(lowered, capture_output=capture_output)


def _execute_stage1(
    func: builtin.FuncOp,
    boundary_op: dgen.Op,
    field_name: str,
    comptime_value: dgen.Value,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None,
    capture_output: bool,
) -> str | None:
    """Execute stage-1 JIT to resolve a runtime-dependent Comptime field."""
    subgraph = _trace_dependencies(comptime_value, func)

    # --- Stage 1: compute the comptime value from runtime args ---
    stage1_ops = list(subgraph) + [builtin.ReturnOp(value=comptime_value)]
    stage1_func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=stage1_ops, args=list(func.body.args)),
        type=builtin.Function(result=comptime_value.type),
    )
    stage1_module = builtin.Module(functions=[stage1_func])
    lowered_s1 = lower(stage1_module)

    # Prepare ctypes args and JIT stage 1
    ctypes_args, _host_bufs = _prepare_ctypes_args(
        func.body.args, args if args is not None else []
    )
    layout = comptime_value.type.__layout__
    result = jit_eval(lowered_s1, layout, args=ctypes_args)

    # --- Stage 2: instantiate with the resolved value ---
    subgraph_ids = {id(op) for op in subgraph}
    stage2_ops = [op for op in func.body.ops if id(op) not in subgraph_ids]

    const_op = builtin.ConstantOp(value=int(result), type=comptime_value.type)
    idx = stage2_ops.index(boundary_op)
    stage2_ops.insert(idx, const_op)
    setattr(boundary_op, field_name, const_op)

    stage2_func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=stage2_ops),
        type=builtin.Function(result=func.type.result),
    )
    stage2_module = builtin.Module(functions=[stage2_func])

    # Run stage 2 through the full pipeline
    typed = infer(stage2_module)
    lowered_s2 = lower(typed)
    return compile_and_run(lowered_s2, capture_output=capture_output)
