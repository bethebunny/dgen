"""Staging evaluator: JIT-compile stage-0 expressions to resolve Comptime fields."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import get_type_hints

import dgen
from dgen.codegen import jit_eval
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
