"""Staging evaluator: JIT-compile stage-0 expressions to resolve Comptime fields."""

from __future__ import annotations

from copy import deepcopy
from typing import get_type_hints

import dgen
from dgen.codegen import jit_eval
from dgen.dialects import builtin, llvm
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


def _lower_builtin_to_llvm(module: builtin.Module) -> builtin.Module:
    """Lower builtin ops (AddIndexOp) in a stage-0 mini-module to LLVM ops."""
    value_map: dict[int, dgen.Value] = {}

    def _map(v: dgen.Value) -> dgen.Value:
        return value_map.get(id(v), v)

    func = module.functions[0]
    new_ops: list[dgen.Op] = []
    for op in func.body.ops:
        if isinstance(op, builtin.ConstantOp):
            new_op = builtin.ConstantOp(value=op.value, type=op.type)
            value_map[id(op)] = new_op
            new_ops.append(new_op)
        elif isinstance(op, builtin.AddIndexOp):
            new_op = llvm.AddOp(lhs=_map(op.lhs), rhs=_map(op.rhs))
            value_map[id(op)] = new_op
            new_ops.append(new_op)
        elif isinstance(op, builtin.ReturnOp):
            val = _map(op.value) if op.value is not None else None
            new_ops.append(builtin.ReturnOp(value=val))

    new_func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=new_ops),
        type=func.type,
    )
    return builtin.Module(functions=[new_func])


def _jit_evaluate(subgraph: list[dgen.Op], target: dgen.Value) -> object:
    """Build a mini-module from the subgraph, lower, JIT, return the result."""
    # Build a function that computes and returns the target value
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=ops),
        type=builtin.Function(result=target.type),
    )
    module = builtin.Module(functions=[func])

    # Lower builtin ops (AddIndexOp) to LLVM ops
    lowered = _lower_builtin_to_llvm(module)

    # JIT compile and execute
    layout = target.type.__layout__
    return jit_eval(lowered, layout)


def evaluate_stage0(module: builtin.Module) -> builtin.Module:
    """Evaluate Comptime fields by JIT-compiling their dependency subgraphs.

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
                # Collect the ops needed to compute this value
                subgraph = _trace_dependencies(value, func)
                # JIT-evaluate the subgraph
                result = _jit_evaluate(subgraph, value)
                # Create a constant op with the computed result
                const_op = builtin.ConstantOp(value=result, type=value.type)
                # Insert the constant before this op and update the field
                idx = func.body.ops.index(op)
                func.body.ops.insert(idx, const_op)
                setattr(op, field_name, const_op)
    return module
