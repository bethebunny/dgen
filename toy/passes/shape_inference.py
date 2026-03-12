"""Shape inference pass: resolve InferredShapeTensor to concrete Tensor."""

from __future__ import annotations

from copy import deepcopy

import dgen
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module, PackOp
from toy.dialects import shape_constant
from toy.dialects import toy


def _resolve_index_value(val: dgen.Value) -> int | None:
    """Try to resolve an index Value to a concrete int.

    Only succeeds if val is a ConstantOp with an integer value.
    Returns None for computed values (add_index, etc.) — those require
    a staging evaluator.
    """
    if isinstance(val, ConstantOp) and isinstance(val.type, builtin.Index):
        result = val.__constant__.to_json()
        assert isinstance(result, int)
        return result
    return None


def _infer_function(
    func: builtin.FunctionOp,
    func_map: dict[str, builtin.FunctionOp],
    type_of: dict[dgen.Value, toy.Tensor],
) -> None:
    """Infer shapes for all ops in a function body."""
    # Seed from block args that already have concrete types
    for arg in func.body.args:
        if isinstance(arg.type, toy.Tensor):
            type_of[arg] = arg.type

    for op in func.body.ops:
        if isinstance(op, ConstantOp) and isinstance(op.type, toy.Tensor):
            type_of[op] = op.type

        elif isinstance(op, toy.ReshapeOp) and isinstance(op.type, toy.Tensor):
            type_of[op] = op.type

        elif isinstance(op, toy.TransposeOp):
            src = type_of.get(op.input)
            if src is not None:
                t = toy.Tensor(
                    shape=shape_constant(
                        list(reversed(src.shape.__constant__.to_json()))
                    )
                )
                op.type = t
                type_of[op] = t

        elif isinstance(op, (toy.MulOp, toy.AddOp)):
            src = type_of.get(op.lhs)
            if src is not None:
                t = toy.Tensor(shape=shape_constant(src.shape.__constant__.to_json()))
                op.type = t
                type_of[op] = t

        elif isinstance(op, toy.ConcatOp):
            lhs = type_of.get(op.lhs)
            rhs = type_of.get(op.rhs)
            if lhs is not None and rhs is not None:
                lhs_dims = lhs.shape.__constant__.to_json()
                rhs_dims = rhs.shape.__constant__.to_json()
                shape = list(lhs_dims)
                axis = op.axis.__constant__.to_json()
                assert isinstance(axis, int)
                shape[axis] = lhs_dims[axis] + rhs_dims[axis]
                t = toy.Tensor(shape=shape_constant(shape))
                op.type = t
                type_of[op] = t

        elif isinstance(op, toy.TileOp):
            src = type_of.get(op.input)
            if src is not None:
                # Try to resolve count by peeking through a constant op
                count_val = (
                    _resolve_index_value(op.count)
                    if isinstance(op.count, dgen.Value)
                    else None
                )
                if count_val is not None:
                    t = toy.Tensor(
                        shape=shape_constant(
                            [count_val] + src.shape.__constant__.to_json()
                        )
                    )
                    op.type = t
                    type_of[op] = t

        elif isinstance(op, builtin.CallOp):
            args_list = op.args.values if isinstance(op.args, PackOp) else [op.args]
            resolved = [type_of.get(a) for a in args_list]
            arg_types = [t for t in resolved if t is not None]
            if len(arg_types) == len(resolved):
                callee = func_map.get(op.callee.name)
                if callee is not None:
                    # Set callee param types from call-site args
                    for param, atype in zip(callee.body.args, arg_types):
                        param.type = atype
                        type_of[param] = atype
                    # Infer callee body
                    _infer_function(callee, func_map, type_of)
                    # Update callee signature
                    ret_op = callee.body.ops[-1]
                    if isinstance(ret_op, builtin.ReturnOp) and not isinstance(
                        ret_op.value, builtin.Nil
                    ):
                        ret_type = type_of.get(ret_op.value)
                        if ret_type is not None:
                            callee.result = ret_type
                            op.type = toy.Tensor(
                                shape=shape_constant(
                                    ret_type.shape.__constant__.to_json()
                                )
                            )
                            type_of[op] = op.type


def infer_shapes(m: Module) -> Module:
    m = deepcopy(m)
    func_map: dict[str, builtin.FunctionOp] = {
        f.name: f for f in m.functions if f.name is not None
    }
    type_of: dict[dgen.Value, toy.Tensor] = {}

    # Process main first (all shapes derivable from constants)
    main = func_map.get("main")
    if main is not None:
        _infer_function(main, func_map, type_of)

    # Process remaining functions (may already be inferred via call sites)
    for func in m.functions:
        if func.name != "main":
            _infer_function(func, func_map, type_of)

    return m
