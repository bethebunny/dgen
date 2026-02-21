"""Shape inference pass: resolve InferredShapeTensor to concrete TensorType."""

from __future__ import annotations

from copy import deepcopy

import dgen
from dgen.dialects import builtin
from toy.dialects import toy


def _resolve(
    type_of: dict[int, toy.TensorType], val: dgen.Value
) -> toy.TensorType | None:
    return type_of.get(id(val))


def _infer_function(
    func: builtin.FuncOp,
    func_map: dict[str, builtin.FuncOp],
    type_of: dict[int, toy.TensorType],
):
    """Infer shapes for all ops in a function body."""
    # Seed from block args that already have concrete types
    for arg in func.body.args:
        if isinstance(arg.type, toy.TensorType):
            type_of[id(arg)] = arg.type

    for op in func.body.ops:
        if isinstance(op, builtin.ConstantOp) and isinstance(op.type, toy.TensorType):
            type_of[id(op)] = op.type

        elif isinstance(op, toy.ReshapeOp) and isinstance(op.type, toy.TensorType):
            type_of[id(op)] = op.type

        elif isinstance(op, toy.TransposeOp):
            src = _resolve(type_of, op.input)
            if src is not None:
                t = toy.TensorType(shape=list(reversed(src.shape)))
                op.type = t
                type_of[id(op)] = t

        elif isinstance(op, (toy.MulOp, toy.AddOp)):
            src = _resolve(type_of, op.lhs)
            if src is not None:
                t = toy.TensorType(shape=list(src.shape))
                op.type = t
                type_of[id(op)] = t

        elif isinstance(op, toy.GenericCallOp):
            arg_types = [_resolve(type_of, a) for a in op.args]
            if all(t is not None for t in arg_types):
                callee = func_map.get(op.callee)
                if callee is not None:
                    # Set callee param types from call-site args
                    for param, atype in zip(callee.body.args, arg_types):
                        param.type = atype
                        type_of[id(param)] = atype
                    # Infer callee body
                    _infer_function(callee, func_map, type_of)
                    # Update callee signature
                    ret_op = callee.body.ops[-1]
                    if (
                        isinstance(ret_op, builtin.ReturnOp)
                        and ret_op.value is not None
                    ):
                        ret_type = _resolve(type_of, ret_op.value)
                        if ret_type is not None:
                            callee.type = toy.FunctionType(
                                inputs=list(arg_types),
                                result=ret_type,
                            )
                            op.type = toy.TensorType(shape=list(ret_type.shape))
                            type_of[id(op)] = op.type


def infer_shapes(m: builtin.Module) -> builtin.Module:
    m = deepcopy(m)
    func_map = {f.name: f for f in m.functions}
    type_of: dict[int, toy.TensorType] = {}

    # Process main first (all shapes derivable from constants)
    main = func_map.get("main")
    if main is not None:
        _infer_function(main, func_map, type_of)

    # Process remaining functions (may already be inferred via call sites)
    for func in m.functions:
        if func.name != "main":
            _infer_function(func, func_map, type_of)

    return m
