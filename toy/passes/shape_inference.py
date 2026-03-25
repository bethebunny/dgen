"""Shape inference pass: resolve InferredShapeTensor to concrete Tensor."""

from __future__ import annotations

import dgen
from dgen.dialects import builtin, function
from dgen.module import ConstantOp, Module, PackOp, _walk_all_ops
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import shape_constant, toy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


class ShapeInference(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._func_map: dict[str, function.DefineOp] = {}

    def verify_postconditions(self, module: Module) -> None:
        super().verify_postconditions(module)
        for func in module.functions:
            for op in _walk_all_ops(func):
                if isinstance(op, tuple(toy.toy.ops.values())):
                    assert not isinstance(op.type, toy.InferredShapeTensor), (
                        f"{type(op).__name__} still typed as InferredShapeTensor "
                        "after ShapeInference"
                    )

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        self._func_map = {f.name: f for f in module.functions if f.name is not None}
        if main := self._func_map.get("main"):
            self._run_block(main.body)
        for f in module.functions:
            if f.name != "main":
                self._run_block(f.body)
        return module

    def _shape(self, val: dgen.Value) -> list[int] | None:
        """Return the concrete shape of val if its type is a resolved Tensor."""
        if isinstance(val.type, toy.Tensor):
            result = val.type.shape.__constant__.to_json()
            assert isinstance(result, list)
            return result
        return None

    @lowering_for(toy.TransposeOp)
    def infer_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        if shape := self._shape(op.input):
            op.type = toy.Tensor(shape=shape_constant(list(reversed(shape))))
        return True

    @lowering_for(toy.MulOp)
    def infer_mul(self, op: toy.MulOp, rewriter: Rewriter) -> bool:
        if shape := self._shape(op.lhs):
            op.type = toy.Tensor(shape=shape_constant(shape))
        return True

    @lowering_for(toy.AddOp)
    def infer_add(self, op: toy.AddOp, rewriter: Rewriter) -> bool:
        if shape := self._shape(op.lhs):
            op.type = toy.Tensor(shape=shape_constant(shape))
        return True

    @lowering_for(toy.ConcatOp)
    def infer_concat(self, op: toy.ConcatOp, rewriter: Rewriter) -> bool:
        lhs_shape = self._shape(op.lhs)
        rhs_shape = self._shape(op.rhs)
        if lhs_shape is not None and rhs_shape is not None:
            axis = op.axis.__constant__.to_json()
            assert isinstance(axis, int)
            shape = list(lhs_shape)
            shape[axis] += rhs_shape[axis]
            op.type = toy.Tensor(shape=shape_constant(shape))
        return True

    @lowering_for(toy.TileOp)
    def infer_tile(self, op: toy.TileOp, rewriter: Rewriter) -> bool:
        if not isinstance(op.count, ConstantOp):
            return True
        count = op.count.__constant__.to_json()
        assert isinstance(count, int)
        if shape := self._shape(op.input):
            op.type = toy.Tensor(shape=shape_constant([count] + shape))
        return True

    @lowering_for(function.CallOp)
    def infer_call(self, op: function.CallOp, rewriter: Rewriter) -> bool:
        args = (
            op.arguments.values if isinstance(op.arguments, PackOp) else [op.arguments]
        )
        callee = self._func_map.get(op.callee.name)
        if callee is None:
            return True
        arg_shapes = [self._shape(a) for a in args]
        if any(s is None for s in arg_shapes):
            return True
        for param, shape in zip(callee.body.args, arg_shapes):
            param.type = toy.Tensor(shape=shape_constant(shape))
        self._run_block(callee.body)
        if isinstance(callee.body.result.type, toy.Tensor):
            ret_type = callee.body.result.type
            ret_shape = ret_type.shape.__constant__.to_json()
            callee.result = ret_type
            op.type = toy.Tensor(shape=shape_constant(ret_shape))
        return True
