"""Shape inference pass: resolve InferredShapeTensor to concrete Tensor."""

from __future__ import annotations

import dgen
from dgen.dialects import function
from dgen.ir.traversal import all_blocks, all_values
from dgen.builtins import PackOp
from dgen.type import Constant
from dgen.passes.pass_ import Pass, lowering_for
from toy.dialects import shape_constant, toy


class ShapeInference(Pass):
    allow_unregistered_ops = True

    def verify_postconditions(self, value: dgen.Value) -> None:
        super().verify_postconditions(value)
        for op in all_values(value):
            assert not isinstance(op.type, toy.InferredShapeTensor), (
                f"{type(op).__name__} still typed as InferredShapeTensor "
                "after ShapeInference"
            )

    def _shape(self, val: dgen.Value) -> list[int] | None:
        """Return the concrete shape of val if its type is a resolved Tensor."""
        if isinstance(val.type, toy.Tensor):
            result = val.type.shape.__constant__.to_json()
            assert isinstance(result, list)
            return result
        return None

    @lowering_for(toy.TransposeOp)
    def infer_transpose(self, op: toy.TransposeOp) -> dgen.Value | None:
        if shape := self._shape(op.input):
            op.type = toy.Tensor(shape=shape_constant(list(reversed(shape))))
        return op

    @lowering_for(toy.MulOp)
    def infer_mul(self, op: toy.MulOp) -> dgen.Value | None:
        if shape := self._shape(op.lhs):
            op.type = toy.Tensor(shape=shape_constant(shape))
        return op

    @lowering_for(toy.AddOp)
    def infer_add(self, op: toy.AddOp) -> dgen.Value | None:
        if shape := self._shape(op.lhs):
            op.type = toy.Tensor(shape=shape_constant(shape))
        return op

    @lowering_for(toy.ConcatOp)
    def infer_concat(self, op: toy.ConcatOp) -> dgen.Value | None:
        lhs_shape = self._shape(op.lhs)
        rhs_shape = self._shape(op.rhs)
        if lhs_shape is not None and rhs_shape is not None:
            axis = op.axis.__constant__.to_json()
            assert isinstance(axis, int)
            shape = list(lhs_shape)
            shape[axis] += rhs_shape[axis]
            op.type = toy.Tensor(shape=shape_constant(shape))
        return op

    @lowering_for(toy.TileOp)
    def infer_tile(self, op: toy.TileOp) -> dgen.Value | None:
        if not isinstance(op.count, Constant):
            return op
        count = op.count.__constant__.to_json()
        assert isinstance(count, int)
        if shape := self._shape(op.input):
            op.type = toy.Tensor(shape=shape_constant([count] + shape))
        return op

    @lowering_for(function.CallOp)
    def infer_call(self, op: function.CallOp) -> dgen.Value | None:
        args = (
            list(op.arguments) if isinstance(op.arguments, PackOp) else [op.arguments]
        )
        callee = op.callee
        if not isinstance(callee, function.FunctionOp):
            return op
        arg_shapes = [self._shape(a) for a in args]
        if any(s is None for s in arg_shapes):
            return op
        for param, shape in zip(callee.body.args, arg_shapes):
            param.type = toy.Tensor(shape=shape_constant(shape))
        # Walk callee with updated arg types. The callee is a capture of
        # the caller's body, so the pass framework's walk stops at it —
        # callee bodies are processed lazily, from each call site.
        for b in all_blocks(callee):
            for v in list(b.values):
                if (result := self._dispatch_handlers(v)) is not None:
                    b.replace_uses_of(v, result)
        if isinstance(callee.body.result.type, toy.Tensor):
            ret_type = callee.body.result.type
            ret_shape = ret_type.shape.__constant__.to_json()
            callee.result_type = ret_type
            op.type = toy.Tensor(shape=shape_constant(ret_shape))
        return op
