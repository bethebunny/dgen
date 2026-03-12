"""Shape inference pass: resolve InferredShapeTensor to concrete Tensor."""

from __future__ import annotations

import dgen
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
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


class ShapeInference(Pass):
    allow_unregistered_ops = True  # skip ops we don't infer

    def __init__(self) -> None:
        self.type_of: dict[dgen.Value, toy.Tensor] = {}
        self.func_map: dict[str, builtin.FunctionOp] = {}

    def run(self, module: Module) -> Module:
        """Override run to build func_map and process main first."""
        self.func_map = {f.name: f for f in module.functions if f.name is not None}
        self.type_of = {}
        main = self.func_map.get("main")
        if main is not None:
            self._seed_and_run(main)
        for func in module.functions:
            if func.name != "main":
                self._seed_and_run(func)
        return module

    def _seed_and_run(self, func: builtin.FunctionOp) -> None:
        for arg in func.body.args:
            if isinstance(arg.type, toy.Tensor):
                self.type_of[arg] = arg.type
        self._run_block(func.body)

    @lowering_for(ConstantOp)
    def infer_constant(self, op: ConstantOp, rewriter: Rewriter) -> bool:
        if isinstance(op.type, toy.Tensor):
            self.type_of[op] = op.type
        return True

    @lowering_for(toy.ReshapeOp)
    def infer_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        if isinstance(op.type, toy.Tensor):
            self.type_of[op] = op.type
        return True

    @lowering_for(toy.TransposeOp)
    def infer_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        src = self.type_of.get(op.input)
        if src is not None:
            t = toy.Tensor(
                shape=shape_constant(list(reversed(src.shape.__constant__.to_json())))
            )
            op.type = t
            self.type_of[op] = t
        return True

    @lowering_for(toy.MulOp)
    def infer_mul(self, op: toy.MulOp, rewriter: Rewriter) -> bool:
        src = self.type_of.get(op.lhs)
        if src is not None:
            t = toy.Tensor(shape=shape_constant(src.shape.__constant__.to_json()))
            op.type = t
            self.type_of[op] = t
        return True

    @lowering_for(toy.AddOp)
    def infer_add(self, op: toy.AddOp, rewriter: Rewriter) -> bool:
        src = self.type_of.get(op.lhs)
        if src is not None:
            t = toy.Tensor(shape=shape_constant(src.shape.__constant__.to_json()))
            op.type = t
            self.type_of[op] = t
        return True

    @lowering_for(toy.ConcatOp)
    def infer_concat(self, op: toy.ConcatOp, rewriter: Rewriter) -> bool:
        lhs = self.type_of.get(op.lhs)
        rhs = self.type_of.get(op.rhs)
        if lhs is not None and rhs is not None:
            lhs_dims = lhs.shape.__constant__.to_json()
            rhs_dims = rhs.shape.__constant__.to_json()
            shape = list(lhs_dims)
            axis = op.axis.__constant__.to_json()
            assert isinstance(axis, int)
            shape[axis] = lhs_dims[axis] + rhs_dims[axis]
            t = toy.Tensor(shape=shape_constant(shape))
            op.type = t
            self.type_of[op] = t
        return True

    @lowering_for(toy.TileOp)
    def infer_tile(self, op: toy.TileOp, rewriter: Rewriter) -> bool:
        src = self.type_of.get(op.input)
        if src is not None:
            count_val = (
                _resolve_index_value(op.count)
                if isinstance(op.count, dgen.Value)
                else None
            )
            if count_val is not None:
                t = toy.Tensor(
                    shape=shape_constant([count_val] + src.shape.__constant__.to_json())
                )
                op.type = t
                self.type_of[op] = t
        return True

    @lowering_for(builtin.CallOp)
    def infer_call(self, op: builtin.CallOp, rewriter: Rewriter) -> bool:
        args_list = op.args.values if isinstance(op.args, PackOp) else [op.args]
        resolved = [self.type_of.get(a) for a in args_list]
        arg_types = [t for t in resolved if t is not None]
        if len(arg_types) == len(resolved):
            callee = self.func_map.get(op.callee.name)
            if callee is not None:
                for param, atype in zip(callee.body.args, arg_types):
                    param.type = atype
                    self.type_of[param] = atype
                self._seed_and_run(callee)
                ret_op = callee.body.ops[-1]
                if isinstance(ret_op, builtin.ReturnOp) and not isinstance(
                    ret_op.value, builtin.Nil
                ):
                    ret_type = self.type_of.get(ret_op.value)
                    if ret_type is not None:
                        callee.result = ret_type
                        op.type = toy.Tensor(
                            shape=shape_constant(ret_type.shape.__constant__.to_json())
                        )
                        self.type_of[op] = op.type
        return True


def infer_shapes(m: Module) -> Module:
    return ShapeInference().run(m)
