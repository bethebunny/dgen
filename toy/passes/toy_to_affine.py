"""Toy IR to Affine IR lowering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, index
from dgen.dialects.index import Index
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from dgen.dialects import control_flow
from toy.dialects import affine, memory, shape_constant, toy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))


def _index_pack(*values: dgen.Value) -> PackOp:
    return PackOp(values=list(values), type=builtin.List(element_type=Index()))


def _chain(*values: dgen.Value, type: dgen.Type) -> dgen.Value:
    """Left-associative chain ensuring all values are visited before the last."""
    result: dgen.Value = values[0]
    for v in values[1:]:
        result = builtin.ChainOp(lhs=result, rhs=v, type=type)
    return result


def _nested_for(
    shape: Sequence[int],
    body_fn: Callable[[Sequence[dgen.Value]], dgen.Value],
    outer_block_args: Sequence[BlockArgument] = (),
) -> affine.ForOp:
    """Build nested ForOps for each dimension, innermost first.

    Outer-loop ivars are threaded as explicit extra block arguments on inner
    body blocks so that every block remains closed (no out-of-scope captures).
    outer_block_args: function-parameter BlockArguments referenced in body_fn;
    threaded at every nesting level so they remain in scope.
    """
    ivars = [BlockArgument(type=Index()) for _ in shape]
    innermost: dgen.Value = body_fn(ivars)
    for depth in range(len(shape) - 1, -1, -1):
        outer_ivars = ivars[:depth]
        captured: list[dgen.Value] = outer_ivars + list(outer_block_args)
        init_pack = (
            PackOp(values=captured, type=builtin.List(element_type=builtin.Nil()))
            if captured
            else _EMPTY_PACK
        )
        innermost = control_flow.ForOp(
            lower_bound=Index().constant(0),
            upper_bound=Index().constant(shape[depth]),
            initial_arguments=init_pack,
            body=dgen.Block(
                result=innermost,
                args=[ivars[depth]] + captured,
            ),
        )
    return innermost  # type: ignore[return-value]


def _get_block_args(*values: dgen.Value) -> list[BlockArgument]:
    """Return only the BlockArgument instances from values."""
    return [v for v in values if isinstance(v, BlockArgument)]


class ToyToAffine(Pass):
    allow_unregistered_ops = True

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        return Module(
            ops=[
                self._lower_function(op) if isinstance(op, FunctionOp) else op
                for op in module.ops
            ]
        )

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self._run_block(f.body)
        return FunctionOp(
            name=f.name, body=f.body, result=f.result, type=Function(result=f.result)
        )

    def _shape(self, val: dgen.Value) -> list[int]:
        assert isinstance(val.type, (toy.Tensor, memory.MemRef))
        result = val.type.shape.__constant__.to_json()
        assert isinstance(result, list)
        return result

    def _alloc(self, shape_val: dgen.Value) -> memory.AllocOp:
        return memory.AllocOp(shape=shape_val, type=memory.MemRef(shape=shape_val))

    @lowering_for(toy.TransposeOp)
    def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.type, toy.Tensor)
        in_shape = self._shape(op.input)
        alloc = self._alloc(op.type.shape)

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            load = memory.LoadOp(memref=op.input, indices=_index_pack(*ivars))
            return memory.StoreOp(
                value=load, memref=alloc, indices=_index_pack(*reversed(list(ivars)))
            )

        loop = _nested_for(in_shape, body, _get_block_args(op.input))
        rewriter.replace_uses(op, _chain(alloc, op.input, loop, type=alloc.type))
        return True

    @lowering_for(toy.MulOp)
    def lower_mul(self, op: toy.MulOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, affine.MulFOp, rewriter)

    @lowering_for(toy.AddOp)
    def lower_add(self, op: toy.AddOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, affine.AddFOp, rewriter)

    def _lower_binop(
        self,
        op: dgen.Op,
        lhs: dgen.Value,
        rhs: dgen.Value,
        cls: type,
        rewriter: Rewriter,
    ) -> bool:
        shape = self._shape(lhs)
        alloc = self._alloc(lhs.type.shape)

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            idx = _index_pack(*ivars)
            res = cls(
                lhs=memory.LoadOp(memref=lhs, indices=idx),
                rhs=memory.LoadOp(memref=rhs, indices=idx),
            )
            return memory.StoreOp(value=res, memref=alloc, indices=idx)

        loop = _nested_for(shape, body, _get_block_args(lhs, rhs))
        rewriter.replace_uses(op, _chain(alloc, lhs, rhs, loop, type=alloc.type))
        return True

    @lowering_for(toy.ReshapeOp)
    def lower_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, op.input)
        return True

    @lowering_for(toy.PrintOp)
    def lower_print(self, op: toy.PrintOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, memory.PrintMemrefOp(input=op.input))
        return True

    @lowering_for(toy.DimSizeOp)
    def lower_dim_size(self, op: toy.DimSizeOp, rewriter: Rewriter) -> bool:
        shape = self._shape(op.input)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)
        rewriter.replace_uses(op, ConstantOp(value=shape[axis], type=Index()))
        return True

    @lowering_for(toy.TileOp)
    def lower_tile(self, op: toy.TileOp, rewriter: Rewriter) -> bool:
        count = op.count.__constant__.to_json()
        assert isinstance(count, int)
        in_shape = self._shape(op.input)
        out_shape = [count] + in_shape
        alloc = self._alloc(shape_constant(out_shape))

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            load = memory.LoadOp(memref=op.input, indices=_index_pack(*ivars[1:]))
            return memory.StoreOp(value=load, memref=alloc, indices=_index_pack(*ivars))

        loop = _nested_for(out_shape, body, _get_block_args(op.input))
        rewriter.replace_uses(op, _chain(alloc, op.input, loop, type=alloc.type))
        return True

    @lowering_for(toy.ConcatOp)
    def lower_concat(self, op: toy.ConcatOp, rewriter: Rewriter) -> bool:
        lhs_shape = self._shape(op.lhs)
        rhs_shape = self._shape(op.rhs)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)
        out_shape = list(lhs_shape)
        out_shape[axis] += rhs_shape[axis]
        alloc = self._alloc(shape_constant(out_shape))

        def lhs_body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            idx = _index_pack(*ivars)
            return memory.StoreOp(
                value=memory.LoadOp(memref=op.lhs, indices=idx),
                memref=alloc,
                indices=idx,
            )

        lhs_loop = _nested_for(lhs_shape, lhs_body, _get_block_args(op.lhs))
        offset = ConstantOp(value=lhs_shape[axis], type=Index())

        def rhs_body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            shifted = list(ivars)
            shifted[axis] = index.AddOp(left=ivars[axis], right=offset)
            return memory.StoreOp(
                value=memory.LoadOp(memref=op.rhs, indices=_index_pack(*ivars)),
                memref=alloc,
                indices=_index_pack(*shifted),
            )

        rhs_loop = _nested_for(rhs_shape, rhs_body, _get_block_args(op.rhs))
        rewriter.replace_uses(
            op, _chain(alloc, op.lhs, lhs_loop, op.rhs, rhs_loop, type=alloc.type)
        )
        return True
