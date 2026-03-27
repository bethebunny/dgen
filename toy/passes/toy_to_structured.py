"""Toy IR to structured IR lowering (ndbuffer + control_flow + algebra)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, memory
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean, Float64
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from dgen.dialects import control_flow
from toy.dialects import ndbuffer, shape_constant, toy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


def _empty_pack() -> PackOp:
    return PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))


def _index_pack(*values: dgen.Value) -> PackOp:
    return PackOp(values=list(values), type=builtin.List(element_type=Index()))


def _nested_for(
    shape: Sequence[int],
    body_fn: Callable[[Sequence[dgen.Value]], dgen.Value],
    captures: Sequence[dgen.Value] = (),
) -> control_flow.ForOp:
    """Build nested ForOps for each dimension, innermost first.

    captures: values from the enclosing scope referenced by body_fn (e.g.
    allocs, tensor constants, function parameters). Outer loop IVs are
    also added as captures at each nesting level.
    """
    ivars = [BlockArgument(type=Index()) for _ in shape]
    innermost: dgen.Value = body_fn(ivars)
    for depth in range(len(shape) - 1, -1, -1):
        # Outer IVs are captures, not threaded block args.
        outer_ivars = ivars[:depth]
        all_captures = list(captures) + outer_ivars
        innermost = control_flow.ForOp(
            lower_bound=Index().constant(0),
            upper_bound=Index().constant(shape[depth]),
            initial_arguments=_empty_pack(),
            body=dgen.Block(
                result=innermost,
                args=[ivars[depth]],
                captures=all_captures,
            ),
        )
    return innermost  # type: ignore[return-value]


class ToyToStructured(Pass):
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
        assert isinstance(val.type, (toy.Tensor, ndbuffer.NDBuffer))
        result = val.type.shape.__constant__.to_json()
        assert isinstance(result, list)
        return result

    def _alloc(self, shape_val: dgen.Value) -> ndbuffer.AllocOp:
        return ndbuffer.AllocOp(
            shape=shape_val, type=ndbuffer.NDBuffer(shape=shape_val)
        )

    @lowering_for(toy.TransposeOp)
    def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.type, toy.Tensor)
        in_shape = self._shape(op.input)
        alloc = self._alloc(op.type.shape)

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            load = ndbuffer.LoadOp(memref=op.input, indices=_index_pack(*ivars))
            return ndbuffer.StoreOp(
                value=load, memref=alloc, indices=_index_pack(*reversed(list(ivars)))
            )

        loop = _nested_for(in_shape, body, captures=[alloc, op.input])
        rewriter.replace_uses(op, ChainOp(lhs=alloc, rhs=loop, type=alloc.type))
        return True

    @lowering_for(toy.MulOp)
    def lower_mul(self, op: toy.MulOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, algebra.MultiplyOp, rewriter)

    @lowering_for(toy.AddOp)
    def lower_add(self, op: toy.AddOp, rewriter: Rewriter) -> bool:
        return self._lower_binop(op, op.lhs, op.rhs, algebra.AddOp, rewriter)

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
            lhs_elem = ndbuffer.LoadOp(memref=lhs, indices=idx)
            res = cls(
                left=lhs_elem,
                right=ndbuffer.LoadOp(memref=rhs, indices=idx),
                type=lhs_elem.type,
            )
            return ndbuffer.StoreOp(value=res, memref=alloc, indices=idx)

        loop = _nested_for(shape, body, captures=[alloc, lhs, rhs])
        rewriter.replace_uses(op, ChainOp(lhs=alloc, rhs=loop, type=alloc.type))
        return True

    @lowering_for(toy.ReshapeOp)
    def lower_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, op.input)
        return True

    @lowering_for(toy.PrintOp)
    def lower_print(self, op: toy.PrintOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, ndbuffer.PrintMemrefOp(input=op.input))
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
            load = ndbuffer.LoadOp(memref=op.input, indices=_index_pack(*ivars[1:]))
            return ndbuffer.StoreOp(
                value=load, memref=alloc, indices=_index_pack(*ivars)
            )

        loop = _nested_for(out_shape, body, captures=[alloc, op.input])
        rewriter.replace_uses(op, ChainOp(lhs=alloc, rhs=loop, type=alloc.type))
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
            return ndbuffer.StoreOp(
                value=ndbuffer.LoadOp(memref=op.lhs, indices=idx),
                memref=alloc,
                indices=idx,
            )

        lhs_loop = _nested_for(lhs_shape, lhs_body, captures=[alloc, op.lhs])
        offset = ConstantOp(value=lhs_shape[axis], type=Index())

        def rhs_body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            shifted = list(ivars)
            shifted[axis] = algebra.AddOp(left=ivars[axis], right=offset, type=Index())
            return ndbuffer.StoreOp(
                value=ndbuffer.LoadOp(memref=op.rhs, indices=_index_pack(*ivars)),
                memref=alloc,
                indices=_index_pack(*shifted),
            )

        rhs_loop = _nested_for(
            rhs_shape,
            rhs_body,
            captures=[alloc, op.rhs, offset],
        )
        after_lhs = ChainOp(lhs=alloc, rhs=lhs_loop, type=alloc.type)
        rewriter.replace_uses(op, ChainOp(lhs=after_lhs, rhs=rhs_loop, type=alloc.type))
        return True

    @lowering_for(toy.NonzeroCountOp)
    def lower_nonzero_count(self, op: toy.NonzeroCountOp, rewriter: Rewriter) -> bool:
        """Count nonzero elements: stack-alloc accumulator, nested loop, load/compare/add."""
        shape = self._shape(op.input)
        reference_type = memory.Reference(element_type=Index())
        accumulator = memory.StackAllocateOp(element_type=Index(), type=reference_type)
        initial_store = memory.StoreOp(
            value=ConstantOp(value=0, type=Index()), ptr=accumulator
        )
        initialized = ChainOp(lhs=accumulator, rhs=initial_store, type=reference_type)
        zero = ConstantOp(value=0.0, type=Float64())

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            element = ndbuffer.LoadOp(memref=op.input, indices=_index_pack(*ivars))
            nonzero = algebra.NotEqualOp(left=element, right=zero, type=Boolean())
            current = memory.LoadOp(ptr=initialized, type=Index())
            updated = algebra.AddOp(
                left=current,
                right=algebra.CastOp(input=nonzero, type=Index()),
                type=Index(),
            )
            return memory.StoreOp(value=updated, ptr=initialized)

        loop = _nested_for(
            shape,
            body,
            captures=[initialized, zero, op.input],
        )
        after_loop = ChainOp(lhs=initialized, rhs=loop, type=reference_type)
        result = memory.LoadOp(ptr=after_loop, type=Index())
        rewriter.replace_uses(op, result)
        return True
