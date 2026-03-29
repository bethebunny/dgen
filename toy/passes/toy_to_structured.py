"""Toy IR to structured IR lowering (ndbuffer + control_flow + algebra)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, memory
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean, Float64
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, pack
from dgen.passes.pass_ import Pass, lowering_for
from dgen.dialects import control_flow
from dgen.dialects import ndbuffer
from toy.dialects import shape_constant, toy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


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
            initial_arguments=pack(),
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
    def lower_transpose(self, op: toy.TransposeOp) -> dgen.Value | None:
        assert isinstance(op.type, toy.Tensor)
        in_shape = self._shape(op.input)
        alloc = self._alloc(op.type.shape)

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            load = ndbuffer.LoadOp(memref=op.input, indices=pack(ivars))
            return ndbuffer.StoreOp(
                value=load, memref=alloc, indices=pack(reversed(list(ivars)))
            )

        loop = _nested_for(in_shape, body, captures=[alloc, op.input])
        return ChainOp(lhs=alloc, rhs=loop, type=alloc.type)

    @lowering_for(toy.MulOp)
    def lower_mul(self, op: toy.MulOp) -> dgen.Value | None:
        return self._lower_binop(op.lhs, op.rhs, algebra.MultiplyOp)

    @lowering_for(toy.AddOp)
    def lower_add(self, op: toy.AddOp) -> dgen.Value | None:
        return self._lower_binop(op.lhs, op.rhs, algebra.AddOp)

    def _lower_binop(
        self,
        lhs: dgen.Value,
        rhs: dgen.Value,
        cls: type,
    ) -> dgen.Value:
        shape = self._shape(lhs)
        alloc = self._alloc(lhs.type.shape)

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            idx = pack(ivars)
            lhs_elem = ndbuffer.LoadOp(memref=lhs, indices=idx)
            res = cls(
                left=lhs_elem,
                right=ndbuffer.LoadOp(memref=rhs, indices=idx),
                type=lhs_elem.type,
            )
            return ndbuffer.StoreOp(value=res, memref=alloc, indices=idx)

        loop = _nested_for(shape, body, captures=[alloc, lhs, rhs])
        return ChainOp(lhs=alloc, rhs=loop, type=alloc.type)

    @lowering_for(toy.ReshapeOp)
    def lower_reshape(self, op: toy.ReshapeOp) -> dgen.Value | None:
        return op.input

    @lowering_for(toy.PrintOp)
    def lower_print(self, op: toy.PrintOp) -> dgen.Value | None:
        return ndbuffer.PrintMemrefOp(input=op.input)

    @lowering_for(toy.DimSizeOp)
    def lower_dim_size(self, op: toy.DimSizeOp) -> dgen.Value | None:
        shape = self._shape(op.input)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)
        return ConstantOp(value=shape[axis], type=Index())

    @lowering_for(toy.TileOp)
    def lower_tile(self, op: toy.TileOp) -> dgen.Value | None:
        count = op.count.__constant__.to_json()
        assert isinstance(count, int)
        in_shape = self._shape(op.input)
        out_shape = [count] + in_shape
        alloc = self._alloc(shape_constant(out_shape))

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            load = ndbuffer.LoadOp(memref=op.input, indices=pack(ivars[1:]))
            return ndbuffer.StoreOp(value=load, memref=alloc, indices=pack(ivars))

        loop = _nested_for(out_shape, body, captures=[alloc, op.input])
        return ChainOp(lhs=alloc, rhs=loop, type=alloc.type)

    @lowering_for(toy.ConcatOp)
    def lower_concat(self, op: toy.ConcatOp) -> dgen.Value | None:
        lhs_shape = self._shape(op.lhs)
        rhs_shape = self._shape(op.rhs)
        axis = op.axis.__constant__.to_json()
        assert isinstance(axis, int)
        out_shape = list(lhs_shape)
        out_shape[axis] += rhs_shape[axis]
        alloc = self._alloc(shape_constant(out_shape))

        def lhs_body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            idx = pack(ivars)
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
                value=ndbuffer.LoadOp(memref=op.rhs, indices=pack(ivars)),
                memref=alloc,
                indices=pack(shifted),
            )

        rhs_loop = _nested_for(
            rhs_shape,
            rhs_body,
            captures=[alloc, op.rhs, offset],
        )
        after_lhs = ChainOp(lhs=alloc, rhs=lhs_loop, type=alloc.type)
        return ChainOp(lhs=after_lhs, rhs=rhs_loop, type=alloc.type)

    @lowering_for(toy.NonzeroCountOp)
    def lower_nonzero_count(self, op: toy.NonzeroCountOp) -> dgen.Value | None:
        """Count nonzero elements: stack-alloc accumulator, nested loop, load/compare/add."""
        shape = self._shape(op.input)
        reference_type = memory.Reference(element_type=Index())
        accumulator = memory.StackAllocateOp(element_type=Index(), type=reference_type)
        initial_store = memory.StoreOp(
            mem=accumulator, value=ConstantOp(value=0, type=Index()), ptr=accumulator
        )
        zero = ConstantOp(value=0.0, type=Float64())

        def body(ivars: Sequence[dgen.Value]) -> dgen.Value:
            element = ndbuffer.LoadOp(memref=op.input, indices=pack(ivars))
            nonzero = algebra.NotEqualOp(left=element, right=zero, type=Boolean())
            current = memory.LoadOp(mem=initial_store, ptr=accumulator, type=Index())
            updated = algebra.AddOp(
                left=current,
                right=algebra.CastOp(input=nonzero, type=Index()),
                type=Index(),
            )
            return memory.StoreOp(mem=current, value=updated, ptr=accumulator)

        loop = _nested_for(
            shape,
            body,
            captures=[accumulator, initial_store, zero, op.input],
        )
        return memory.LoadOp(mem=loop, ptr=accumulator, type=Index())
