"""Ch5: Toy IR to Affine IR lowering."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product
from typing import Iterable

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin
from toy.dialects import affine, toy


class ToyToAffineLowering:
    def __init__(self):
        self.alloc_map: dict[dgen.Value, dgen.Value] = {}
        self.live_allocs: list[dgen.Value] = []

    def lower_module(self, m: builtin.Module) -> builtin.Module:
        functions = [self.lower_function(f) for f in m.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: builtin.FuncOp) -> builtin.FuncOp:
        self.alloc_map = {}
        self.live_allocs = []
        ops = []
        for op in f.body.ops:
            ops.extend(self.lower_op(op))
        return builtin.FuncOp(
            name=f.name,
            body=dgen.Block(ops=ops),
            type=builtin.Function(result=builtin.Nil()),
        )

    def lower_op(self, op: dgen.Op) -> Iterator[dgen.Op]:
        if isinstance(op, builtin.ConstantOp) and isinstance(op.value, list):
            yield from self._lower_constant(op)
        elif isinstance(op, toy.TransposeOp):
            yield from self._lower_transpose(op)
        elif isinstance(op, (toy.MulOp, toy.AddOp)):
            yield from self._lower_binop(op, op.lhs, op.rhs)
        elif isinstance(op, toy.ReshapeOp):
            self._lower_reshape(op)
        elif isinstance(op, toy.PrintOp):
            yield from self._lower_print(op)
        elif isinstance(op, builtin.ReturnOp):
            yield from self._lower_return(op)

    def _nested_for(
        self,
        shape: list[int],
        body_fn: Callable[[Iterable[BlockArgument]], list[dgen.Op]],
    ) -> Iterator[dgen.Op]:
        """Build nested ForOps for each dimension. body_fn(ivars) -> innermost ops."""
        ivars = [BlockArgument(type=builtin.IndexType()) for _ in shape]
        ops = list(body_fn(ivars))
        for dim, var in reversed(list(zip(shape, ivars))):
            ops = [affine.ForOp(lo=0, hi=dim, body=dgen.Block(ops=ops, args=[var]))]
        yield ops[0]

    def _lower_constant(self, op: builtin.ConstantOp) -> Iterator[dgen.Op]:
        assert isinstance(op.type, toy.TensorType)
        shape = op.type.shape

        alloc_op = affine.AllocOp(shape=shape)
        yield alloc_op
        self.live_allocs.append(alloc_op)

        values = op.value
        assert isinstance(values, list)

        for flat, nd_idx in enumerate(product(*(range(d) for d in shape))):
            cst = builtin.ConstantOp(value=values[flat], type=builtin.F64Type())
            indices = [
                builtin.ConstantOp(value=dim_val, type=builtin.IndexType())
                for dim_val in nd_idx
            ]
            yield cst
            yield from indices
            yield affine.StoreOp(value=cst, memref=alloc_op, indices=indices)

        self.alloc_map[op] = alloc_op

    def _lower_transpose(self, op: toy.TransposeOp) -> Iterator[dgen.Op]:
        assert isinstance(op.type, toy.TensorType)
        assert isinstance(op.input.type, toy.TensorType)
        in_shape = op.input.type.shape

        alloc_op = affine.AllocOp(shape=op.type.shape)
        yield alloc_op
        self.live_allocs.append(alloc_op)
        in_alloc = self.alloc_map.get(op.input, op.input)

        def body(ivars):
            load = affine.LoadOp(memref=in_alloc, indices=ivars)
            store = affine.StoreOp(
                value=load, memref=alloc_op, indices=list(reversed(ivars))
            )
            return [load, store]

        yield from self._nested_for(in_shape, body)
        self.alloc_map[op] = alloc_op

    def _lower_binop(
        self, result_op: dgen.Op, lhs_val: dgen.Value, rhs_val: dgen.Value
    ) -> Iterator[dgen.Op]:
        assert isinstance(lhs_val.type, toy.TensorType)
        shape = lhs_val.type.shape
        alloc_op = affine.AllocOp(shape=shape)
        yield alloc_op
        self.live_allocs.append(alloc_op)
        lhs_alloc = self.alloc_map.get(lhs_val, lhs_val)
        rhs_alloc = self.alloc_map.get(rhs_val, rhs_val)

        binop = {
            toy.MulOp: affine.ArithMulFOp,
            toy.AddOp: affine.ArithAddFOp,
        }

        def body(ivars):
            lv = affine.LoadOp(memref=lhs_alloc, indices=ivars)
            rv = affine.LoadOp(memref=rhs_alloc, indices=ivars)
            res = binop[type(result_op)](lhs=lv, rhs=rv)
            store = affine.StoreOp(value=res, memref=alloc_op, indices=ivars)
            return [lv, rv, res, store]

        yield from self._nested_for(shape, body)
        self.alloc_map[result_op] = alloc_op

    def _lower_reshape(self, op: toy.ReshapeOp) -> None:
        self.alloc_map[op] = self.alloc_map.get(op.input, op.input)

    def _lower_print(self, op: toy.PrintOp) -> Iterator[dgen.Op]:
        alloc = self.alloc_map.get(op.input, op.input)
        yield affine.PrintOp(input=alloc)

    def _lower_return(self, op: builtin.ReturnOp) -> Iterator[dgen.Op]:
        for alloc_val in self.live_allocs:
            yield affine.DeallocOp(input=alloc_val)
        yield builtin.ReturnOp(value=op.value)


def lower_to_affine(m: builtin.Module) -> builtin.Module:
    lowering = ToyToAffineLowering()
    return lowering.lower_module(m)
