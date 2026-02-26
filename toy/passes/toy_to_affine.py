"""Ch5: Toy IR to Affine IR lowering."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import dgen
from dgen.block import BlockArgument
from dgen.dialects.builtin import Nil
from dgen.dialects import builtin
from dgen.layout import Array
from toy.dialects import affine, toy


class ToyToAffineLowering:
    def __init__(self) -> None:
        self.alloc_map: dict[dgen.Value, dgen.Value] = {}
        self.live_allocs: list[dgen.Value] = []

    def lower_module(self, m: builtin.Module) -> builtin.Module:
        functions = [self.lower_function(f) for f in m.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: builtin.FuncOp) -> builtin.FuncOp:
        self.alloc_map = {}
        self.live_allocs = []
        # Register block args (function parameters) as themselves
        for arg in f.body.args:
            self.alloc_map[arg] = arg
        ops = []
        for op in f.body.ops:
            ops.extend(self.lower_op(op))
        return builtin.FuncOp(
            name=f.name,
            body=dgen.Block(ops=ops, args=f.body.args),
            type=builtin.Function(result=f.type.result),
        )

    def lower_op(self, op: dgen.Op) -> Iterator[dgen.Op]:
        if isinstance(op, builtin.ConstantOp) and isinstance(op.value.layout, Array):
            yield from self._lower_constant(op)
        elif isinstance(op, builtin.ConstantOp):
            yield op
        elif isinstance(op, builtin.AddIndexOp):
            new_op = builtin.AddIndexOp(
                lhs=self.alloc_map.get(op.lhs, op.lhs),
                rhs=self.alloc_map.get(op.rhs, op.rhs),
            )
            yield new_op
            self.alloc_map[op] = new_op
        elif isinstance(op, toy.TransposeOp):
            yield from self._lower_transpose(op)
        elif isinstance(op, (toy.MulOp, toy.AddOp)):
            yield from self._lower_binop(op, op.lhs, op.rhs)
        elif isinstance(op, toy.ReshapeOp):
            self._lower_reshape(op)
        elif isinstance(op, toy.NonzeroCountOp):
            new_op = toy.NonzeroCountOp(
                input=self.alloc_map.get(op.input, op.input),
            )
            yield new_op
            self.alloc_map[op] = new_op
        elif isinstance(op, toy.TileOp):
            yield from self._lower_tile(op)
        elif isinstance(op, toy.ConcatOp):
            yield from self._lower_concat(op)
        elif isinstance(op, toy.PrintOp):
            yield from self._lower_print(op)
        elif isinstance(op, builtin.ReturnOp):
            yield from self._lower_return(op)

    def _make_alloc(self, shape: dgen.Value) -> affine.AllocOp:
        """Create an AllocOp from a shape Constant."""
        return affine.AllocOp(
            shape=shape,
            type=affine.MemRefType(shape=shape),
        )

    def _nested_for(
        self,
        shape: list[int],
        body_fn: Callable[[list[dgen.Value]], list[dgen.Op]],
    ) -> Iterator[dgen.Op]:
        """Build nested ForOps for each dimension. body_fn(ivars) -> innermost ops."""
        ivars: list[dgen.Value] = [
            BlockArgument(type=builtin.IndexType()) for _ in shape
        ]
        ops = list(body_fn(ivars))
        for dim, var in reversed(list(zip(shape, ivars))):
            ops = [affine.ForOp(lo=0, hi=dim, body=dgen.Block(ops=ops, args=[var]))]
        yield ops[0]

    def _lower_constant(self, op: builtin.ConstantOp) -> Iterator[dgen.Op]:
        new_const = builtin.ConstantOp(value=op.value, type=op.type)
        yield new_const
        self.alloc_map[op] = new_const
        self.live_allocs.append(new_const)

    def _lower_transpose(self, op: toy.TransposeOp) -> Iterator[dgen.Op]:
        assert isinstance(op.type, toy.TensorType)
        assert isinstance(op.input.type, toy.TensorType)
        in_shape = op.input.type.unpack_shape()

        alloc_op = self._make_alloc(op.type.shape)
        yield alloc_op
        self.live_allocs.append(alloc_op)
        in_alloc = self.alloc_map.get(op.input, op.input)

        def body(ivars: list[dgen.Value]) -> list[dgen.Op]:
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
        shape = lhs_val.type.unpack_shape()
        alloc_op = self._make_alloc(lhs_val.type.shape)
        yield alloc_op
        self.live_allocs.append(alloc_op)
        lhs_alloc = self.alloc_map.get(lhs_val, lhs_val)
        rhs_alloc = self.alloc_map.get(rhs_val, rhs_val)

        binop = {
            toy.MulOp: affine.ArithMulFOp,
            toy.AddOp: affine.ArithAddFOp,
        }

        def body(ivars: list[dgen.Value]) -> list[dgen.Op]:
            lv = affine.LoadOp(memref=lhs_alloc, indices=ivars)
            rv = affine.LoadOp(memref=rhs_alloc, indices=ivars)
            res = binop[type(result_op)](lhs=lv, rhs=rv)
            store = affine.StoreOp(value=res, memref=alloc_op, indices=ivars)
            return [lv, rv, res, store]

        yield from self._nested_for(shape, body)
        self.alloc_map[result_op] = alloc_op

    def _lower_tile(self, op: toy.TileOp) -> Iterator[dgen.Op]:
        assert isinstance(op.count, builtin.ConstantOp)
        count = op.count.__constant__.unpack()[0]
        assert isinstance(count, int)
        assert isinstance(op.input.type, toy.TensorType)
        input_shape = op.input.type.unpack_shape()
        output_shape: list[int] = [count] + input_shape

        alloc_op = self._make_alloc(affine.shape_constant(output_shape))
        yield alloc_op
        self.live_allocs.append(alloc_op)
        in_alloc = self.alloc_map.get(op.input, op.input)

        def body(ivars: list[dgen.Value]) -> list[dgen.Op]:
            inner_ivars = ivars[1:]  # indices into input tensor
            load = affine.LoadOp(memref=in_alloc, indices=inner_ivars)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=ivars)
            return [load, store]

        yield from self._nested_for(output_shape, body)
        self.alloc_map[op] = alloc_op

    def _lower_concat(self, op: toy.ConcatOp) -> Iterator[dgen.Op]:
        assert isinstance(op.lhs.type, toy.TensorType)
        assert isinstance(op.rhs.type, toy.TensorType)
        lhs_shape = op.lhs.type.unpack_shape()
        rhs_shape = op.rhs.type.unpack_shape()
        axis = op.axis

        output_shape = list(lhs_shape)
        output_shape[axis] = lhs_shape[axis] + rhs_shape[axis]

        alloc_op = self._make_alloc(affine.shape_constant(output_shape))
        yield alloc_op
        self.live_allocs.append(alloc_op)
        lhs_alloc = self.alloc_map.get(op.lhs, op.lhs)
        rhs_alloc = self.alloc_map.get(op.rhs, op.rhs)

        # Copy lhs into output[0:lhs_shape[axis], ...]
        def lhs_body(ivars: list[dgen.Value]) -> list[dgen.Op]:
            load = affine.LoadOp(memref=lhs_alloc, indices=ivars)
            store = affine.StoreOp(value=load, memref=alloc_op, indices=ivars)
            return [load, store]

        yield from self._nested_for(lhs_shape, lhs_body)

        # Copy rhs into output[lhs_shape[axis]:, ...] with offset along axis
        offset_const = builtin.ConstantOp(
            value=lhs_shape[axis], type=builtin.IndexType()
        )
        yield offset_const

        def rhs_body(ivars: list[dgen.Value]) -> list[dgen.Op]:
            load = affine.LoadOp(memref=rhs_alloc, indices=ivars)
            offset_idx = builtin.AddIndexOp(lhs=ivars[axis], rhs=offset_const)
            out_indices: list[dgen.Value] = list(ivars)
            out_indices[axis] = offset_idx
            store = affine.StoreOp(value=load, memref=alloc_op, indices=out_indices)
            return [load, offset_idx, store]

        yield from self._nested_for(rhs_shape, rhs_body)
        self.alloc_map[op] = alloc_op

    def _lower_reshape(self, op: toy.ReshapeOp) -> None:
        self.alloc_map[op] = self.alloc_map.get(op.input, op.input)

    def _lower_print(self, op: toy.PrintOp) -> Iterator[dgen.Op]:
        alloc = self.alloc_map.get(op.input, op.input)
        yield affine.PrintOp(input=alloc)

    def _lower_return(self, op: builtin.ReturnOp) -> Iterator[dgen.Op]:
        for alloc_val in self.live_allocs:
            yield affine.DeallocOp(input=alloc_val)
        if isinstance(op.value, Nil):
            yield builtin.ReturnOp()
        else:
            yield builtin.ReturnOp(value=self.alloc_map.get(op.value, op.value))


def lower_to_affine(m: builtin.Module) -> builtin.Module:
    lowering = ToyToAffineLowering()
    return lowering.lower_module(m)
