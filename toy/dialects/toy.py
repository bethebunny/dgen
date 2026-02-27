"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from dgen import Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.dialects.builtin import IndexType
from dgen.layout import FLOAT64, VOID, Array, Layout
from dgen.type import Memory
from toy.dialects.affine import ShapeType

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

toy = Dialect("toy")


@toy.type("Tensor")
@dataclass(frozen=True)
class TensorType(Type):
    """toy.Tensor<[2, 3], f64>."""

    shape: Value[ShapeType]
    dtype: Type = builtin.F64Type()

    __params__ = (("shape", ShapeType), ("dtype", Type))

    def unpack_shape(self) -> list[int]:
        """Extract concrete shape dimensions as a list of ints."""
        return list(self.shape.__constant__.unpack())

    @property
    def __layout__(self) -> Layout:
        assert self.shape.ready
        shape: Memory[ShapeType] = self.shape.__constant__
        return Array(FLOAT64, prod(shape.unpack()))


@toy.type("InferredShapeTensor")
@dataclass(frozen=True)
class InferredShapeTensor(Type):
    """toy.InferredShapeTensor<f64> — shape to be filled in by inference."""

    __layout__ = VOID
    dtype: Type = builtin.F64Type()

    __params__ = (("dtype", Type),)


@dataclass
class FunctionType(builtin.Function):
    """(toy.Tensor<(2, 3), f64>) -> ()"""

    inputs: list[Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@toy.op("transpose")
@dataclass(eq=False, kw_only=True)
class TransposeOp(Op):
    input: Value
    type: Type

    __operands__ = (("input", Type),)


@toy.op("reshape")
@dataclass(eq=False, kw_only=True)
class ReshapeOp(Op):
    input: Value
    type: Type

    __operands__ = (("input", Type),)


@toy.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type

    __operands__ = (("lhs", Type), ("rhs", Type))


@toy.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type

    __operands__ = (("lhs", Type), ("rhs", Type))


@toy.op("generic_call")
@dataclass(eq=False, kw_only=True)
class GenericCallOp(Op):
    callee: str
    args: list[Value]
    type: Type

    __operands__ = (("callee", builtin.String), ("args", Type))


@toy.op("concat")
@dataclass(eq=False, kw_only=True)
class ConcatOp(Op):
    axis: Value[IndexType]
    lhs: Value
    rhs: Value
    type: Type

    __params__ = (("axis", IndexType),)
    __operands__ = (("lhs", Type), ("rhs", Type))


@toy.op("tile")
@dataclass(eq=False, kw_only=True)
class TileOp(Op):
    count: Value[IndexType]
    input: Value
    type: Type

    __params__ = (("count", IndexType),)
    __operands__ = (("input", Type),)


@toy.op("nonzero_count")
@dataclass(eq=False, kw_only=True)
class NonzeroCountOp(Op):
    input: Value
    type: Type = builtin.IndexType()

    __operands__ = (("input", Type),)


@toy.op("dim_size")
@dataclass(eq=False, kw_only=True)
class DimSizeOp(Op):
    axis: Value[IndexType]
    input: Value
    type: Type = builtin.IndexType()

    __params__ = (("axis", IndexType),)
    __operands__ = (("input", Type),)

    def resolve_constant(self) -> int | None:
        """Return constant value if input type has a resolved shape."""
        shape = getattr(self.input.type, "shape", None)
        if shape is not None and getattr(shape, "ready", False):
            return shape.__constant__.unpack()[self.axis.__constant__.unpack()[0]]
        return None


@toy.op("print")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()

    __operands__ = (("input", Type),)
