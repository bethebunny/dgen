"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Annotated

from dgen import Constant, Dialect, Op, Type, Value
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
    """toy.Tensor([2, 3], f64)."""

    shape: Annotated[Value[ShapeType], Constant]
    dtype: Type = builtin.F64Type()

    __constant_fields__ = (("shape", ShapeType), ("dtype", Type))

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
    """toy.InferredShapeTensor[f64] — shape to be filled in by inference."""

    __layout__ = VOID
    dtype: Type = builtin.F64Type()

    __constant_fields__ = (("dtype", Type),)


@dataclass
class FunctionType(builtin.Function):
    """(toy.Tensor[(2, 3), f64]) -> ()"""

    inputs: list[Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@toy.op("transpose")
@dataclass(eq=False, kw_only=True)
class TransposeOp(Op):
    input: Value
    type: Type

    __arg_fields__ = ("input",)


@toy.op("reshape")
@dataclass(eq=False, kw_only=True)
class ReshapeOp(Op):
    input: Value
    type: Type

    __arg_fields__ = ("input",)


@toy.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type

    __arg_fields__ = ("lhs", "rhs")


@toy.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type

    __arg_fields__ = ("lhs", "rhs")


@toy.op("generic_call")
@dataclass(eq=False, kw_only=True)
class GenericCallOp(Op):
    callee: str
    args: list[Value]
    type: Type

    __arg_fields__ = ("callee", "args")


@toy.op("concat")
@dataclass(eq=False, kw_only=True)
class ConcatOp(Op):
    lhs: Value
    rhs: Value
    axis: int
    type: Type

    __arg_fields__ = ("lhs", "rhs", "axis")


@toy.op("tile")
@dataclass(eq=False, kw_only=True)
class TileOp(Op):
    input: Value
    count: Annotated[Value[IndexType], Constant]
    type: Type

    __arg_fields__ = ("input", "count")


@toy.op("nonzero_count")
@dataclass(eq=False, kw_only=True)
class NonzeroCountOp(Op):
    input: Value
    type: Type = builtin.IndexType()

    __arg_fields__ = ("input",)


@toy.op("dim_size")
@dataclass(eq=False, kw_only=True)
class DimSizeOp(Op):
    input: Value
    axis: int
    type: Type = builtin.IndexType()

    __arg_fields__ = ("input", "axis")

    def resolve_constant(self) -> int | None:
        """Return constant value if input type has a resolved shape."""
        shape = getattr(self.input.type, "shape", None)
        if shape is not None and getattr(shape, "ready", False):
            return shape.__constant__.unpack()[self.axis]
        return None


@toy.op("print")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()

    __arg_fields__ = ("input",)
