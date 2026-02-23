"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, cast

from dgen import Comptime, Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.layout import FLOAT64, VOID, Array, Layout

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

toy = Dialect("toy")


@toy.type("Tensor")
@dataclass
class TensorType:
    """toy.Tensor[(2, 3), f64]."""

    shape: list[int]
    dtype: Type = builtin.F64Type()

    @property
    def __layout__(self) -> Layout:
        return Array(FLOAT64, prod(self.shape))


@toy.type("InferredShapeTensor")
@dataclass(frozen=True)
class InferredShapeTensor:
    """toy.InferredShapeTensor[f64] — shape to be filled in by inference."""

    __layout__ = VOID
    dtype: Type = builtin.F64Type()


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


@toy.op("reshape")
@dataclass(eq=False, kw_only=True)
class ReshapeOp(Op):
    input: Value
    type: Type


@toy.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@toy.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@toy.op("generic_call")
@dataclass(eq=False, kw_only=True)
class GenericCallOp(Op):
    callee: str
    args: list[Value]
    type: Type


@toy.op("concat")
@dataclass(eq=False, kw_only=True)
class ConcatOp(Op):
    lhs: Value
    rhs: Value
    axis: int
    type: Type


@toy.op("tile")
@dataclass(eq=False, kw_only=True)
class TileOp(Op):
    input: Value
    count: Comptime
    type: Type


@toy.op("nonzero_count")
@dataclass(eq=False, kw_only=True)
class NonzeroCountOp(Op):
    input: Value
    type: Type = builtin.IndexType()


@toy.op("dim_size")
@dataclass(eq=False, kw_only=True)
class DimSizeOp(Op):
    input: Value
    axis: int
    type: Type = builtin.IndexType()

    def resolve_constant(self) -> int | None:
        """Return constant value if input type has a resolved shape."""
        if hasattr(self.input.type, "shape"):
            return cast(Any, self.input.type).shape[self.axis]
        return None


@toy.op("print")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()
