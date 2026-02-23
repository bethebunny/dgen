"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from dgen import Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.layout import FLOAT64, Array

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
    def __layout__(self):
        return Array(FLOAT64, prod(self.shape))


@toy.type("InferredShapeTensor")
@dataclass(frozen=True)
class InferredShapeTensor:
    """toy.InferredShapeTensor[f64] — shape to be filled in by inference."""

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
    count: Value
    type: Type


@toy.op("print")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()
