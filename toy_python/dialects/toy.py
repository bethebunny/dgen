"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Function, Nil, Op, Type, Value
from toy_python.asm.formatting import Sym

if TYPE_CHECKING:
    from toy_python.asm.parser import IRParser

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class TensorType:
    """toy.Tensor[(2, 3), f64]."""

    shape: list[int]

    @property
    def asm(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"toy.Tensor[({dims}), f64]"


@dataclass
class InferredShapeTensor:
    """toy.InferredShapeTensor[f64] — shape to be filled in by inference."""

    dtype: str = "f64"

    @property
    def asm(self) -> str:
        return f"toy.InferredShapeTensor[{self.dtype}]"


@dataclass
class FunctionType(Function):
    """(toy.Tensor[(2, 3), f64]) -> ()"""

    inputs: list[Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

toy = Dialect("toy")
TensorType._dialect = toy
InferredShapeTensor._dialect = toy


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
    callee: Sym
    args: list[Value]
    type: Type


@toy.op("print")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value


# ===----------------------------------------------------------------------=== #
# Type parser
# ===----------------------------------------------------------------------=== #


@toy.type("Tensor")
def _parse_tensor_type(parser: IRParser) -> TensorType:
    parser.expect("[(")
    shape = [parser.parse_int()]
    while parser.peek() == ",":
        parser.expect(",")
        parser.skip_whitespace()
        shape.append(parser.parse_int())
    parser.expect(")")
    parser.expect(",")
    parser.skip_whitespace()
    parser.expect("f64]")
    return TensorType(shape=shape)


@toy.type("InferredShapeTensor")
def _parse_inferred_type(parser: IRParser) -> InferredShapeTensor:
    parser.expect("[")
    dtype = parser.parse_identifier()
    parser.expect("]")
    return InferredShapeTensor(dtype=dtype)
