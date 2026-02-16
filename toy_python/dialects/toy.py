"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Function, Nil, Op, Type, Value, String, StringList
from toy_python.asm.formatting import Shape, Sym

if TYPE_CHECKING:
    from toy_python.asm.parser import IRParser

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class UnrankedTensorType:
    """tensor<*xf64>"""

    @property
    def asm(self) -> str:
        return "tensor<*xf64>"


@dataclass
class RankedTensorType:
    """tensor<2x3xf64>"""

    shape: list[int]

    @property
    def asm(self) -> str:
        return "tensor<" + "x".join(str(d) for d in self.shape) + "xf64>"


@dataclass
class FunctionType(Function):
    """(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>"""

    inputs: list[Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

toy = Dialect("toy")


@toy.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op):
    shape: Shape
    value: list[float]
    type: Type


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


@toy.type("tensor")
def _parse_tensor_type(parser: IRParser) -> UnrankedTensorType | RankedTensorType:
    parser.expect("<")
    if parser.peek() == "*":
        parser.expect("*xf64>")
        return UnrankedTensorType()
    shape = [parser.parse_int()]
    while parser.peek() == "x":
        parser.expect("x")
        if parser.peek() == "f":
            break
        shape.append(parser.parse_int())
    parser.expect("f64>")
    return RankedTensorType(shape=shape)
