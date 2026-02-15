"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import FuncType, Nil, Type
from toy_python.asm.formatting import Bare, BareList, Shape, Ssa, SsaList, Sym

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
class FunctionType(FuncType):
    """(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>"""

    inputs: list[Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

toy = Dialect("toy")


@toy.op("constant")
class ConstantOp:
    result: Ssa
    shape: Shape
    value: list[float]
    type: Type


@toy.op("transpose")
class TransposeOp:
    result: Ssa
    input: Ssa
    type: Type


@toy.op("reshape")
class ReshapeOp:
    result: Ssa
    input: Ssa
    type: Type


@toy.op("mul")
class MulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa
    type: Type


@toy.op("add")
class AddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa
    type: Type


@toy.op("generic_call")
class GenericCallOp:
    result: Ssa
    callee: Sym
    args: SsaList
    type: Type


@toy.op("print")
class PrintOp:
    input: Ssa


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


def parse_toy_module(text: str):
    from toy_python.asm.parser import parse_module

    return parse_module(text)
