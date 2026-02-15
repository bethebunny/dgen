"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from toy_python.dialects.builtin import FuncType, Nil, Type
from toy_python.ir_format import Bare, BareList, Shape, Ssa, SsaList, Sym, op, build_tables

if TYPE_CHECKING:
    from toy_python.ir_parser import IRParser

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


@op("constant")
class ConstantOp:
    result: Ssa
    shape: Shape
    value: list[float]
    type: Type


@op("transpose")
class TransposeOp:
    result: Ssa
    input: Ssa
    type: Type


@op("reshape")
class ReshapeOp:
    result: Ssa
    input: Ssa
    type: Type


@op("mul")
class MulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa
    type: Type


@op("add")
class AddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa
    type: Type


@op("generic_call")
class GenericCallOp:
    result: Ssa
    callee: Sym
    args: SsaList
    type: Type


@op("print")
class PrintOp:
    input: Ssa


@op("return", builtin=True)
class ReturnOp:
    value: Ssa | None


# ===----------------------------------------------------------------------=== #
# Dialect tables & convenience parser
# ===----------------------------------------------------------------------=== #

DIALECT_NAME = "toy"


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


_ALL_OPS = [
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
]
OP_TABLE, KEYWORD_TABLE = build_tables(_ALL_OPS, dialect=DIALECT_NAME)
TYPE_TABLE = {"tensor": _parse_tensor_type}


def parse_toy_module(text: str):
    from toy_python.ir_parser import parse_module

    return parse_module(text)
