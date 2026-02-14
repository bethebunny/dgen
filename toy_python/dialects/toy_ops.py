"""Toy dialect IR types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class UnrankedTensorType:
    """tensor<*xf64>"""

    def __str__(self) -> str:
        return "tensor<*xf64>"


@dataclass
class RankedTensorType:
    """tensor<2x3xf64>"""

    shape: list[int]

    def __str__(self) -> str:
        return "tensor<" + "x".join(str(d) for d in self.shape) + "xf64>"


AnyToyType = Union[UnrankedTensorType, RankedTensorType]


@dataclass
class FunctionType:
    """(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>"""

    inputs: list[AnyToyType]
    result: AnyToyType | None


def type_to_string(t: AnyToyType) -> str:
    return str(t)


# ===----------------------------------------------------------------------=== #
# Formatting helpers
# ===----------------------------------------------------------------------=== #


def format_shape(shape: list[int]) -> str:
    return "<" + "x".join(str(d) for d in shape) + ">"


def format_float_list(values: list[float]) -> str:
    parts = []
    for v in values:
        iv = int(v)
        if float(iv) == v:
            parts.append(f"{iv}.0")
        else:
            parts.append(str(v))
    return "[" + ", ".join(parts) + "]"


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@dataclass
class ConstantOp:
    result: str
    value: list[float]
    shape: list[int]
    type: AnyToyType


@dataclass
class TransposeOp:
    result: str
    input: str
    type: AnyToyType


@dataclass
class ReshapeOp:
    result: str
    input: str
    type: AnyToyType


@dataclass
class MulOp:
    result: str
    lhs: str
    rhs: str
    type: AnyToyType


@dataclass
class AddOp:
    result: str
    lhs: str
    rhs: str
    type: AnyToyType


@dataclass
class GenericCallOp:
    result: str
    callee: str
    args: list[str]
    type: AnyToyType


@dataclass
class PrintOp:
    input: str


@dataclass
class ReturnOp:
    value: str | None


AnyToyOp = Union[
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    GenericCallOp,
    PrintOp,
    ReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #


@dataclass
class ToyValue:
    """A block argument (function parameter)."""

    name: str
    type: AnyToyType


@dataclass
class Block:
    args: list[ToyValue]
    ops: list[AnyToyOp]


@dataclass
class FuncOp:
    name: str
    func_type: FunctionType
    body: Block


@dataclass
class Module:
    functions: list[FuncOp]
