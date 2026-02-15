"""Toy dialect IR types and operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from toy_python.dialects.builtin import FuncType, Nil, Type

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
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        yield (
            f"%{self.result} = Constant({format_shape(self.shape)} "
            f"{format_float_list(self.value)}) : {self.type.asm}"
        )


@dataclass
class TransposeOp:
    result: str
    input: str
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Transpose(%{self.input}) : {self.type.asm}"


@dataclass
class ReshapeOp:
    result: str
    input: str
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Reshape(%{self.input}) : {self.type.asm}"


@dataclass
class MulOp:
    result: str
    lhs: str
    rhs: str
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Mul(%{self.lhs}, %{self.rhs}) : {self.type.asm}"


@dataclass
class AddOp:
    result: str
    lhs: str
    rhs: str
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Add(%{self.lhs}, %{self.rhs}) : {self.type.asm}"


@dataclass
class GenericCallOp:
    result: str
    callee: str
    args: list[str]
    type: Type

    @property
    def asm(self) -> Iterable[str]:
        args_str = ", ".join(f"%{a}" for a in self.args)
        yield (
            f"%{self.result} = GenericCall @{self.callee}({args_str}) : "
            f"{self.type.asm}"
        )


@dataclass
class PrintOp:
    input: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"Print(%{self.input})"


@dataclass
class ReturnOp:
    value: str | None

    @property
    def asm(self) -> Iterable[str]:
        if self.value is not None:
            yield f"return %{self.value}"
        else:
            yield "return"




