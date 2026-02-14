"""Toy dialect IR types and operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

from toy_python import asm


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


AnyType = Union[UnrankedTensorType, RankedTensorType]


@dataclass
class FunctionType:
    """(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>"""

    inputs: list[AnyType]
    result: AnyType | None


def type_to_string(t: AnyType) -> str:
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
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        yield (
            f"%{self.result} = Constant({format_shape(self.shape)} "
            f"{format_float_list(self.value)}) : {self.type}"
        )


@dataclass
class TransposeOp:
    result: str
    input: str
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Transpose(%{self.input}) : {self.type}"


@dataclass
class ReshapeOp:
    result: str
    input: str
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Reshape(%{self.input}) : {self.type}"


@dataclass
class MulOp:
    result: str
    lhs: str
    rhs: str
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Mul(%{self.lhs}, %{self.rhs}) : {self.type}"


@dataclass
class AddOp:
    result: str
    lhs: str
    rhs: str
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Add(%{self.lhs}, %{self.rhs}) : {self.type}"


@dataclass
class GenericCallOp:
    result: str
    callee: str
    args: list[str]
    type: AnyType

    @property
    def asm(self) -> Iterable[str]:
        args_str = ", ".join(f"%{a}" for a in self.args)
        yield (
            f"%{self.result} = GenericCall @{self.callee}({args_str}) : "
            f"{self.type}"
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


AnyOp = Union[
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
class Value:
    """A block argument (function parameter)."""

    name: str
    type: AnyType


@dataclass
class Block:
    args: list[Value]
    ops: list[AnyOp]

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from asm.indent(op.asm)


@dataclass
class FuncOp:
    name: str
    func_type: FunctionType
    body: Block

    @property
    def asm(self) -> Iterable[str]:
        args = ", ".join(f"%{a.name}: {a.type}" for a in self.body.args)
        header = f"%{self.name} = function ({args})"
        if self.func_type.result is not None:
            header += f" -> {self.func_type.result}"
        header += ":"
        yield header
        yield from self.body.asm


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        yield "from toy use *"
        for function in self.functions:
            yield ""
            yield from function.asm
