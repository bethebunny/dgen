"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

from toy_python import asm

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class MemRefType:
    shape: list[int]

    def __str__(self) -> str:
        return f"memref<{'x'.join(str(d) for d in self.shape)}xf64>"


@dataclass
class IndexType:
    def __str__(self) -> str:
        return "index"


@dataclass
class F64Type:
    def __str__(self) -> str:
        return "f64"


AnyAffineType = Union[MemRefType, IndexType, F64Type]


# ===----------------------------------------------------------------------=== #
# Formatting helpers
# ===----------------------------------------------------------------------=== #


def format_float(v: float) -> str:
    iv = int(v)
    if float(iv) == v:
        return f"{iv}.0"
    return str(v)


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@dataclass
class AllocOp:
    result: str
    shape: list[int]

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = Alloc<{'x'.join(str(d) for d in self.shape)}>()"


@dataclass
class DeallocOp:
    input: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"Dealloc(%{self.input})"


@dataclass
class AffineLoadOp:
    result: str
    memref: str
    indices: list[str]

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = AffineLoad %{self.memref}[{', '.join(self.indices)}]"


@dataclass
class AffineStoreOp:
    value: str
    memref: str
    indices: list[str]

    @property
    def asm(self) -> Iterable[str]:
        yield f"AffineStore %{self.value}, %{self.memref}[{', '.join(self.indices)}]"


@dataclass
class ArithConstantOp:
    result: str
    value: float

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = ArithConstant({format_float(self.value)})"


@dataclass
class IndexConstantOp:
    result: str
    value: int

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = IndexConstant({self.value})"


@dataclass
class ArithMulFOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = MulF(%{self.lhs}, %{self.rhs})"


@dataclass
class ArithAddFOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = AddF(%{self.lhs}, %{self.rhs})"


@dataclass
class AffinePrintOp:
    input: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"PrintMemRef(%{self.input})"


@dataclass
class AffineReturnOp:
    value: str | None

    @property
    def asm(self) -> Iterable[str]:
        if self.value is not None:
            yield f"return %{self.value}"
        else:
            yield "return"


@dataclass
class AffineForOp:
    var_name: str
    lo: int
    hi: int
    body: list[AnyAffineOp]

    @property
    def asm(self) -> Iterable[str]:
        yield f"AffineFor %{self.var_name} = {self.lo} to {self.hi}:"
        for child_op in self.body:
            yield from asm.indent(child_op.asm)


AnyAffineOp = Union[
    AllocOp,
    DeallocOp,
    AffineLoadOp,
    AffineStoreOp,
    AffineForOp,
    ArithConstantOp,
    IndexConstantOp,
    ArithMulFOp,
    ArithAddFOp,
    AffinePrintOp,
    AffineReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #


@dataclass
class AffineValue:
    name: str
    type: AnyAffineType


@dataclass
class AffineBlock:
    args: list[AffineValue]
    ops: list[AnyAffineOp]

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from asm.indent(op.asm)


@dataclass
class AffineFuncOp:
    name: str
    body: AffineBlock

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.name} = function ():"
        yield from self.body.asm


@dataclass
class AffineModule:
    functions: list[AffineFuncOp]

    @property
    def asm(self) -> Iterable[str]:
        for function in self.functions:
            yield from function.asm
