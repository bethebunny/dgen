"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from toy_python import asm
from toy_python.dialects.builtin import Op

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class MemRefType:
    shape: list[int]

    @property
    def asm(self) -> str:
        return f"memref<{'x'.join(str(d) for d in self.shape)}xf64>"


@dataclass
class IndexType:
    @property
    def asm(self) -> str:
        return "index"


@dataclass
class F64Type:
    @property
    def asm(self) -> str:
        return "f64"


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
class LoadOp:
    result: str
    memref: str
    indices: list[str]

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = AffineLoad %{self.memref}[{', '.join(self.indices)}]"


@dataclass
class StoreOp:
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
class PrintOp:
    input: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"PrintMemRef(%{self.input})"


@dataclass
class ReturnOp:
    value: str | None

    @property
    def asm(self) -> Iterable[str]:
        yield "return" if self.value is None else f"return %{self.value}"


@dataclass
class ForOp:
    var_name: str
    lo: int
    hi: int
    body: list[Op]

    @property
    def asm(self) -> Iterable[str]:
        yield f"AffineFor %{self.var_name} = {self.lo} to {self.hi}:"
        for child_op in self.body:
            yield from asm.indent(child_op.asm)
