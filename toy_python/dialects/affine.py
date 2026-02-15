"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Op
from toy_python.asm.formatting import Bare, BareList, Shape, Ssa, SsaList, Sym

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
# Operations
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.op("alloc")
class AllocOp:
    result: Ssa
    shape: Shape


@affine.op("dealloc")
class DeallocOp:
    result: Ssa
    input: Ssa


@affine.op("load")
class LoadOp:
    result: Ssa
    memref: Ssa
    indices: BareList


@affine.op("store")
class StoreOp:
    result: Ssa
    value: Ssa
    memref: Ssa
    indices: BareList


@affine.op("arith_constant")
class ArithConstantOp:
    result: Ssa
    value: float


@affine.op("index_constant")
class IndexConstantOp:
    result: Ssa
    value: int


@affine.op("mul_f")
class ArithMulFOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@affine.op("add_f")
class ArithAddFOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@affine.op("print_memref")
class PrintOp:
    result: Ssa
    input: Ssa


@affine.op("for")
class ForOp:
    result: Ssa
    var_name: Ssa
    lo: int
    hi: int
    body: list[Op]


