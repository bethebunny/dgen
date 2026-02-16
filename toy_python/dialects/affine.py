"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Block, Op, Value, String, StringList
from toy_python.asm.formatting import Shape, Sym

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


@affine.type("index")
def _parse_index_type(_parser):
    return IndexType()


@affine.op("alloc")
class AllocOp(Op):
    shape: Shape


@affine.op("dealloc")
class DeallocOp(Op):
    input: Value


@affine.op("load")
class LoadOp(Op):
    memref: Value
    indices: list[Value]


@affine.op("store")
class StoreOp(Op):
    value: Value
    memref: Value
    indices: list[Value]


@affine.op("arith_constant")
class ArithConstantOp(Op):
    value: float


@affine.op("index_constant")
class IndexConstantOp(Op):
    value: int


@affine.op("mul_f")
class ArithMulFOp(Op):
    lhs: Value
    rhs: Value


@affine.op("add_f")
class ArithAddFOp(Op):
    lhs: Value
    rhs: Value


@affine.op("print_memref")
class PrintOp(Op):
    input: Value


@affine.op("for")
class ForOp(Op):
    lo: int
    hi: int
    body: Block
