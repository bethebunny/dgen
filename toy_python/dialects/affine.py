"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.asm.formatting import Shape
from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Block, Op, Value

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
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: Shape


@affine.op("dealloc")
@dataclass(eq=False, kw_only=True)
class DeallocOp(Op):
    input: Value


@affine.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    memref: Value
    indices: list[Value]


@affine.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    memref: Value
    indices: list[Value]


@affine.op("arith_constant")
@dataclass(eq=False, kw_only=True)
class ArithConstantOp(Op):
    value: float


@affine.op("index_constant")
@dataclass(eq=False, kw_only=True)
class IndexConstantOp(Op):
    value: int


@affine.op("mul_f")
@dataclass(eq=False, kw_only=True)
class ArithMulFOp(Op):
    lhs: Value
    rhs: Value


@affine.op("add_f")
@dataclass(eq=False, kw_only=True)
class ArithAddFOp(Op):
    lhs: Value
    rhs: Value


@affine.op("print_memref")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value


@affine.op("for")
@dataclass(eq=False, kw_only=True)
class ForOp(Op):
    lo: int
    hi: int
    body: Block
