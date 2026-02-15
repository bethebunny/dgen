"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialects.builtin import Op
from toy_python.ir_format import Bare, BareList, Shape, Ssa, SsaList, Sym, op, build_tables

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


@op("Alloc")
class AllocOp:
    result: Ssa
    shape: Shape


@op("Dealloc")
class DeallocOp:
    input: Ssa


@op("AffineLoad")
class LoadOp:
    result: Ssa
    memref: Ssa
    indices: BareList


@op("AffineStore")
class StoreOp:
    value: Ssa
    memref: Ssa
    indices: BareList


@op("ArithConstant")
class ArithConstantOp:
    result: Ssa
    value: float


@op("IndexConstant")
class IndexConstantOp:
    result: Ssa
    value: int


@op("MulF")
class ArithMulFOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("AddF")
class ArithAddFOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("PrintMemRef")
class PrintOp:
    input: Ssa


@op("Return")
class ReturnOp:
    value: Ssa | None


@op("AffineFor")
class ForOp:
    var_name: Ssa
    lo: int
    hi: int
    body: list[Op]


# ===----------------------------------------------------------------------=== #
# Dialect tables & convenience parser
# ===----------------------------------------------------------------------=== #

_ALL_OPS = [
    AllocOp, DeallocOp, LoadOp, StoreOp, ArithConstantOp, IndexConstantOp,
    ArithMulFOp, ArithAddFOp, PrintOp, ReturnOp, ForOp,
]
OP_TABLE, KEYWORD_TABLE = build_tables(_ALL_OPS)
TYPE_TABLE: dict = {}


def parse_affine_module(text: str):
    from toy_python.ir_parser import parse_module

    return parse_module(text, ops=OP_TABLE, keywords=KEYWORD_TABLE, types=TYPE_TABLE)
