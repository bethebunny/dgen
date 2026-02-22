"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Block, Dialect, Op, Type, Value
from dgen.dialects import builtin

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.type("MemRef")
@dataclass(frozen=True)
class MemRefType:
    shape: list[int]
    dtype: Type = builtin.F64Type()


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@affine.op("alloc")
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: list[int]


@affine.op("dealloc")
@dataclass(eq=False, kw_only=True)
class DeallocOp(Op):
    input: Value
    type: Type = builtin.Nil()


@affine.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    memref: Value
    indices: list[Value]
    type: Type = builtin.Nil()


@affine.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    memref: Value
    indices: list[Value]
    type: Type = builtin.Nil()


@affine.op("mul_f")
@dataclass(eq=False, kw_only=True)
class ArithMulFOp(Op):
    lhs: Value
    rhs: Value
    type: Type = builtin.Nil()


@affine.op("add_f")
@dataclass(eq=False, kw_only=True)
class ArithAddFOp(Op):
    lhs: Value
    rhs: Value
    type: Type = builtin.Nil()


@affine.op("print_memref")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()


@affine.op("for")
@dataclass(eq=False, kw_only=True)
class ForOp(Op):
    lo: int
    hi: int
    body: Block
    type: Type = builtin.Nil()
