"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Block, Comptime, Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.layout import INT, VOID, Array, Layout, Pointer
from dgen.type import Memory

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.type("Shape")
@dataclass(frozen=True)
class ShapeType:
    ndim: int

    @property
    def __layout__(self) -> Layout:
        return Array(INT, self.ndim)

    @classmethod
    def for_value(cls, value: object) -> ShapeType:
        assert isinstance(value, list), f"ShapeType.for_value expects list, got {type(value).__name__}"
        return cls(ndim=len(value))


def shape_memory(dims: list[int]) -> Memory:
    """Create a Memory object for a shape."""
    return Memory.from_value(ShapeType(ndim=len(dims)), dims)


@affine.type("MemRef")
@dataclass
class MemRefType:
    __layout__ = Pointer(VOID)
    shape: ShapeType
    dtype: Type = builtin.F64Type()


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@affine.op("alloc")
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: Comptime


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
