"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from dgen import Block, Constant, Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.layout import INT, VOID, Array, Layout, Pointer

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.type("Shape")
@dataclass(frozen=True)
class ShapeType(Type):
    rank: Annotated[Value[builtin.IndexType], Constant]

    __constant_fields__ = (("rank", builtin.IndexType),)

    @property
    def __layout__(self) -> Layout:
        assert self.rank.ready
        return Array(INT, self.rank.__constant__.unpack()[0])

    @classmethod
    def for_value(cls, value: object) -> ShapeType:
        if isinstance(value, Constant):
            assert isinstance(value.type, builtin.IndexType)
            return cls(rank=value.type.constant(value.__constant__.unpack()[0]))
        assert isinstance(value, list)
        return cls(rank=builtin.IndexType().constant(len(value)))


def shape_constant(dims: list[int]) -> Constant:
    """Create a Constant[ShapeType] from a list of dims."""
    rank = builtin.IndexType().constant(len(dims))
    return ShapeType(rank=rank).constant(dims)


@affine.type("MemRef")
@dataclass
class MemRefType(Type):
    __layout__ = Pointer(VOID)

    shape: Annotated[Value[ShapeType], Constant]
    dtype: Type = builtin.F64Type()

    __constant_fields__ = (("shape", ShapeType), ("dtype", Type))


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@affine.op("alloc")
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: Annotated[Value[ShapeType], Constant]


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
