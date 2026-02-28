"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dgen import Block, Constant, Dialect, Op, Type, Value
from dgen.dialects import builtin
from dgen.dialects.builtin import HasSingleBlock, IndexType
from dgen.layout import INT, VOID, Array, Layout, Pointer

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.type("Shape")
@dataclass(frozen=True)
class ShapeType(Type):
    rank: Value[IndexType]

    __params__ = (("rank", IndexType),)

    @property
    def __layout__(self) -> Layout:
        assert self.rank.ready
        return Array(INT, self.rank.__constant__.unpack()[0])

    @classmethod
    def for_value(cls, value: object) -> ShapeType:
        if isinstance(value, Constant):
            assert isinstance(value.type, IndexType)
            return cls(rank=value.type.constant(value.__constant__.unpack()[0]))
        assert isinstance(value, list)
        return cls(rank=IndexType().constant(len(value)))


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[ShapeType] from a list of dims."""
    rank = IndexType().constant(len(dims))
    return ShapeType(rank=rank).constant(dims)


@affine.type("MemRef")
@dataclass
class MemRefType(Type):
    __layout__ = Pointer(VOID)

    shape: Value[ShapeType]
    dtype: Type = builtin.F64Type()

    __params__ = (("shape", ShapeType), ("dtype", Type))


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@affine.op("alloc")
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: Value[ShapeType]

    __operands__ = (("shape", ShapeType),)


@affine.op("dealloc")
@dataclass(eq=False, kw_only=True)
class DeallocOp(Op):
    input: Value
    type: Type = builtin.Nil()

    __operands__ = (("input", Type),)


@affine.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    memref: Value
    indices: Value
    type: Type = builtin.Nil()

    __operands__ = (("memref", Type), ("indices", IndexType))


@affine.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    memref: Value
    indices: Value
    type: Type = builtin.Nil()

    __operands__ = (("value", Type), ("memref", Type), ("indices", IndexType))


@affine.op("mul_f")
@dataclass(eq=False, kw_only=True)
class ArithMulFOp(Op):
    lhs: Value
    rhs: Value
    type: Type = builtin.Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@affine.op("add_f")
@dataclass(eq=False, kw_only=True)
class ArithAddFOp(Op):
    lhs: Value
    rhs: Value
    type: Type = builtin.Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@affine.op("print_memref")
@dataclass(eq=False, kw_only=True)
class PrintOp(Op):
    input: Value
    type: Type = builtin.Nil()

    __operands__ = (("input", Type),)


@affine.op("for")
@dataclass(eq=False, kw_only=True)
class ForOp(HasSingleBlock, Op):
    lo: Value[IndexType]
    hi: Value[IndexType]
    body: Block
    type: Type = builtin.Nil()

    __params__ = (("lo", IndexType), ("hi", IndexType))
    __blocks__ = ("body",)
