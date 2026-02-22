"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Block, Dialect, Op, Type, Value
from dgen.asm.formatting import Shape
from dgen.dialects import builtin

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class MemRefType:
    shape: list[int]

    @property
    def asm(self) -> str:
        dims = ", ".join(str(d) for d in self.shape)
        return f"affine.MemRef[({dims}), f64]"


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

affine = Dialect("affine")


@affine.type("MemRef")
def _parse_memref_type(parser) -> MemRefType:
    parser.expect("[(")
    shape = [parser.parse_int()]
    while parser.peek() == ",":
        parser.expect(",")
        parser.skip_whitespace()
        shape.append(parser.parse_int())
    parser.expect(")")
    parser.expect(",")
    parser.skip_whitespace()
    parser.expect("f64]")
    return MemRefType(shape=shape)


@affine.op("alloc")
@dataclass(eq=False, kw_only=True)
class AllocOp(Op):
    shape: Shape


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
