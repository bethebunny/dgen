"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Op, Type, Value
from dgen.asm.formatting import Sym
from dgen.dialect import Dialect
from dgen.dialects.builtin import Nil, StaticString

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class PtrType:
    @property
    def asm(self) -> str:
        return "ptr"


@dataclass
class IntType:
    bits: int

    @property
    def asm(self) -> str:
        return f"i{self.bits}"


@dataclass
class FloatType:
    @property
    def asm(self) -> str:
        return "f64"


@dataclass
class VoidType:
    @property
    def asm(self) -> str:
        return "void"


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

llvm = Dialect("llvm")


@llvm.op("alloca")
@dataclass(eq=False, kw_only=True)
class AllocaOp(Op):
    elem_count: int
    type: Type = Nil()


@llvm.op("gep")
@dataclass(eq=False, kw_only=True)
class GepOp(Op):
    base: Value
    index: Value
    type: Type = Nil()


@llvm.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    ptr: Value
    type: Type = Nil()


@llvm.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    ptr: Value
    type: Type = Nil()


@llvm.op("fadd")
@dataclass(eq=False, kw_only=True)
class FAddOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("fmul")
@dataclass(eq=False, kw_only=True)
class FMulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("icmp")
@dataclass(eq=False, kw_only=True)
class IcmpOp(Op):
    pred: StaticString
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("br")
@dataclass(eq=False, kw_only=True)
class BrOp(Op):
    dest: StaticString
    type: Type = Nil()


@llvm.op("cond_br")
@dataclass(eq=False, kw_only=True)
class CondBrOp(Op):
    cond: Value
    true_dest: StaticString
    false_dest: StaticString
    type: Type = Nil()


@llvm.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: StaticString
    type: Type = Nil()


@llvm.op("phi")
@dataclass(eq=False, kw_only=True)
class PhiOp(Op):
    values: list[Value]
    labels: list[StaticString]
    type: Type = Nil()


@llvm.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: Sym
    args: list[Value]
    type: Type = Nil()
