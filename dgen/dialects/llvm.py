"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Op, Type, Value
from dgen.dialect import Dialect
from dgen.dialects.builtin import Nil
from dgen.layout import FLOAT64, INT, VOID, Pointer

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class PtrType:
    __layout__ = Pointer(VOID)

    @property
    def asm(self) -> str:
        return "ptr"


@dataclass
class IntType:
    __layout__ = INT
    bits: int

    @property
    def asm(self) -> str:
        return f"i{self.bits}"


@dataclass
class FloatType:
    __layout__ = FLOAT64

    @property
    def asm(self) -> str:
        return "f64"


@dataclass
class VoidType:
    __layout__ = VOID

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
    pred: str
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("br")
@dataclass(eq=False, kw_only=True)
class BrOp(Op):
    dest: str
    type: Type = Nil()


@llvm.op("cond_br")
@dataclass(eq=False, kw_only=True)
class CondBrOp(Op):
    cond: Value
    true_dest: str
    false_dest: str
    type: Type = Nil()


@llvm.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: str
    type: Type = Nil()


@llvm.op("phi")
@dataclass(eq=False, kw_only=True)
class PhiOp(Op):
    values: list[Value]
    labels: list[str]
    type: Type = Nil()


@llvm.op("fcmp")
@dataclass(eq=False, kw_only=True)
class FcmpOp(Op):
    pred: str
    lhs: Value
    rhs: Value
    type: Type = Nil()


@llvm.op("zext")
@dataclass(eq=False, kw_only=True)
class ZextOp(Op):
    input: Value
    type: Type = Nil()


@llvm.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: str
    args: list[Value]
    type: Type = Nil()
