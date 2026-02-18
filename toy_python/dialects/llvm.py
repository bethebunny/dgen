"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Op, StaticString, Value
from toy_python.asm.formatting import Sym

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


@llvm.op("gep")
@dataclass(eq=False, kw_only=True)
class GepOp(Op):
    base: Value
    index: Value


@llvm.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    ptr: Value


@llvm.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    ptr: Value


@llvm.op("fadd")
@dataclass(eq=False, kw_only=True)
class FAddOp(Op):
    lhs: Value
    rhs: Value


@llvm.op("fmul")
@dataclass(eq=False, kw_only=True)
class FMulOp(Op):
    lhs: Value
    rhs: Value


@llvm.op("fconst")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op):
    value: float


@llvm.op("iconst")
@dataclass(eq=False, kw_only=True)
class IndexConstOp(Op):
    value: int


@llvm.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value


@llvm.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value


@llvm.op("icmp")
@dataclass(eq=False, kw_only=True)
class IcmpOp(Op):
    pred: StaticString
    lhs: Value
    rhs: Value


@llvm.op("br")
@dataclass(eq=False, kw_only=True)
class BrOp(Op):
    dest: StaticString


@llvm.op("cond_br")
@dataclass(eq=False, kw_only=True)
class CondBrOp(Op):
    cond: Value
    true_dest: StaticString
    false_dest: StaticString


@llvm.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: StaticString


@llvm.op("phi")
@dataclass(eq=False, kw_only=True)
class PhiOp(Op):
    values: list[Value]
    labels: list[StaticString]


@llvm.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: Sym
    args: list[Value]
