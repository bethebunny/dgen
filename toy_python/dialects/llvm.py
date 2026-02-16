"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Op, Ssa, String, StringList
from toy_python.asm.formatting import Shape, SsaList, Sym, format_float

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
class AllocaOp(Op):
    elem_count: int


@llvm.op("gep")
class GepOp(Op):
    base: Ssa
    index: Ssa


@llvm.op("load")
class LoadOp(Op):
    ptr: Ssa


@llvm.op("store")
class StoreOp(Op):
    value: Ssa
    ptr: Ssa


@llvm.op("fadd")
class FAddOp(Op):
    lhs: Ssa
    rhs: Ssa


@llvm.op("fmul")
class FMulOp(Op):
    lhs: Ssa
    rhs: Ssa


@llvm.op("fconst")
class ConstantOp(Op):
    value: float


@llvm.op("iconst")
class IndexConstOp(Op):
    value: int


@llvm.op("add")
class AddOp(Op):
    lhs: Ssa
    rhs: Ssa


@llvm.op("mul")
class MulOp(Op):
    lhs: Ssa
    rhs: Ssa


@llvm.op("icmp")
class IcmpOp(Op):
    pred: String
    lhs: Ssa
    rhs: Ssa


@llvm.op("br")
class BrOp(Op):
    dest: String


@llvm.op("cond_br")
class CondBrOp(Op):
    cond: Ssa
    true_dest: String
    false_dest: String


@llvm.op("label")
class LabelOp(Op):
    name: String


@llvm.op("phi")
class PhiOp(Op):
    values: SsaList
    labels: StringList


@llvm.op("call")
class CallOp(Op):
    callee: Sym
    args: SsaList


