"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.asm.formatting import Bare, BareList, Shape, Ssa, SsaList, Sym, format_float

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
class AllocaOp:
    result: Ssa
    elem_count: int


@llvm.op("gep")
class GepOp:
    result: Ssa
    base: Ssa
    index: Ssa


@llvm.op("load")
class LoadOp:
    result: Ssa
    ptr: Ssa


@llvm.op("store")
class StoreOp:
    value: Ssa
    ptr: Ssa


@llvm.op("fadd")
class FAddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@llvm.op("fmul")
class FMulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@llvm.op("fconst")
class ConstantOp:
    result: Ssa
    value: float


@llvm.op("iconst")
class IndexConstOp:
    result: Ssa
    value: int


@llvm.op("add")
class AddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@llvm.op("mul")
class MulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@llvm.op("icmp")
class IcmpOp:
    result: Ssa
    pred: Bare
    lhs: Ssa
    rhs: Ssa


@llvm.op("br")
class BrOp:
    dest: Bare


@llvm.op("cond_br")
class CondBrOp:
    cond: Ssa
    true_dest: Bare
    false_dest: Bare


@llvm.op("label")
class LabelOp:
    name: Bare


@llvm.op("phi")
class PhiOp:
    result: Ssa
    values: SsaList
    labels: BareList


@llvm.op("call")
class CallOp:
    result: Ssa | None
    callee: Sym
    args: SsaList


