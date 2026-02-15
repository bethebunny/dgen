"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.dialect import Dialect
from toy_python.dialects.builtin import Ssa, String, StringList
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
    result: Ssa
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
    pred: String
    lhs: Ssa
    rhs: Ssa


@llvm.op("br")
class BrOp:
    result: Ssa
    dest: String


@llvm.op("cond_br")
class CondBrOp:
    result: Ssa
    cond: Ssa
    true_dest: String
    false_dest: String


@llvm.op("label")
class LabelOp:
    result: Ssa
    name: String


@llvm.op("phi")
class PhiOp:
    result: Ssa
    values: SsaList
    labels: StringList


@llvm.op("call")
class CallOp:
    result: Ssa
    callee: Sym
    args: SsaList


