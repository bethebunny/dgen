"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from toy_python.ir_format import Bare, BareList, Shape, Ssa, SsaList, Sym, op, build_tables, format_float

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


@op("alloca")
class AllocaOp:
    result: Ssa
    elem_count: int


@op("gep")
class GepOp:
    result: Ssa
    base: Ssa
    index: Ssa


@op("load")
class LoadOp:
    result: Ssa
    ptr: Ssa


@op("store")
class StoreOp:
    value: Ssa
    ptr: Ssa


@op("fadd")
class FAddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("fmul")
class FMulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("fconst")
class ConstantOp:
    result: Ssa
    value: float


@op("iconst")
class IndexConstOp:
    result: Ssa
    value: int


@op("add")
class AddOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("mul")
class MulOp:
    result: Ssa
    lhs: Ssa
    rhs: Ssa


@op("icmp")
class IcmpOp:
    result: Ssa
    pred: Bare
    lhs: Ssa
    rhs: Ssa


@op("br")
class BrOp:
    dest: Bare


@op("cond_br")
class CondBrOp:
    cond: Ssa
    true_dest: Bare
    false_dest: Bare


@op("label")
class LabelOp:
    name: Bare


@op("phi")
class PhiOp:
    result: Ssa
    values: SsaList
    labels: BareList


@op("call")
class CallOp:
    result: Ssa | None
    callee: Sym
    args: SsaList


@op("return")
class ReturnOp:
    value: Ssa | None


# ===----------------------------------------------------------------------=== #
# Dialect tables & convenience parser
# ===----------------------------------------------------------------------=== #

_ALL_OPS = [
    AllocaOp, GepOp, LoadOp, StoreOp, FAddOp, FMulOp, ConstantOp,
    IndexConstOp, AddOp, MulOp, IcmpOp, BrOp, CondBrOp, LabelOp,
    PhiOp, CallOp, ReturnOp,
]
OP_TABLE, KEYWORD_TABLE = build_tables(_ALL_OPS)
TYPE_TABLE: dict = {}


def parse_llvm_module(text: str):
    from toy_python.ir_parser import parse_module

    return parse_module(text, ops=OP_TABLE, keywords=KEYWORD_TABLE, types=TYPE_TABLE)
