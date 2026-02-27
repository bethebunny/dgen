"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Op, Type, Value
from dgen.dialect import Dialect
from dgen.dialects.builtin import IndexType, Nil, String
from dgen.layout import FLOAT64, INT, VOID, Pointer

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class PtrType(Type):
    __layout__ = Pointer(VOID)

    @property
    def asm(self) -> str:
        return "ptr"


@dataclass
class IntType(Type):
    __layout__ = INT
    bits: int

    @property
    def asm(self) -> str:
        return f"i{self.bits}"


@dataclass
class FloatType(Type):
    __layout__ = FLOAT64

    @property
    def asm(self) -> str:
        return "f64"


@dataclass
class VoidType(Type):
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
    elem_count: Value[IndexType]
    type: Type = Nil()

    __params__ = (("elem_count", IndexType),)


@llvm.op("gep")
@dataclass(eq=False, kw_only=True)
class GepOp(Op):
    base: Value
    index: Value
    type: Type = Nil()

    __operands__ = (("base", Type), ("index", Type))


@llvm.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    ptr: Value
    type: Type = Nil()

    __operands__ = (("ptr", Type),)


@llvm.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    ptr: Value
    type: Type = Nil()

    __operands__ = (("value", Type), ("ptr", Type))


@llvm.op("fadd")
@dataclass(eq=False, kw_only=True)
class FAddOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("fmul")
@dataclass(eq=False, kw_only=True)
class FMulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("icmp")
@dataclass(eq=False, kw_only=True)
class IcmpOp(Op):
    pred: str
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("pred", String), ("lhs", Type), ("rhs", Type))


@llvm.op("br")
@dataclass(eq=False, kw_only=True)
class BrOp(Op):
    dest: str
    type: Type = Nil()

    __operands__ = (("dest", String),)


@llvm.op("cond_br")
@dataclass(eq=False, kw_only=True)
class CondBrOp(Op):
    cond: Value
    true_dest: str
    false_dest: str
    type: Type = Nil()

    __operands__ = (("cond", Type), ("true_dest", String), ("false_dest", String))


@llvm.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: str
    type: Type = Nil()

    __operands__ = (("label_name", String),)


@llvm.op("phi")
@dataclass(eq=False, kw_only=True)
class PhiOp(Op):
    values: list[Value]
    labels: list[str]
    type: Type = Nil()

    __operands__ = (("values", Type), ("labels", String))


@llvm.op("fcmp")
@dataclass(eq=False, kw_only=True)
class FcmpOp(Op):
    pred: str
    lhs: Value
    rhs: Value
    type: Type = Nil()

    __operands__ = (("pred", String), ("lhs", Type), ("rhs", Type))


@llvm.op("zext")
@dataclass(eq=False, kw_only=True)
class ZextOp(Op):
    input: Value
    type: Type = Nil()

    __operands__ = (("input", Type),)


@llvm.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: str
    args: list[Value]
    type: Type = Nil()

    __operands__ = (("callee", String), ("args", Type))
