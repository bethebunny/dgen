"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Op, Type, Value
from dgen.dialect import Dialect
from dgen.dialects.builtin import IndexType, Nil, String
from dgen.layout import Float64, Int, Pointer, Void

# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True)
class PtrType(Type):
    __layout__ = Pointer(Void())

    @property
    def asm(self) -> str:
        return "ptr"


@dataclass(frozen=True)
class IntType(Type):
    __layout__ = Int()
    bits: int

    @property
    def asm(self) -> str:
        return f"i{self.bits}"


@dataclass(frozen=True)
class FloatType(Type):
    __layout__ = Float64()

    @property
    def asm(self) -> str:
        return "f64"


@dataclass(frozen=True)
class VoidType(Type):
    __layout__ = Void()

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
    type: Type = PtrType()

    __params__ = (("elem_count", IndexType),)


@llvm.op("gep")
@dataclass(eq=False, kw_only=True)
class GepOp(Op):
    base: Value
    index: Value
    type: Type = PtrType()

    __operands__ = (("base", Type), ("index", Type))


@llvm.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    ptr: Value
    type: Type = FloatType()

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
    type: Type = FloatType()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("fmul")
@dataclass(eq=False, kw_only=True)
class FMulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = FloatType()

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type = IntType(bits=64)

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type = IntType(bits=64)

    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("icmp")
@dataclass(eq=False, kw_only=True)
class IcmpOp(Op):
    pred: Value[String]
    lhs: Value
    rhs: Value
    type: Type = IntType(bits=1)

    __params__ = (("pred", String),)
    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("br")
@dataclass(eq=False, kw_only=True)
class BrOp(Op):
    dest: Value[String]
    type: Type = Nil()

    __params__ = (("dest", String),)


@llvm.op("cond_br")
@dataclass(eq=False, kw_only=True)
class CondBrOp(Op):
    cond: Value
    true_dest: Value[String]
    false_dest: Value[String]
    type: Type = Nil()

    __params__ = (("true_dest", String), ("false_dest", String))
    __operands__ = (("cond", Type),)


@llvm.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: Value[String]
    type: Type = Nil()

    __params__ = (("label_name", String),)


@llvm.op("phi")
@dataclass(eq=False, kw_only=True)
class PhiOp(Op):
    values: list[Value]
    labels: list[Value[String]]
    type: Type = Nil()

    __params__ = (("labels", String),)
    __operands__ = (("values", Type),)


@llvm.op("fcmp")
@dataclass(eq=False, kw_only=True)
class FcmpOp(Op):
    pred: Value[String]
    lhs: Value
    rhs: Value
    type: Type = IntType(bits=1)

    __params__ = (("pred", String),)
    __operands__ = (("lhs", Type), ("rhs", Type))


@llvm.op("zext")
@dataclass(eq=False, kw_only=True)
class ZextOp(Op):
    input: Value
    type: Type = IntType(bits=64)

    __operands__ = (("input", Type),)


@llvm.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: Value[String]
    args: list[Value]
    type: Type = Nil()

    __params__ = (("callee", String),)
    __operands__ = (("args", Type),)
