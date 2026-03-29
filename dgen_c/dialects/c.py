"""C dialect for dgen — types and operations for C language constructs."""

from __future__ import annotations

from dataclasses import dataclass

import dgen
from dgen import Block, Dialect, Op, Type, Value

c = Dialect("c")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@c.type("CInt")
@dataclass(frozen=True, eq=False)
class CInt(Type):
    """C integer type. bits=width, signed=1 for signed, 0 for unsigned."""

    bits: Value[dgen.TypeType]
    signed: Value[dgen.TypeType]


@c.type("CFloat")
@dataclass(frozen=True, eq=False)
class CFloat(Type):
    """C float type. kind=0 float, 1 double, 2 long double."""

    kind: Value[dgen.TypeType]


@c.type("CVoid")
@dataclass(frozen=True, eq=False)
class CVoid(Type):
    """C void type."""

    pass


@c.type("CPtr")
@dataclass(frozen=True, eq=False)
class CPtr(Type):
    """Pointer to a C type."""

    pointee: Value[dgen.TypeType]


@c.type("CArray")
@dataclass(frozen=True, eq=False)
class CArray(Type):
    """Fixed-size C array."""

    element: Value[dgen.TypeType]
    count: Value[dgen.TypeType]


@c.type("CStruct")
@dataclass(frozen=True, eq=False)
class CStruct(Type):
    """C struct type identified by name, with field names and types."""

    tag_name: Value[dgen.TypeType]
    field_names: Value[dgen.TypeType]
    field_types: Value[dgen.TypeType]


@c.type("CUnion")
@dataclass(frozen=True, eq=False)
class CUnion(Type):
    """C union type."""

    tag_name: Value[dgen.TypeType]
    field_names: Value[dgen.TypeType]
    field_types: Value[dgen.TypeType]


@c.type("CFuncType")
@dataclass(frozen=True, eq=False)
class CFuncType(Type):
    """C function type."""

    return_type: Value[dgen.TypeType]
    param_types: Value[dgen.TypeType]
    variadic: Value[dgen.TypeType]


# Convenience constructors for common C types
def c_int(bits: int = 32, signed: bool = True) -> CInt:
    """Create a CInt type constant."""
    from dgen.dialects.index import Index

    return CInt(
        bits=Index().constant(bits),
        signed=Index().constant(1 if signed else 0),
    )


def c_float() -> CFloat:
    from dgen.dialects.index import Index

    return CFloat(kind=Index().constant(0))


def c_double() -> CFloat:
    from dgen.dialects.index import Index

    return CFloat(kind=Index().constant(1))


def c_void() -> CVoid:
    return CVoid()


def c_ptr(pointee: Type) -> CPtr:
    return CPtr(pointee=pointee)


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------


@c.op("add")
@dataclass(eq=False, kw_only=True)
class AddOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("sub")
@dataclass(eq=False, kw_only=True)
class SubOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("mul")
@dataclass(eq=False, kw_only=True)
class MulOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("div")
@dataclass(eq=False, kw_only=True)
class DivOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("mod")
@dataclass(eq=False, kw_only=True)
class ModOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("neg")
@dataclass(eq=False, kw_only=True)
class NegOp(Op):
    operand: Value
    type: Type


@c.op("bitnot")
@dataclass(eq=False, kw_only=True)
class BitnotOp(Op):
    operand: Value
    type: Type


@c.op("lognot")
@dataclass(eq=False, kw_only=True)
class LognotOp(Op):
    operand: Value
    type: Type


@c.op("bitand")
@dataclass(eq=False, kw_only=True)
class BitandOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("bitor")
@dataclass(eq=False, kw_only=True)
class BitorOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("bitxor")
@dataclass(eq=False, kw_only=True)
class BitxorOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("shl")
@dataclass(eq=False, kw_only=True)
class ShlOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("shr")
@dataclass(eq=False, kw_only=True)
class ShrOp(Op):
    lhs: Value
    rhs: Value
    type: Type


# ---------------------------------------------------------------------------
# Comparison ops
# ---------------------------------------------------------------------------

_BOOL = None  # lazily initialized


def _bool_type() -> CInt:
    global _BOOL
    if _BOOL is None:
        _BOOL = c_int(32, signed=True)
    return _BOOL


@c.op("eq")
@dataclass(eq=False, kw_only=True)
class EqOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("ne")
@dataclass(eq=False, kw_only=True)
class NeOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("lt")
@dataclass(eq=False, kw_only=True)
class LtOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("le")
@dataclass(eq=False, kw_only=True)
class LeOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("gt")
@dataclass(eq=False, kw_only=True)
class GtOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("ge")
@dataclass(eq=False, kw_only=True)
class GeOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("logand")
@dataclass(eq=False, kw_only=True)
class LogandOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("logor")
@dataclass(eq=False, kw_only=True)
class LogorOp(Op):
    lhs: Value
    rhs: Value
    type: Type


# ---------------------------------------------------------------------------
# Memory ops
# ---------------------------------------------------------------------------


@c.op("alloca")
@dataclass(eq=False, kw_only=True)
class AllocaOp(Op):
    element_type: Value[dgen.TypeType]
    type: Type


@c.op("load")
@dataclass(eq=False, kw_only=True)
class LoadOp(Op):
    ptr: Value
    type: Type


@c.op("store")
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    value: Value
    ptr: Value
    type: Type = CVoid()


@c.op("gep")
@dataclass(eq=False, kw_only=True)
class GepOp(Op):
    field_index: Value[dgen.TypeType]
    base: Value
    type: Type


@c.op("array_index")
@dataclass(eq=False, kw_only=True)
class ArrayIndexOp(Op):
    base: Value
    index: Value
    type: Type


@c.op("address_of")
@dataclass(eq=False, kw_only=True)
class AddressOfOp(Op):
    operand: Value
    type: Type


@c.op("deref")
@dataclass(eq=False, kw_only=True)
class DerefOp(Op):
    ptr: Value
    type: Type


# ---------------------------------------------------------------------------
# Cast ops
# ---------------------------------------------------------------------------


@c.op("cast")
@dataclass(eq=False, kw_only=True)
class CastOp(Op):
    target_type: Value[dgen.TypeType]
    operand: Value
    type: Type


@c.op("bitcast")
@dataclass(eq=False, kw_only=True)
class BitcastOp(Op):
    target_type: Value[dgen.TypeType]
    operand: Value
    type: Type


# ---------------------------------------------------------------------------
# Function ops
# ---------------------------------------------------------------------------


@c.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: Value[dgen.TypeType]
    arguments: Value
    type: Type


@c.op("call_indirect")
@dataclass(eq=False, kw_only=True)
class CallIndirectOp(Op):
    callee: Value
    arguments: Value
    type: Type


@c.op("return_void")
@dataclass(eq=False, kw_only=True)
class ReturnVoidOp(Op):
    type: Type = CVoid()


@c.op("return_value")
@dataclass(eq=False, kw_only=True)
class ReturnValueOp(Op):
    value: Value
    type: Type = CVoid()


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


@c.op("if")
@dataclass(eq=False, kw_only=True)
class IfOp(Op):
    condition: Value
    type: Type
    then_body: Block
    else_body: Block


@c.op("while")
@dataclass(eq=False, kw_only=True)
class WhileOp(Op):
    condition_init: Value
    type: Type = CVoid()
    condition: Block = None  # type: ignore[assignment]
    body: Block = None  # type: ignore[assignment]


@c.op("for_loop")
@dataclass(eq=False, kw_only=True)
class ForLoopOp(Op):
    init: Value
    type: Type = CVoid()
    condition: Block = None  # type: ignore[assignment]
    update: Block = None  # type: ignore[assignment]
    body: Block = None  # type: ignore[assignment]


@c.op("do_while")
@dataclass(eq=False, kw_only=True)
class DoWhileOp(Op):
    init: Value
    type: Type = CVoid()
    body: Block = None  # type: ignore[assignment]
    condition: Block = None  # type: ignore[assignment]


@c.op("switch")
@dataclass(eq=False, kw_only=True)
class SwitchOp(Op):
    case_values: Value[dgen.TypeType]
    selector: Value
    type: Type = CVoid()
    default_body: Block = None  # type: ignore[assignment]


@c.op("break")
@dataclass(eq=False, kw_only=True)
class BreakOp(Op):
    type: Type = CVoid()


@c.op("continue")
@dataclass(eq=False, kw_only=True)
class ContinueOp(Op):
    type: Type = CVoid()


@c.op("goto")
@dataclass(eq=False, kw_only=True)
class GotoOp(Op):
    label: Value[dgen.TypeType]
    type: Type = CVoid()


@c.op("label")
@dataclass(eq=False, kw_only=True)
class LabelOp(Op):
    label_name: Value[dgen.TypeType]
    type: Type = CVoid()


# ---------------------------------------------------------------------------
# Struct/union member access
# ---------------------------------------------------------------------------


@c.op("struct_member")
@dataclass(eq=False, kw_only=True)
class StructMemberOp(Op):
    field_name: Value[dgen.TypeType]
    base: Value
    type: Type


@c.op("struct_ptr_member")
@dataclass(eq=False, kw_only=True)
class StructPtrMemberOp(Op):
    field_name: Value[dgen.TypeType]
    base: Value
    type: Type


# ---------------------------------------------------------------------------
# Misc ops
# ---------------------------------------------------------------------------


@c.op("sizeof")
@dataclass(eq=False, kw_only=True)
class SizeofOp(Op):
    target_type: Value[dgen.TypeType]
    type: Type


@c.op("comma")
@dataclass(eq=False, kw_only=True)
class CommaOp(Op):
    lhs: Value
    rhs: Value
    type: Type


@c.op("ternary")
@dataclass(eq=False, kw_only=True)
class TernaryOp(Op):
    condition: Value
    true_val: Value
    false_val: Value
    type: Type
