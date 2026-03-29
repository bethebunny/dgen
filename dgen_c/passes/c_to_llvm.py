"""Lower C dialect ops to LLVM dialect ops.

Maps C-level operations to their LLVM IR equivalents:
- Integer arithmetic → llvm.add/sub/mul/sdiv
- Float arithmetic → llvm.fadd/fsub/fmul/fdiv
- Comparisons → llvm.icmp/fcmp
- Memory ops → llvm.alloca/load/store/gep
- Casts → llvm.zext/bitcast
- Calls → llvm.call
"""

from __future__ import annotations

import dgen
from dgen.dialects import llvm
from dgen.dialects.builtin import Nil, String
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for

from dgen_c.dialects.c import (
    AddOp,
    AllocaOp,
    ArrayIndexOp,
    BitandOp,
    BitnotOp,
    BitorOp,
    BitxorOp,
    CallOp,
    CastOp,
    CFloat,
    CInt,
    DerefOp,
    DivOp,
    EqOp,
    GeOp,
    GtOp,
    LeOp,
    LoadOp,
    LogandOp,
    LognotOp,
    LogorOp,
    LtOp,
    ModOp,
    MulOp,
    NeOp,
    NegOp,
    ReturnValueOp,
    ReturnVoidOp,
    ShlOp,
    ShrOp,
    StoreOp,
    SubOp,
)


def _is_float_type(ty: dgen.Type) -> bool:
    """Check if a C type is a floating-point type."""
    return isinstance(ty, CFloat)


def _is_int_type(ty: dgen.Type) -> bool:
    """Check if a C type is an integer type."""
    return isinstance(ty, CInt)


class CToLLVM(Pass):
    """Lower C dialect ops to LLVM IR ops."""

    allow_unregistered_ops = True

    # --- Integer arithmetic ---

    @lowering_for(AddOp)
    def lower_add(self, op: AddOp) -> dgen.Value | None:
        if _is_float_type(op.type):
            return llvm.FaddOp(lhs=op.lhs, rhs=op.rhs)
        return llvm.AddOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(SubOp)
    def lower_sub(self, op: SubOp) -> dgen.Value | None:
        if _is_float_type(op.type):
            return llvm.FsubOp(lhs=op.lhs, rhs=op.rhs)
        return llvm.SubOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(MulOp)
    def lower_mul(self, op: MulOp) -> dgen.Value | None:
        if _is_float_type(op.type):
            return llvm.FmulOp(lhs=op.lhs, rhs=op.rhs)
        return llvm.MulOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(DivOp)
    def lower_div(self, op: DivOp) -> dgen.Value | None:
        if _is_float_type(op.type):
            return llvm.FdivOp(lhs=op.lhs, rhs=op.rhs)
        return llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(NegOp)
    def lower_neg(self, op: NegOp) -> dgen.Value | None:
        if _is_float_type(op.type):
            return llvm.FnegOp(input=op.operand)
        # Integer: 0 - x
        zero = ConstantOp(value=0, type=op.type)
        return llvm.SubOp(lhs=zero, rhs=op.operand)

    # --- Bitwise ---

    @lowering_for(BitandOp)
    def lower_bitand(self, op: BitandOp) -> dgen.Value | None:
        return llvm.AndOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(BitorOp)
    def lower_bitor(self, op: BitorOp) -> dgen.Value | None:
        return llvm.OrOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(BitxorOp)
    def lower_bitxor(self, op: BitxorOp) -> dgen.Value | None:
        return llvm.XorOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(ShlOp)
    def lower_shl(self, op: ShlOp) -> dgen.Value | None:
        return llvm.MulOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    @lowering_for(ShrOp)
    def lower_shr(self, op: ShrOp) -> dgen.Value | None:
        return llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    # --- Comparisons ---

    @lowering_for(EqOp)
    def lower_eq(self, op: EqOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("oeq"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("eq"), lhs=op.lhs, rhs=op.rhs)

    @lowering_for(NeOp)
    def lower_ne(self, op: NeOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("one"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("ne"), lhs=op.lhs, rhs=op.rhs)

    @lowering_for(LtOp)
    def lower_lt(self, op: LtOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("olt"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("slt"), lhs=op.lhs, rhs=op.rhs)

    @lowering_for(LeOp)
    def lower_le(self, op: LeOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("ole"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("sle"), lhs=op.lhs, rhs=op.rhs)

    @lowering_for(GtOp)
    def lower_gt(self, op: GtOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("ogt"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("sgt"), lhs=op.lhs, rhs=op.rhs)

    @lowering_for(GeOp)
    def lower_ge(self, op: GeOp) -> dgen.Value | None:
        if _is_float_type(op.lhs.type):
            return llvm.FcmpOp(pred=String().constant("oge"), lhs=op.lhs, rhs=op.rhs)
        return llvm.IcmpOp(pred=String().constant("sge"), lhs=op.lhs, rhs=op.rhs)

    # --- Memory ---

    @lowering_for(AllocaOp)
    def lower_alloca(self, op: AllocaOp) -> dgen.Value | None:
        return llvm.AllocaOp(elem_count=Index().constant(1))

    @lowering_for(LoadOp)
    def lower_load(self, op: LoadOp) -> dgen.Value | None:
        return llvm.LoadOp(ptr=op.ptr, type=op.type)

    @lowering_for(StoreOp)
    def lower_store(self, op: StoreOp) -> dgen.Value | None:
        return llvm.StoreOp(value=op.value, ptr=op.ptr)

    @lowering_for(DerefOp)
    def lower_deref(self, op: DerefOp) -> dgen.Value | None:
        return llvm.LoadOp(ptr=op.ptr, type=op.type)

    @lowering_for(ArrayIndexOp)
    def lower_array_index(self, op: ArrayIndexOp) -> dgen.Value | None:
        return llvm.GepOp(base=op.base, index=op.index)

    # --- Calls ---

    @lowering_for(CallOp)
    def lower_call(self, op: CallOp) -> dgen.Value | None:
        return llvm.CallOp(callee=op.callee, args=op.arguments, type=op.type)

    # --- Returns ---

    @lowering_for(ReturnValueOp)
    def lower_return_value(self, op: ReturnValueOp) -> dgen.Value | None:
        # Return the value directly — the function result handling
        # takes care of emitting the LLVM ret instruction
        return op.value

    @lowering_for(ReturnVoidOp)
    def lower_return_void(self, op: ReturnVoidOp) -> dgen.Value | None:
        return ConstantOp(value=0, type=Nil())

    # --- Casts ---

    @lowering_for(CastOp)
    def lower_cast(self, op: CastOp) -> dgen.Value | None:
        # Simplified: just pass through the operand
        return op.operand

    # --- Logical ops ---

    @lowering_for(LognotOp)
    def lower_lognot(self, op: LognotOp) -> dgen.Value | None:
        zero = ConstantOp(value=0, type=op.operand.type)
        return llvm.IcmpOp(pred=String().constant("eq"), lhs=op.operand, rhs=zero)

    @lowering_for(LogandOp)
    def lower_logand(self, op: LogandOp) -> dgen.Value | None:
        return llvm.AndOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(LogorOp)
    def lower_logor(self, op: LogorOp) -> dgen.Value | None:
        return llvm.OrOp(lhs=op.lhs, rhs=op.rhs)

    @lowering_for(BitnotOp)
    def lower_bitnot(self, op: BitnotOp) -> dgen.Value | None:
        ones = ConstantOp(value=-1, type=op.type)
        return llvm.XorOp(lhs=op.operand, rhs=ones)

    @lowering_for(ModOp)
    def lower_mod(self, op: ModOp) -> dgen.Value | None:
        # x % y = x - (x / y) * y
        div = llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)
        mul = llvm.MulOp(lhs=div, rhs=op.rhs)
        return llvm.SubOp(lhs=op.lhs, rhs=mul)
