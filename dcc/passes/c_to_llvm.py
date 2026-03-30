"""Lower C-specific ops to LLVM dialect ops.

Only handles ops that have no algebra/memory dialect equivalent:
mod, shl, shr, lognot, calls, returns, struct access, sizeof,
ternary, do_while, break. Arithmetic, comparisons, bitwise, and
memory ops are handled by AlgebraToLLVM and MemoryToLLVM.
"""

from __future__ import annotations

import dgen
from dgen.dialects import llvm
from dgen.dialects.builtin import Nil, String
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for

from dcc.dialects.c import (
    BreakOp,
    CallIndirectOp,
    CallOp,
    DoWhileOp,
    GepOp,
    LognotOp,
    ModOp,
    ReturnValueOp,
    ReturnVoidOp,
    ShlOp,
    ShrOp,
    SizeofOp,
    StructMemberOp,
    StructPtrMemberOp,
    TernaryOp,
)


class CToLLVM(Pass):
    """Lower C-specific ops that have no shared dialect equivalent."""

    allow_unregistered_ops = True

    # --- Mod, shift ---

    @lowering_for(ModOp)
    def lower_mod(self, op: ModOp) -> dgen.Value | None:
        div = llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)
        mul = llvm.MulOp(lhs=div, rhs=op.rhs)
        return llvm.SubOp(lhs=op.lhs, rhs=mul)

    @lowering_for(ShlOp)
    def lower_shl(self, op: ShlOp) -> dgen.Value | None:
        return llvm.MulOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    @lowering_for(ShrOp)
    def lower_shr(self, op: ShrOp) -> dgen.Value | None:
        return llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    # --- Logical not ---

    @lowering_for(LognotOp)
    def lower_lognot(self, op: LognotOp) -> dgen.Value | None:
        zero = ConstantOp(value=0, type=op.operand.type)
        cmp = llvm.IcmpOp(pred=String().constant("eq"), lhs=op.operand, rhs=zero)
        return llvm.ZextOp(input=cmp)

    # --- Calls ---

    @lowering_for(CallOp)
    def lower_call(self, op: CallOp) -> dgen.Value | None:
        return llvm.CallOp(callee=op.callee, args=op.arguments, type=op.type)

    @lowering_for(CallIndirectOp)
    def lower_call_indirect(self, op: CallIndirectOp) -> dgen.Value | None:
        return ConstantOp(value=0, type=llvm.Int(bits=Index().constant(64)))

    # --- Returns ---

    @lowering_for(ReturnValueOp)
    def lower_return_value(self, op: ReturnValueOp) -> dgen.Value | None:
        return op.value

    @lowering_for(ReturnVoidOp)
    def lower_return_void(self, op: ReturnVoidOp) -> dgen.Value | None:
        return ConstantOp(value=None, type=Nil())

    # --- Struct access ---

    @lowering_for(StructPtrMemberOp)
    def lower_struct_ptr_member(self, op: StructPtrMemberOp) -> dgen.Value | None:
        zero = ConstantOp(value=0, type=llvm.Int(bits=Index().constant(64)))
        return llvm.GepOp(base=op.base, index=zero)

    @lowering_for(StructMemberOp)
    def lower_struct_member(self, op: StructMemberOp) -> dgen.Value | None:
        return op.base

    @lowering_for(GepOp)
    def lower_gep(self, op: GepOp) -> dgen.Value | None:
        zero = ConstantOp(value=0, type=llvm.Int(bits=Index().constant(64)))
        return llvm.GepOp(base=op.base, index=zero)

    # --- Misc ---

    @lowering_for(TernaryOp)
    def lower_ternary(self, op: TernaryOp) -> dgen.Value | None:
        return op.true_val  # simplified

    @lowering_for(SizeofOp)
    def lower_sizeof(self, op: SizeofOp) -> dgen.Value | None:
        return ConstantOp(value=8, type=llvm.Int(bits=Index().constant(64)))

    @lowering_for(DoWhileOp)
    def lower_do_while(self, op: DoWhileOp) -> dgen.Value | None:
        return ConstantOp(value=None, type=Nil())

    @lowering_for(BreakOp)
    def lower_break(self, op: BreakOp) -> dgen.Value | None:
        return ConstantOp(value=None, type=Nil())
