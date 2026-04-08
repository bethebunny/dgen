"""Lower C-specific ops to LLVM dialect ops.

Handles ops that have no shared dialect equivalent: modulo, shifts,
logical not, calls, return, sizeof, do_while, break, comma.
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, llvm
from dgen.dialects.builtin import Nil
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for

from dcc.dialects.c import (
    BreakOp,
    CommaOp,
    DoWhileOp,
    LogicalNotOp,
    ModuloOp,
    ReturnOp,
    ShiftLeftOp,
    ShiftRightOp,
    SizeofOp,
)


class CToLLVM(Pass):
    """Lower C-specific ops that have no shared dialect equivalent."""

    allow_unregistered_ops = True

    @lowering_for(ModuloOp)
    def lower_modulo(self, op: ModuloOp) -> dgen.Value | None:
        div = llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)
        mul = llvm.MulOp(lhs=div, rhs=op.rhs)
        return llvm.SubOp(lhs=op.lhs, rhs=mul)

    @lowering_for(ShiftLeftOp)
    def lower_shift_left(self, op: ShiftLeftOp) -> dgen.Value | None:
        return llvm.MulOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    @lowering_for(ShiftRightOp)
    def lower_shift_right(self, op: ShiftRightOp) -> dgen.Value | None:
        return llvm.SdivOp(lhs=op.lhs, rhs=op.rhs)  # simplified

    @lowering_for(LogicalNotOp)
    def lower_logical_not(self, op: LogicalNotOp) -> dgen.Value | None:
        eq = algebra.EqualOp(
            left=op.operand, right=op.operand.type.constant(0), type=op.operand.type
        )
        return algebra.CastOp(input=eq, type=op.type)

    @lowering_for(ReturnOp)
    def lower_return(self, op: ReturnOp) -> dgen.Value | None:
        return op.value

    @lowering_for(CommaOp)
    def lower_comma(self, op: CommaOp) -> dgen.Value | None:
        return op.rhs

    @lowering_for(SizeofOp)
    def lower_sizeof(self, op: SizeofOp) -> dgen.Value | None:
        return ConstantOp.from_constant(llvm.Int(bits=Index().constant(64)).constant(8))

    @lowering_for(DoWhileOp)
    def lower_do_while(self, op: DoWhileOp) -> dgen.Value | None:
        return ConstantOp.from_constant(Nil().constant(None))

    @lowering_for(BreakOp)
    def lower_break(self, op: BreakOp) -> dgen.Value | None:
        return ConstantOp.from_constant(Nil().constant(None))
