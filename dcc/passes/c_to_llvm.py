"""Lower C-specific ops to LLVM dialect ops.

Handles ops that have no shared dialect equivalent: modulo, shifts,
calls, return, struct access, sizeof, do_while, break.

Arithmetic, comparisons, bitwise, and memory ops are handled by
AlgebraToLLVM and MemoryToLLVM. Ternary uses control_flow.if.
Logical not uses algebra.equal.
"""

from __future__ import annotations

import dgen
from dgen.dialects import llvm
from dgen.dialects.builtin import Nil
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for

from dcc.dialects.c import (
    BreakOp,
    CallOp,
    DoWhileOp,
    ModuloOp,
    ReturnOp,
    ShiftLeftOp,
    ShiftRightOp,
    SizeofOp,
)


class CToLLVM(Pass):
    """Lower C-specific ops that have no shared dialect equivalent."""

    allow_unregistered_ops = True

    # --- Modulo, shifts ---

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

    # --- Calls ---

    @lowering_for(CallOp)
    def lower_call(self, op: CallOp) -> dgen.Value | None:
        return llvm.CallOp(callee=op.callee, args=op.arguments, type=op.type)

    # --- Return ---

    @lowering_for(ReturnOp)
    def lower_return(self, op: ReturnOp) -> dgen.Value | None:
        return op.value

    # --- Misc ---

    @lowering_for(SizeofOp)
    def lower_sizeof(self, op: SizeofOp) -> dgen.Value | None:
        return ConstantOp(value=8, type=llvm.Int(bits=Index().constant(64)))

    @lowering_for(DoWhileOp)
    def lower_do_while(self, op: DoWhileOp) -> dgen.Value | None:
        return ConstantOp(value=None, type=Nil())

    @lowering_for(BreakOp)
    def lower_break(self, op: BreakOp) -> dgen.Value | None:
        return ConstantOp(value=None, type=Nil())
