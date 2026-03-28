"""Lower algebra dialect ops to LLVM dialect ops.

Type-directed dispatch: inspect the operand type, emit the corresponding LLVM
op. Stateless — no context, no shape tracking, no control flow awareness.

Implemented:
    algebra.add          → llvm.fadd / llvm.add (type-directed)
    algebra.multiply     → llvm.fmul / llvm.mul (type-directed)
    algebra.subtract     → llvm.sub
    algebra.equal        → llvm.icmp("eq") + llvm.zext
    algebra.not_equal    → llvm.fcmp("one") / llvm.icmp("ne") + llvm.zext
    algebra.less_than    → llvm.icmp("slt")
    algebra.cast         → identity (type-system cast, no runtime conversion)

"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, llvm
from dgen.dialects.builtin import String
from dgen.dialects.number import Float64
from dgen.passes.pass_ import Pass, lowering_for


class AlgebraToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(algebra.AddOp)
    def lower_add(self, op: algebra.AddOp) -> dgen.Value:
        llvm_op = llvm.FaddOp if isinstance(op.type, Float64) else llvm.AddOp
        return llvm_op(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.MultiplyOp)
    def lower_multiply(self, op: algebra.MultiplyOp) -> dgen.Value:
        llvm_op = llvm.FmulOp if isinstance(op.type, Float64) else llvm.MulOp
        return llvm_op(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.SubtractOp)
    def lower_subtract(self, op: algebra.SubtractOp) -> dgen.Value:
        return llvm.SubOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.EqualOp)
    def lower_equal(self, op: algebra.EqualOp) -> dgen.Value:
        icmp = llvm.IcmpOp(pred=String().constant("eq"), lhs=op.left, rhs=op.right)
        return llvm.ZextOp(input=icmp)

    @lowering_for(algebra.LessThanOp)
    def lower_less_than(self, op: algebra.LessThanOp) -> dgen.Value:
        return llvm.IcmpOp(pred=String().constant("slt"), lhs=op.left, rhs=op.right)

    @lowering_for(algebra.NotEqualOp)
    def lower_not_equal(self, op: algebra.NotEqualOp) -> dgen.Value:
        if isinstance(op.left.type, Float64):
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=op.left, rhs=op.right)
        else:
            cmp = llvm.IcmpOp(pred=String().constant("ne"), lhs=op.left, rhs=op.right)
        return llvm.ZextOp(input=cmp)

    @lowering_for(algebra.CastOp)
    def lower_cast(self, op: algebra.CastOp) -> dgen.Value:
        return op.input
