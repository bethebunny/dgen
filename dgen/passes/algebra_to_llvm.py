"""Lower algebra dialect ops to LLVM dialect ops.

Type-directed dispatch: inspect the operand type, emit the corresponding LLVM
op. Stateless — no context, no shape tracking, no control flow awareness.

    algebra.add on Float64       → llvm.fadd
    algebra.add on Index/Int → llvm.add
    algebra.multiply on Float64  → llvm.fmul
    algebra.multiply on Int  → llvm.mul
    algebra.subtract         → llvm.sub
    algebra.equal            → llvm.icmp("eq") + llvm.zext
    algebra.less_than        → llvm.icmp("slt")
"""

from __future__ import annotations

from dgen.dialects import algebra, llvm
from dgen.dialects.builtin import String
from dgen.dialects.number import Float64
from dgen.passes.pass_ import Pass, Rewriter, lowering_for


class AlgebraToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(algebra.AddOp)
    def lower_add(self, op: algebra.AddOp, rewriter: Rewriter) -> bool:
        llvm_op = llvm.FaddOp if isinstance(op.type, Float64) else llvm.AddOp
        rewriter.replace_uses(op, llvm_op(lhs=op.left, rhs=op.right))
        return True

    @lowering_for(algebra.MultiplyOp)
    def lower_multiply(self, op: algebra.MultiplyOp, rewriter: Rewriter) -> bool:
        llvm_op = llvm.FmulOp if isinstance(op.type, Float64) else llvm.MulOp
        rewriter.replace_uses(op, llvm_op(lhs=op.left, rhs=op.right))
        return True

    @lowering_for(algebra.SubtractOp)
    def lower_subtract(self, op: algebra.SubtractOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, llvm.SubOp(lhs=op.left, rhs=op.right))
        return True

    @lowering_for(algebra.EqualOp)
    def lower_equal(self, op: algebra.EqualOp, rewriter: Rewriter) -> bool:
        icmp = llvm.IcmpOp(pred=String().constant("eq"), lhs=op.left, rhs=op.right)
        rewriter.replace_uses(op, llvm.ZextOp(input=icmp))
        return True

    @lowering_for(algebra.LessThanOp)
    def lower_less_than(self, op: algebra.LessThanOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(
            op, llvm.IcmpOp(pred=String().constant("slt"), lhs=op.left, rhs=op.right)
        )
        return True
