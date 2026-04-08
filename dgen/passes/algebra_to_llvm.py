"""Lower algebra dialect ops to LLVM dialect ops.

Type-directed dispatch: inspect the operand type, emit the corresponding LLVM
op. Stateless — no context, no shape tracking, no control flow awareness.
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, llvm
from dgen.dialects.builtin import String
from dgen.dialects.number import Float64
from dgen.passes.pass_ import Pass, lowering_for


def _is_float(op: dgen.Op) -> bool:
    return isinstance(op.type, Float64)


def _compare(op: dgen.Op, float_pred: str, int_pred: str) -> dgen.Value:
    """Lower a comparison op to fcmp/icmp (returns i1)."""
    if isinstance(op.left.type, Float64):
        return llvm.FcmpOp(
            pred=String().constant(float_pred), lhs=op.left, rhs=op.right
        )
    return llvm.IcmpOp(pred=String().constant(int_pred), lhs=op.left, rhs=op.right)


class AlgebraToLLVM(Pass):
    allow_unregistered_ops = True

    # --- Arithmetic ---

    @lowering_for(algebra.AddOp)
    def lower_add(self, op: algebra.AddOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FaddOp(lhs=op.left, rhs=op.right)
        return llvm.AddOp(lhs=op.left, rhs=op.right, type=op.type)

    @lowering_for(algebra.NegateOp)
    def lower_negate(self, op: algebra.NegateOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FnegOp(input=op.input)
        # Integer negate: 0 - x
        return llvm.SubOp(lhs=op.type.constant(0), rhs=op.input, type=op.type)

    @lowering_for(algebra.SubtractOp)
    def lower_subtract(self, op: algebra.SubtractOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FsubOp(lhs=op.left, rhs=op.right)
        return llvm.SubOp(lhs=op.left, rhs=op.right, type=op.type)

    @lowering_for(algebra.MultiplyOp)
    def lower_multiply(self, op: algebra.MultiplyOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FmulOp(lhs=op.left, rhs=op.right)
        return llvm.MulOp(lhs=op.left, rhs=op.right, type=op.type)

    @lowering_for(algebra.ReciprocalOp)
    def lower_reciprocal(self, op: algebra.ReciprocalOp) -> dgen.Value:
        return llvm.FdivOp(lhs=Float64().constant(1.0), rhs=op.input)

    @lowering_for(algebra.DivideOp)
    def lower_divide(self, op: algebra.DivideOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FdivOp(lhs=op.left, rhs=op.right)
        return llvm.SdivOp(lhs=op.left, rhs=op.right, type=op.type)

    # --- Lattice / bitwise ---

    @lowering_for(algebra.MeetOp)
    def lower_meet(self, op: algebra.MeetOp) -> dgen.Value:
        return llvm.AndOp(lhs=op.left, rhs=op.right, type=op.type)

    @lowering_for(algebra.JoinOp)
    def lower_join(self, op: algebra.JoinOp) -> dgen.Value:
        return llvm.OrOp(lhs=op.left, rhs=op.right, type=op.type)

    @lowering_for(algebra.ComplementOp)
    def lower_complement(self, op: algebra.ComplementOp) -> dgen.Value:
        # NOT x = XOR x, -1 (all ones)
        return llvm.XorOp(lhs=op.input, rhs=op.type.constant(-1), type=op.type)

    @lowering_for(algebra.SymmetricDifferenceOp)
    def lower_symmetric_difference(
        self, op: algebra.SymmetricDifferenceOp
    ) -> dgen.Value:
        return llvm.XorOp(lhs=op.left, rhs=op.right, type=op.type)

    # --- Comparison ---

    @lowering_for(algebra.EqualOp)
    def lower_equal(self, op: algebra.EqualOp) -> dgen.Value:
        return _compare(op, "oeq", "eq")

    @lowering_for(algebra.NotEqualOp)
    def lower_not_equal(self, op: algebra.NotEqualOp) -> dgen.Value:
        return _compare(op, "one", "ne")

    @lowering_for(algebra.LessThanOp)
    def lower_less_than(self, op: algebra.LessThanOp) -> dgen.Value:
        return _compare(op, "olt", "slt")

    @lowering_for(algebra.LessEqualOp)
    def lower_less_equal(self, op: algebra.LessEqualOp) -> dgen.Value:
        return _compare(op, "ole", "sle")

    @lowering_for(algebra.GreaterThanOp)
    def lower_greater_than(self, op: algebra.GreaterThanOp) -> dgen.Value:
        return _compare(op, "ogt", "sgt")

    @lowering_for(algebra.GreaterEqualOp)
    def lower_greater_equal(self, op: algebra.GreaterEqualOp) -> dgen.Value:
        return _compare(op, "oge", "sge")

    # --- Cast ---

    @lowering_for(algebra.CastOp)
    def lower_cast(self, op: algebra.CastOp) -> dgen.Value:
        # i1 → iN: comparison widening
        if isinstance(op.input, (llvm.IcmpOp, llvm.FcmpOp)):
            return llvm.ZextOp(input=op.input, type=op.type)
        from dgen.dialects.builtin import Array, Pointer
        from dgen.dialects.memory import Reference

        ptr_types: tuple[type, ...] = (Reference, llvm.Ptr, Pointer, Array)
        dst_is_ptr = isinstance(op.type, ptr_types)
        src_is_ptr = isinstance(op.input.type, ptr_types)
        if dst_is_ptr and not src_is_ptr:
            return llvm.InttoptrOp(input=op.input)
        if src_is_ptr and not dst_is_ptr:
            return llvm.PtrtointOp(input=op.input)
        return op.input
