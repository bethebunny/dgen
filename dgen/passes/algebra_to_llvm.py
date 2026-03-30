"""Lower algebra dialect ops to LLVM dialect ops.

Type-directed dispatch: inspect the operand type, emit the corresponding LLVM
op. Stateless — no context, no shape tracking, no control flow awareness.
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, llvm
from dgen.dialects.builtin import String
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger
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
        return llvm.AddOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.NegateOp)
    def lower_negate(self, op: algebra.NegateOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FnegOp(input=op.input)
        # Integer negate: 0 - x
        zero = dgen.module.ConstantOp(value=0, type=op.type)
        return llvm.SubOp(lhs=zero, rhs=op.input)

    @lowering_for(algebra.SubtractOp)
    def lower_subtract(self, op: algebra.SubtractOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FsubOp(lhs=op.left, rhs=op.right)
        return llvm.SubOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.MultiplyOp)
    def lower_multiply(self, op: algebra.MultiplyOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FmulOp(lhs=op.left, rhs=op.right)
        return llvm.MulOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.ReciprocalOp)
    def lower_reciprocal(self, op: algebra.ReciprocalOp) -> dgen.Value:
        one = dgen.module.ConstantOp(value=1.0, type=Float64())
        return llvm.FdivOp(lhs=one, rhs=op.input)

    @lowering_for(algebra.DivideOp)
    def lower_divide(self, op: algebra.DivideOp) -> dgen.Value:
        if _is_float(op):
            return llvm.FdivOp(lhs=op.left, rhs=op.right)
        return llvm.SdivOp(lhs=op.left, rhs=op.right)

    # --- Lattice / bitwise ---

    @lowering_for(algebra.MeetOp)
    def lower_meet(self, op: algebra.MeetOp) -> dgen.Value:
        return llvm.AndOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.JoinOp)
    def lower_join(self, op: algebra.JoinOp) -> dgen.Value:
        return llvm.OrOp(lhs=op.left, rhs=op.right)

    @lowering_for(algebra.ComplementOp)
    def lower_complement(self, op: algebra.ComplementOp) -> dgen.Value:
        # NOT x = XOR x, -1 (all ones)
        all_ones = dgen.module.ConstantOp(value=-1, type=op.type)
        return llvm.XorOp(lhs=op.input, rhs=all_ones)

    @lowering_for(algebra.SymmetricDifferenceOp)
    def lower_symmetric_difference(
        self, op: algebra.SymmetricDifferenceOp
    ) -> dgen.Value:
        return llvm.XorOp(lhs=op.left, rhs=op.right)

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
        src = op.input.type
        dst = op.type

        # i1 → integer: comparison widening (zero-extend)
        if isinstance(op.input, (llvm.IcmpOp, llvm.FcmpOp)):
            return llvm.ZextOp(input=op.input)

        # int → ptr: null pointer constant
        if isinstance(dst, Reference) and isinstance(op.input, dgen.module.ConstantOp):
            return dgen.module.ConstantOp(value=0, type=llvm.Ptr())

        # ptr → ptr: passthrough (different pointee types)
        if isinstance(src, Reference) and isinstance(dst, Reference):
            return op.input

        # int → float / float → int: passthrough at LLVM level
        # (both map to 64-bit register-passable values)
        if _is_float_type(src) != _is_float_type(dst):
            return op.input

        # int → int (different widths/signedness): passthrough
        # (all integers use the same i64 layout currently)
        if _is_int_type(src) and _is_int_type(dst):
            return op.input

        # Default: passthrough
        return op.input


def _is_float_type(ty: dgen.Type) -> bool:
    return isinstance(ty, Float64)


def _is_int_type(ty: dgen.Type) -> bool:
    return isinstance(ty, (SignedInteger, UnsignedInteger))
