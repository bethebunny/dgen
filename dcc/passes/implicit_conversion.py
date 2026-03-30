"""Implicit conversion pass: resolve Unresolved types and insert casts.

Implements C11 §6.3 usual arithmetic conversions. Walks all ops and:
- Resolves Unresolved result types on binary/unary ops
- Inserts algebra.CastOp where operand types don't match
- Widens comparison results from i1 to i32
- Converts integer 0 to null pointer in pointer comparisons
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger
from dgen.module import ConstantOp, Module
from dgen.passes.pass_ import Pass, Rewriter

from dcc.dialects import c_int
from dcc.dialects.c import LogicalNotOp, ModuloOp, ShiftLeftOp, ShiftRightOp, Unresolved


# ---------------------------------------------------------------------------
# Type introspection helpers
# ---------------------------------------------------------------------------


def _int_bits(ty: dgen.Type) -> int:
    """Get bit width from SignedInteger or UnsignedInteger."""
    if isinstance(ty, (SignedInteger, UnsignedInteger)):
        return ty.bits.__constant__.to_json()
    return 0


def _is_integer(ty: dgen.Type) -> bool:
    return isinstance(ty, (SignedInteger, UnsignedInteger))


def _is_signed(ty: dgen.Type) -> bool:
    return isinstance(ty, SignedInteger)


def _is_float(ty: dgen.Type) -> bool:
    return isinstance(ty, Float64)


def _is_pointer(ty: dgen.Type) -> bool:
    return isinstance(ty, Reference)


def _promote_integer(ty: dgen.Type) -> dgen.Type:
    """C integer promotion: types narrower than int promote to int."""
    if _is_integer(ty) and _int_bits(ty) < 32:
        return c_int(32, signed=True)
    return ty


# ---------------------------------------------------------------------------
# Usual arithmetic conversions (C11 §6.3.1.8)
# ---------------------------------------------------------------------------


def _common_type(a: dgen.Type, b: dgen.Type) -> dgen.Type:
    """Compute the common type for a binary operation."""
    # Float wins over everything
    if _is_float(a) or _is_float(b):
        return Float64()

    # Pointer arithmetic: ptr ± int → ptr type
    if _is_pointer(a):
        return a
    if _is_pointer(b):
        return b

    # Non-numeric types: use the first operand's type
    if not _is_integer(a) or not _is_integer(b):
        return a

    # Integer promotion first
    a = _promote_integer(a)
    b = _promote_integer(b)

    a_bits = _int_bits(a)
    b_bits = _int_bits(b)
    a_signed = _is_signed(a)
    b_signed = _is_signed(b)

    # Same signedness: wider wins
    if a_signed == b_signed:
        return a if a_bits >= b_bits else b

    # Different signedness: determine which is unsigned
    unsigned, signed = (a, b) if not a_signed else (b, a)
    u_bits = _int_bits(unsigned)
    s_bits = _int_bits(signed)

    # If unsigned rank >= signed rank, use unsigned
    if u_bits >= s_bits:
        return unsigned

    # If signed can represent all unsigned values, use signed
    if s_bits > u_bits:
        return signed

    # Otherwise: unsigned version of signed type's width
    return c_int(s_bits, signed=False)


# ---------------------------------------------------------------------------
# Binary op detection
# ---------------------------------------------------------------------------

_BINARY_OPS = (
    algebra.AddOp,
    algebra.SubtractOp,
    algebra.MultiplyOp,
    algebra.DivideOp,
    algebra.MeetOp,
    algebra.JoinOp,
    algebra.SymmetricDifferenceOp,
)

_COMPARISON_OPS = (
    algebra.EqualOp,
    algebra.NotEqualOp,
    algebra.LessThanOp,
    algebra.LessEqualOp,
    algebra.GreaterThanOp,
    algebra.GreaterEqualOp,
)

_C_BINARY_OPS = (ModuloOp, ShiftLeftOp, ShiftRightOp)


def _left_operand(op: dgen.Op) -> dgen.Value | None:
    """Get the left/lhs operand of a binary op."""
    if hasattr(op, "left"):
        return op.left
    if hasattr(op, "lhs"):
        return op.lhs
    return None


def _right_operand(op: dgen.Op) -> dgen.Value | None:
    """Get the right/rhs operand of a binary op."""
    if hasattr(op, "right"):
        return op.right
    if hasattr(op, "rhs"):
        return op.rhs
    return None


def _set_left(op: dgen.Op, val: dgen.Value) -> None:
    if hasattr(op, "left"):
        op.left = val
    else:
        op.lhs = val


def _set_right(op: dgen.Op, val: dgen.Value) -> None:
    if hasattr(op, "right"):
        op.right = val
    else:
        op.rhs = val


def _is_int_zero(val: dgen.Value) -> bool:
    """Check if a value is a constant integer zero (for null pointer)."""
    if not isinstance(val, ConstantOp):
        return False
    if not _is_integer(val.type):
        return False
    mem = val.__constant__
    return mem.to_json() == 0


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class ImplicitConversion(Pass):
    """Insert implicit conversions per C11 §6.3."""

    allow_unregistered_ops = True

    def run(self, module: Module, compiler: object) -> Module:
        for func in module.functions:
            self._run_block(func.body)
        return module

    def _run_block(self, block: dgen.Block) -> None:
        rewriter = Rewriter(block)
        for op in list(block.ops):
            self._convert_op(op, rewriter)
            # Recurse into nested blocks
            for _, child_block in op.blocks:
                self._run_block(child_block)

    def _convert_op(self, op: dgen.Op, rewriter: Rewriter) -> None:
        """Apply implicit conversion rules to a single op."""
        if isinstance(op, (*_COMPARISON_OPS,)):
            self._convert_comparison(op, rewriter)
        elif isinstance(op, (*_BINARY_OPS, *_C_BINARY_OPS)):
            self._convert_binary(op, rewriter)
        elif isinstance(op, algebra.CastOp):
            pass  # explicit casts are fine
        elif isinstance(op, (algebra.NegateOp, algebra.ComplementOp)):
            self._convert_unary(op)
        elif isinstance(op, LogicalNotOp):
            self._convert_logical_not(op)
        elif isinstance(op.type, Unresolved):
            # Fallback: infer type from first operand
            for _, operand in op.operands:
                if not isinstance(operand.type, Unresolved):
                    op.type = operand.type
                    break

    def _convert_binary(self, op: dgen.Op, rewriter: Rewriter) -> None:
        """Resolve types for a binary arithmetic op."""
        left = _left_operand(op)
        right = _right_operand(op)
        if left is None or right is None:
            return

        common = _common_type(left.type, right.type)

        if left.type != common:
            cast = algebra.CastOp(input=left, type=common)
            _set_left(op, cast)
        if right.type != common:
            # For pointer arithmetic, don't cast the integer index
            if _is_pointer(common) and _is_integer(right.type):
                pass
            else:
                cast = algebra.CastOp(input=right, type=common)
                _set_right(op, cast)

        if isinstance(op.type, Unresolved):
            op.type = common

    def _convert_comparison(self, op: dgen.Op, rewriter: Rewriter) -> None:
        """Resolve types for a comparison op.

        Comparisons need operand matching AND result widening to i32.
        Also handles ptr == 0 by converting 0 to null pointer.
        """
        left = _left_operand(op)
        right = _right_operand(op)
        if left is None or right is None:
            return

        # Handle pointer == 0 / pointer != 0
        if _is_pointer(left.type) and _is_int_zero(right):
            null = ConstantOp(value=0, type=left.type)
            _set_right(op, null)
        elif _is_pointer(right.type) and _is_int_zero(left):
            null = ConstantOp(value=0, type=right.type)
            _set_left(op, null)
        else:
            # Standard arithmetic conversion on operands
            common = _common_type(left.type, right.type)
            if left.type != common:
                cast = algebra.CastOp(input=left, type=common)
                _set_left(op, cast)
            if right.type != common:
                cast = algebra.CastOp(input=right, type=common)
                _set_right(op, cast)

        # Comparison result type: C returns int, algebra returns i1.
        # The comparison op keeps its natural type (the operand type, which
        # AlgebraToLLVM will narrow to i1). We insert a CastOp to widen
        # the i1 result to i32 for C semantics.
        result_type = c_int(32)
        if isinstance(op.type, Unresolved):
            # Set to operand type so AlgebraToLLVM sees matching types
            op.type = left.type
        cast = algebra.CastOp(input=op, type=result_type)
        rewriter.replace_uses(op, cast)
        # Fix the cast's input back to the original op (replace_uses
        # changed it to point to itself)
        cast.input = op

    def _convert_unary(self, op: dgen.Op) -> None:
        """Resolve Unresolved type for unary ops."""
        if isinstance(op.type, Unresolved):
            op.type = op.input.type

    def _convert_logical_not(self, op: LogicalNotOp) -> None:
        """LogicalNot produces int (C semantics)."""
        if isinstance(op.type, Unresolved):
            op.type = c_int(32)
