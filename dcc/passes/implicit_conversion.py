"""Implicit conversion pass: resolve Unresolved types and insert casts.

Implements C11 §6.3 usual arithmetic conversions. For each op type that
may have mismatched operands or an Unresolved result type, a @lowering_for
handler computes the common type, inserts algebra.CastOp where needed,
and sets the resolved result type.
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger
from dgen.module import ConstantOp, Module
from dgen.passes.pass_ import Pass, Rewriter, lowering_for

from dcc.dialects import c_int
from dcc.dialects.c import LogicalNotOp, ModuloOp, ShiftLeftOp, ShiftRightOp, Unresolved


# ---------------------------------------------------------------------------
# Type introspection helpers
# ---------------------------------------------------------------------------


def _int_bits(ty: dgen.Type) -> int:
    """Get bit width.  Caller must verify _is_integer(ty) first."""
    assert isinstance(ty, (SignedInteger, UnsignedInteger)), (
        f"not an integer type: {ty}"
    )
    return ty.bits.__constant__.to_json()


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
    if _int_bits(ty) < 32:
        return c_int(32, signed=True)
    return ty


def _is_int_zero(val: dgen.Value) -> bool:
    """Check if a value is a constant integer zero (for null pointer)."""
    if not isinstance(val, ConstantOp):
        return False
    if not _is_integer(val.type):
        return False
    return val.__constant__.to_json() == 0


# ---------------------------------------------------------------------------
# Usual arithmetic conversions (C11 §6.3.1.8)
# ---------------------------------------------------------------------------


def _common_type(a: dgen.Type, b: dgen.Type) -> dgen.Type:
    """Compute the common type for a binary operation."""
    if _is_float(a) or _is_float(b):
        return Float64()
    if _is_pointer(a):
        return a
    if _is_pointer(b):
        return b
    if not _is_integer(a) or not _is_integer(b):
        return a

    a = _promote_integer(a)
    b = _promote_integer(b)
    a_bits, b_bits = _int_bits(a), _int_bits(b)
    a_signed, b_signed = _is_signed(a), _is_signed(b)

    if a_signed == b_signed:
        return a if a_bits >= b_bits else b

    unsigned, signed = (a, b) if not a_signed else (b, a)
    u_bits, s_bits = _int_bits(unsigned), _int_bits(signed)

    if u_bits >= s_bits:
        return unsigned
    if s_bits > u_bits:
        return signed
    return c_int(s_bits, signed=False)


# ---------------------------------------------------------------------------
# Cast-insertion helpers (used by handlers)
# ---------------------------------------------------------------------------


def _cast_if_needed(val: dgen.Value, target: dgen.Type) -> dgen.Value:
    """Wrap val in a CastOp if its type differs from target."""
    if val.type == target:
        return val
    return algebra.CastOp(input=val, type=target)


def _convert_arith_pair(
    left: dgen.Value, right: dgen.Value
) -> tuple[dgen.Value, dgen.Value, dgen.Type]:
    """Apply usual arithmetic conversions to a pair of operands."""
    common = _common_type(left.type, right.type)
    left = _cast_if_needed(left, common)
    # For pointer arithmetic, don't cast the integer index
    if not (_is_pointer(common) and _is_integer(right.type)):
        right = _cast_if_needed(right, common)
    return left, right, common


def _convert_cmp_pair(
    left: dgen.Value, right: dgen.Value
) -> tuple[dgen.Value, dgen.Value]:
    """Match comparison operand types, handling ptr == 0."""
    if _is_pointer(left.type) and _is_int_zero(right):
        return left, ConstantOp(value=0, type=left.type)
    if _is_pointer(right.type) and _is_int_zero(left):
        return ConstantOp(value=0, type=right.type), right
    common = _common_type(left.type, right.type)
    return _cast_if_needed(left, common), _cast_if_needed(right, common)


def _widen_comparison(op: dgen.Op) -> dgen.Value:
    """Wrap a comparison result in CastOp(→ i32) for C semantics.

    Returns the CastOp.  The caller must arrange for replace_uses(op, cast)
    and then fix cast.input back to op afterwards.
    """
    if isinstance(op.type, Unresolved):
        op.type = op.left.type
    return algebra.CastOp(input=op, type=c_int(32))


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

    # --- Algebra binary arithmetic ---

    @lowering_for(algebra.AddOp)
    @lowering_for(algebra.SubtractOp)
    @lowering_for(algebra.MultiplyOp)
    @lowering_for(algebra.DivideOp)
    @lowering_for(algebra.MeetOp)
    @lowering_for(algebra.JoinOp)
    @lowering_for(algebra.SymmetricDifferenceOp)
    def convert_algebra_binop(self, op: dgen.Op) -> dgen.Value | None:
        op.left, op.right, common = _convert_arith_pair(op.left, op.right)
        if isinstance(op.type, Unresolved):
            op.type = common
        return None

    # --- C binary arithmetic ---

    @lowering_for(ModuloOp)
    @lowering_for(ShiftLeftOp)
    @lowering_for(ShiftRightOp)
    def convert_c_binop(self, op: dgen.Op) -> dgen.Value | None:
        op.lhs, op.rhs, common = _convert_arith_pair(op.lhs, op.rhs)
        if isinstance(op.type, Unresolved):
            op.type = common
        return None

    # --- Comparisons ---

    @lowering_for(algebra.EqualOp)
    @lowering_for(algebra.NotEqualOp)
    @lowering_for(algebra.LessThanOp)
    @lowering_for(algebra.LessEqualOp)
    @lowering_for(algebra.GreaterThanOp)
    @lowering_for(algebra.GreaterEqualOp)
    def convert_comparison(self, op: dgen.Op) -> dgen.Value | None:
        op.left, op.right = _convert_cmp_pair(op.left, op.right)
        return _widen_comparison(op)

    # --- Unary ---

    @lowering_for(algebra.NegateOp)
    @lowering_for(algebra.ComplementOp)
    def convert_unary(self, op: dgen.Op) -> dgen.Value | None:
        if isinstance(op.type, Unresolved):
            op.type = op.input.type
        return None

    @lowering_for(LogicalNotOp)
    def convert_logical_not(self, op: LogicalNotOp) -> dgen.Value | None:
        if isinstance(op.type, Unresolved):
            op.type = c_int(32)
        return None

    # --- Override _run_block for Unresolved fallback and self-ref fixup ---

    def _run_block(self, block: dgen.Block) -> None:
        rewriter = Rewriter(block)
        for op in list(block.ops):
            handlers = self._handlers.get(type(op), [])
            result: dgen.Value | None = None
            for handler in handlers:
                result = handler(self, op)
                if result is not None:
                    rewriter.replace_uses(op, result)
                    # replace_uses may have replaced result's reference
                    # to op with result itself (self-reference).  Fix it.
                    result.replace_operand(result, op)
                    break
            if result is None:
                # Fallback: resolve any remaining Unresolved types
                if isinstance(op.type, Unresolved):
                    for _, operand in op.operands:
                        if not isinstance(operand.type, Unresolved):
                            op.type = operand.type
                            break
                # Recurse into nested blocks
                for _, child_block in op.blocks:
                    self._run_block(child_block)
