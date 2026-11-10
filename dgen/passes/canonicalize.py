"""Fold constants and simplify algebraic identities in the algebra dialect.

Runs to a fixed point, so a rewrite that exposes another one is applied too,
eg. ``add(0, x)`` turns the constant to the right, then ``add(x, 0)`` folds.

Integer folds follow LLVM's 64-bit two's complement and truncating division.
A fold that would divide by zero, or identities that don't hold for floats
(``x + 0``, ``x * 0``, ``x - x``), are left alone.
"""

from __future__ import annotations

from collections.abc import Callable
from operator import add, eq, ge, gt, le, lt, mul, ne, sub

import dgen
from dgen.dialects import algebra
from dgen.dialects.number import Float64
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, rewrite_for
from dgen.type import Constant

type Number = int | float
type Binary = (
    algebra.AddOp
    | algebra.SubtractOp
    | algebra.MultiplyOp
    | algebra.DivideOp
    | algebra.EqualOp
    | algebra.NotEqualOp
    | algebra.LessThanOp
    | algebra.LessEqualOp
    | algebra.GreaterThanOp
    | algebra.GreaterEqualOp
)


def _value(v: dgen.Value) -> Number | None:
    """The number v holds, if it is a constant."""
    if not isinstance(v, Constant):
        return None
    value = v.__constant__.to_json()
    return value if isinstance(value, (int, float)) else None


def _wrapped(value: Number) -> Number:
    """value as a 64-bit two's complement integer, or unchanged if a float."""
    return (value + 2**63) % 2**64 - 2**63 if isinstance(value, int) else value


def _quotient(left: Number, right: Number) -> Number:
    if isinstance(left, float) or isinstance(right, float):
        return left / right
    return abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1)


def _is_float(op: dgen.Op) -> bool:
    return isinstance(op.type, Float64)


class AlgebraCanonicalize(Pass):
    allow_unregistered_ops = True
    fixed_point = True
    emits = (ConstantOp, algebra.AddOp, algebra.MultiplyOp)

    def _folded(
        self, op: Binary, fn: Callable[[Number, Number], Number | bool]
    ) -> dgen.Value | None:
        left, right = _value(op.left), _value(op.right)
        if left is None or right is None:
            return None
        result = fn(left, right)
        return ConstantOp(
            value=result if isinstance(result, bool) else _wrapped(result),
            type=op.type,
        )

    def _commuted(self, op: algebra.AddOp | algebra.MultiplyOp) -> dgen.Value | None:
        """op with its constant operand on the right, so identities see it there."""
        if _value(op.left) is None or _value(op.right) is not None:
            return None
        return type(op)(left=op.right, right=op.left, type=op.type)

    @rewrite_for(algebra.AddOp)
    def fold_add(self, op: algebra.AddOp) -> dgen.Value | None:
        if (folded := self._folded(op, add)) is not None:
            return folded
        # x + 0.0 is not x when x is -0.0.
        if _value(op.right) == 0 and not _is_float(op):
            return op.left
        return self._commuted(op)

    @rewrite_for(algebra.SubtractOp)
    def fold_subtract(self, op: algebra.SubtractOp) -> dgen.Value | None:
        if (folded := self._folded(op, sub)) is not None:
            return folded
        if _value(op.right) == 0:
            return op.left
        if op.left is op.right and not _is_float(op):
            return ConstantOp(value=0, type=op.type)
        return None

    @rewrite_for(algebra.MultiplyOp)
    def fold_multiply(self, op: algebra.MultiplyOp) -> dgen.Value | None:
        if (folded := self._folded(op, mul)) is not None:
            return folded
        match _value(op.right):
            case 1:
                return op.left
            case 0 if not _is_float(op):
                return ConstantOp(value=0, type=op.type)
        return self._commuted(op)

    @rewrite_for(algebra.DivideOp)
    def fold_divide(self, op: algebra.DivideOp) -> dgen.Value | None:
        if _value(op.right) == 1:
            return op.left
        if _value(op.right) == 0:
            return None
        return self._folded(op, _quotient)

    @rewrite_for(algebra.NegateOp)
    def fold_negate(self, op: algebra.NegateOp) -> dgen.Value | None:
        if isinstance(op.input, algebra.NegateOp):
            return op.input.input
        if (value := _value(op.input)) is None:
            return None
        return ConstantOp(value=_wrapped(-value), type=op.type)

    @rewrite_for(algebra.EqualOp)
    def fold_equal(self, op: algebra.EqualOp) -> dgen.Value | None:
        return self._folded(op, eq)

    @rewrite_for(algebra.NotEqualOp)
    def fold_not_equal(self, op: algebra.NotEqualOp) -> dgen.Value | None:
        return self._folded(op, ne)

    @rewrite_for(algebra.LessThanOp)
    def fold_less_than(self, op: algebra.LessThanOp) -> dgen.Value | None:
        return self._folded(op, lt)

    @rewrite_for(algebra.LessEqualOp)
    def fold_less_equal(self, op: algebra.LessEqualOp) -> dgen.Value | None:
        return self._folded(op, le)

    @rewrite_for(algebra.GreaterThanOp)
    def fold_greater_than(self, op: algebra.GreaterThanOp) -> dgen.Value | None:
        return self._folded(op, gt)

    @rewrite_for(algebra.GreaterEqualOp)
    def fold_greater_equal(self, op: algebra.GreaterEqualOp) -> dgen.Value | None:
        return self._folded(op, ge)
