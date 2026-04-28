"""Tests for the Linear / Affine trait pair and ``verify_linearity``.

The classification — defined in ``docs/linear_types.md`` — is read off
the value's *type* (not the value's own declared traits):

- ``Linearity.LINEAR``       — type has trait ``Linear``: consume exactly once
- ``Linearity.AFFINE``       — type has trait ``Affine``: consume at most once
- ``Linearity.UNRESTRICTED`` — neither

The verifier enforces three rules: at most one direct consumer per LA
value (Rule 1), no leak of linear values at block exit (Rule 2), and
PARTIAL ops drain in-scope linear values before they execute (Rule 3).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import pytest

from dgen import Dialect, Op, Type, Value, layout
from dgen.block import Block, BlockParameter
from dgen.dialects import builtin, error, goto
from dgen.dialects.builtin import ChainOp, Linear
from dgen.dialects.index import Index
from dgen.ir.verification import (
    DoubleConsumeError,
    LinearityError,
    LinearLeakAtPartialError,
    LinearLeakError,
    verify_linearity,
)
from dgen.type import Fields, Linearity


# -- Test-only linear marker type -------------------------------------------

_test = Dialect("_linear_test")


@_test.type("LinearMarker")
@dataclass(frozen=True, eq=False)
class LinearMarker(Type):
    """A test-only type that declares itself ``Linear``. Stands in for the
    future ``Origin`` (and any other linear resource) so the verifier can
    be tested without dragging the origin implementation in."""

    __layout__ = layout.Void()

    @property
    def traits(self) -> Iterator[Type]:
        yield Linear()
        yield from super().traits


@_test.op("introduce_linear")
@dataclass(eq=False, kw_only=True)
class IntroduceLinearOp(Op):
    """Constructs a fresh linear-typed value out of nothing — the test
    analogue of an op like ``origin(ref)``."""

    type: Type = LinearMarker()


@_test.op("consume_linear")
@dataclass(eq=False, kw_only=True)
class ConsumeLinearOp(Op):
    """Consumes a linear-typed input. Result type is ``Index`` (a normal
    value), so the op is itself ``UNRESTRICTED``."""

    input: Value
    type: Type = Index()
    __operands__: ClassVar[Fields] = (("input", Type),)


# -- Trait wiring -----------------------------------------------------------


def test_raise_handler_is_affine() -> None:
    h = error.RaiseHandler(error_type=Index())
    assert h.has_trait(builtin.Affine())


def test_label_is_not_affine() -> None:
    """Labels are reentrant continuations; the simple Affine rule does
    not fit. See docs/linear_types.md (Reentrant continuations)."""
    assert not goto.Label().has_trait(builtin.Affine())


def test_index_is_unrestricted() -> None:
    assert not Index().has_trait(builtin.Affine())
    assert not Index().has_trait(builtin.Linear())


# -- Value.linearity / is_linear / is_affine_or_linear ---------------------


def test_constant_is_unrestricted() -> None:
    c = Index().constant(7)
    assert c.linearity is Linearity.UNRESTRICTED
    assert not c.is_affine_or_linear


def test_type_value_is_unrestricted() -> None:
    """Type instances are universe-1 metadata, not resources."""
    assert Index().linearity is Linearity.UNRESTRICTED
    assert LinearMarker().linearity is Linearity.UNRESTRICTED


def test_raise_handler_value_is_affine() -> None:
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    assert h.linearity is Linearity.AFFINE
    assert h.is_affine_or_linear
    assert not h.is_linear


def test_linear_marker_value_is_linear() -> None:
    op = IntroduceLinearOp(name="o")
    assert op.linearity is Linearity.LINEAR
    assert op.is_linear
    assert op.is_affine_or_linear


# -- Verifier: Rule 1 (at most one direct consumer) -------------------------


def test_unused_affine_handler_is_ok() -> None:
    """A try whose body never raises: the handler is a body parameter,
    captured nowhere, used nowhere. Affine permits zero consumers."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    body = Block(result=Index().constant(7), parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    verify_linearity(try_op)


def test_raise_consumes_handler_once_ok() -> None:
    """A `raise` consumes its handler operand exactly once → fine."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=Index().constant(0),
        name="r",
    )
    body = Block(result=raise_op, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    verify_linearity(try_op)


def test_double_consume_of_affine_raises() -> None:
    """Wrap a raise's handler in a chain so two ops both reference it."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=Index().constant(0),
        name="r",
    )
    chain = ChainOp(lhs=raise_op, rhs=h, type=Index().constant(1).type, name="c")
    body = Block(result=chain, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    with pytest.raises(DoubleConsumeError, match="consumed 2 times"):
        verify_linearity(try_op)


def test_double_consume_of_linear_raises() -> None:
    intro = IntroduceLinearOp(name="lin")
    a = ConsumeLinearOp(input=intro, name="a")
    b = ConsumeLinearOp(input=intro, name="b")
    chain = ChainOp(lhs=a, rhs=b, type=Index(), name="c")
    with pytest.raises(DoubleConsumeError, match="consumed 2 times"):
        verify_linearity(chain)


# -- Verifier: Rule 2 (linear leak at block exit) ---------------------------


def test_linear_consumed_to_block_result_ok() -> None:
    """A linear value reaches the block result via a single consumer."""
    intro = IntroduceLinearOp(name="lin")
    consumed = ConsumeLinearOp(input=intro, name="c")
    verify_linearity(consumed)


def test_linear_as_block_result_ok() -> None:
    """A linear value that *is* the block result is not leaked — the
    obligation transfers to the surrounding scope."""
    intro = IntroduceLinearOp(name="lin")
    verify_linearity(intro)


def test_linear_leak_in_inner_block_raises() -> None:
    """An inner block captures a linear value, doesn't consume it, and
    isn't its own block result. Reported on the inner block."""
    intro = IntroduceLinearOp(name="lin")
    inner_result = Index().constant(7)
    # Inner block "uses" intro only by capturing it (without consuming
    # via an op).
    inner = Block(result=inner_result, captures=[intro])
    # Outer wraps the inner via a ChainOp that ties them together;
    # ChainOp takes operands, not blocks, so we use TryOp which has
    # blocks. We need an op that owns a block to surface the inner.
    # error.TryOp requires error_type and two blocks, so build one.
    try_op = error.TryOp(
        error_type=Index(),
        body=inner,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    with pytest.raises(LinearLeakError, match="never consumed"):
        verify_linearity(try_op)


# -- Verifier: Rule 3 (PARTIAL drains linear) ------------------------------


def test_partial_op_with_no_live_linear_ok() -> None:
    """A `raise` with no linear values in scope passes."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=Index().constant(0),
    )
    body = Block(result=raise_op, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    verify_linearity(try_op)


def test_partial_op_with_live_affine_ok() -> None:
    """An affine value live at a PARTIAL op is fine — Rule 3 only
    drains linear values. The handler itself is the canonical case."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=Index().constant(0),
    )
    body = Block(result=raise_op, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    verify_linearity(try_op)  # %h (Affine) live at raise → ok


def test_partial_op_leaks_live_linear_raises() -> None:
    """Introduce a linear value, then encounter a PARTIAL op (raise)
    earlier in topo order before consuming the linear. The verifier
    fires LinearLeakAtPartialError."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    intro = IntroduceLinearOp(name="lin")
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=Index().constant(0),
        name="r",
    )
    # Force topo order: raise depends on intro via a chain; the chain
    # spine ensures raise sees intro as already-introduced-but-not-yet-
    # consumed when it runs. Use ChainOp(lhs=intro, rhs=raise_op): the
    # block result is the chain, which ties both into the use-def graph.
    # raise comes after intro in topo (post-order).
    chain = ChainOp(lhs=intro, rhs=raise_op, type=LinearMarker(), name="c")
    body = Block(result=chain, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    with pytest.raises(LinearLeakAtPartialError, match="still live"):
        verify_linearity(try_op)


def test_partial_op_consuming_the_linear_itself_ok() -> None:
    """A PARTIAL op that itself consumes the live linear value drains it
    as part of its own deps — Rule 3 is satisfied."""
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    intro = IntroduceLinearOp(name="lin")
    # Use a ChainOp that depends on both `intro` and a raise: the raise
    # consumes the handler, the chain consumes intro (into the result),
    # and we order things so intro is consumed *before* the raise.
    consumed = ConsumeLinearOp(input=intro, name="c")
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=h,
        error=consumed,  # forces intro to be consumed before raise via use-def
        name="r",
    )
    body = Block(result=raise_op, parameters=[h])
    try_op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=Block(result=Index().constant(0)),
        type=Index(),
    )
    verify_linearity(try_op)


# -- LinearityError hierarchy ----------------------------------------------


def test_error_classes_share_base() -> None:
    assert issubclass(DoubleConsumeError, LinearityError)
    assert issubclass(LinearLeakError, LinearityError)
    assert issubclass(LinearLeakAtPartialError, LinearityError)
