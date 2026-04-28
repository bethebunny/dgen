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

import pytest

import dgen.imports
from dgen.asm.parser import parse
from dgen.dialects import builtin, error, goto
from dgen.dialects.index import Index
from dgen.ir.verification import (
    DoubleConsumeError,
    LinearityError,
    LinearLeakAtPartialError,
    LinearLeakError,
    verify_linearity,
)
from dgen.testing import strip_prefix
from dgen.type import Linearity


# A test-only dialect that introduces and consumes a linear-typed value.
# Loaded via the ``source=`` form of dgen.imports.load so the ASM parser
# resolves ``import _linear_test`` like any other dialect.
dgen.imports.load(
    "_linear_test",
    source=strip_prefix("""
        | from builtin import Linear
        |
        | type LinearMarker:
        |     layout Void
        |     has trait Linear
        |
        | op introduce_linear() -> LinearMarker
        | op consume_linear(input: LinearMarker)
    """),
)


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


def test_raise_handler_value_is_affine() -> None:
    """An IR-parsed RaiseHandler block parameter classifies as AFFINE."""
    value = parse(
        strip_prefix("""
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    handler = value.body.parameters[0]
    assert handler.linearity is Linearity.AFFINE
    assert handler.is_affine_or_linear
    assert not handler.is_linear


def test_linear_marker_value_is_linear() -> None:
    value = parse(
        strip_prefix("""
        | import _linear_test
        | %x : _linear_test.LinearMarker = _linear_test.introduce_linear()
    """)
    )
    assert value.linearity is Linearity.LINEAR
    assert value.is_linear


# -- Verifier: Rule 1 (at most one direct consumer) -------------------------


def test_unused_affine_handler_is_ok() -> None:
    """A try whose body never raises: the handler is a body parameter,
    captured nowhere, used nowhere. Affine permits zero consumers."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_raise_consumes_handler_once_ok() -> None:
    """A `raise` consumes its handler operand exactly once → fine."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        |     %r : Never = error.raise<index.Index>(%h, %v)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_double_consume_of_affine_raises() -> None:
    """Two ops both reference the same affine handler. Use a chain to
    keep both ops reachable from the body's result."""
    value = parse(
        strip_prefix("""
        | import builtin
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        |     %first : index.Index = builtin.chain(%v, %h)
        |     %r : Never = error.raise<index.Index>(%h, %v)
        |     %result : index.Index = builtin.chain(%first, %r)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    with pytest.raises(DoubleConsumeError, match="consumed 2 times"):
        verify_linearity(value)


def test_double_consume_of_linear_raises() -> None:
    """Two ops both consume the same linear value."""
    value = parse(
        strip_prefix("""
        | import _linear_test
        | import builtin
        | %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
        | %a : Nil = _linear_test.consume_linear(%lin)
        | %b : Nil = _linear_test.consume_linear(%lin)
        | %result : Nil = builtin.chain(%a, %b)
    """)
    )
    with pytest.raises(DoubleConsumeError, match="consumed 2 times"):
        verify_linearity(value)


# -- Verifier: Rule 2 (linear leak at block exit) ---------------------------


def test_linear_consumed_to_block_result_ok() -> None:
    """A linear value reaches the block result via a single consumer."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import _linear_test
        | %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
        | %c : Nil = _linear_test.consume_linear(%lin)
    """)
        )
    )


def test_linear_as_block_result_ok() -> None:
    """A linear value that *is* the block result is not leaked — the
    obligation transfers to the surrounding scope."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import _linear_test
        | %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
    """)
        )
    )


def test_linear_leak_in_inner_block_raises() -> None:
    """An inner block captures a linear value, doesn't consume it, and
    isn't its own block result. Reported on the inner block. (Flat-block
    leaks are structurally impossible: anything in ``block.values`` is
    reachable from ``block.result``, hence already consumed somewhere.)"""
    value = parse(
        strip_prefix("""
        | import _linear_test
        | import error
        | import index
        | %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%lin):
        |     %v : index.Index = 7
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    with pytest.raises(LinearLeakError, match="never consumed"):
        verify_linearity(value)


# -- Verifier: Rule 3 (PARTIAL drains linear) ------------------------------


def test_partial_op_with_no_live_linear_ok() -> None:
    """A `raise` with no linear values in scope passes."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        |     %r : Never = error.raise<index.Index>(%h, %v)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_partial_op_with_live_affine_ok() -> None:
    """An affine value live at a PARTIAL op is fine — Rule 3 only drains
    linear values. The handler itself is the canonical case."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        |     %r : Never = error.raise<index.Index>(%h, %v)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_partial_op_leaks_live_linear_raises() -> None:
    """Introduce a linear value, then encounter a PARTIAL op (raise)
    earlier in topo order before consuming the linear. The verifier
    fires LinearLeakAtPartialError."""
    value = parse(
        strip_prefix("""
        | import _linear_test
        | import builtin
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
        |     %v : index.Index = 7
        |     %r : Never = error.raise<index.Index>(%h, %v)
        |     %c : _linear_test.LinearMarker = builtin.chain(%lin, %r)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    with pytest.raises(LinearLeakAtPartialError, match="still live"):
        verify_linearity(value)


def test_partial_op_consuming_the_linear_itself_ok() -> None:
    """A PARTIAL op that itself consumes the live linear value drains it
    as part of its own deps — Rule 3 is satisfied."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import _linear_test
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
        |     %consumed : Nil = _linear_test.consume_linear(%lin)
        |     %r : Never = error.raise<index.Index>(%h, %consumed)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


# -- LinearityError hierarchy ----------------------------------------------


def test_error_classes_share_base() -> None:
    assert issubclass(DoubleConsumeError, LinearityError)
    assert issubclass(LinearLeakError, LinearityError)
    assert issubclass(LinearLeakAtPartialError, LinearityError)
