"""Tests for the Linear / Affine trait pair and ``verify_linearity``.

The classification — defined in ``docs/linear_types.md`` — is read off
the value's *type*:

- ``Linearity.LINEAR``       — type has trait ``Linear``: consume exactly once
- ``Linearity.AFFINE``       — type has trait ``Affine``: consume at most once
- ``Linearity.UNRESTRICTED`` — neither

The verifier maintains a per-block context ``Γ : Value → {Available,
Consumed}``; ops transition Γ entries to ``Consumed`` when they
reference a substructural operand / parameter / capture-into-child;
at block exit any ``Linear`` value still ``Available`` (and not
``block.result``, which is yielded to the surrounding scope) is a
leak.
"""

from __future__ import annotations

import pytest

import dgen.imports
from dgen.asm.parser import parse
from dgen.dialects import builtin, error, goto
from dgen.dialects.index import Index
from dgen.ir.verification import (
    DoubleConsumeError,
    Linearity,
    LinearityError,
    LinearLeakError,
    is_affine_or_linear,
    is_linear,
    linearity,
    verify_linearity,
)
from dgen.testing import strip_prefix


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


def test_label_is_affine() -> None:
    """Labels are divergence handlers — branched to once per execution
    path. The verifier's MaybeAvailable state lets multiple goto.label
    ops in a region all capture the same %exit without tripping
    DoubleConsume."""
    assert goto.Label().has_trait(builtin.Affine())


def test_index_is_unrestricted() -> None:
    assert not Index().has_trait(builtin.Affine())
    assert not Index().has_trait(builtin.Linear())


# -- linearity / is_linear / is_affine_or_linear ---------------------------


def test_constant_is_unrestricted() -> None:
    c = Index().constant(7)
    assert linearity(c) is Linearity.UNRESTRICTED
    assert not is_affine_or_linear(c)


def test_type_value_is_unrestricted() -> None:
    """Type instances are universe-1 metadata, not resources."""
    assert linearity(Index()) is Linearity.UNRESTRICTED


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
    assert linearity(handler) is Linearity.AFFINE
    assert is_affine_or_linear(handler)
    assert not is_linear(handler)


def test_linear_marker_value_is_linear() -> None:
    value = parse(
        strip_prefix("""
        | import _linear_test
        | %x : _linear_test.LinearMarker = _linear_test.introduce_linear()
    """)
    )
    assert linearity(value) is Linearity.LINEAR
    assert is_linear(value)


# -- Verifier: pass cases --------------------------------------------------


def test_unused_affine_handler_is_ok() -> None:
    """A try whose body never raises: the handler is a body parameter,
    captured nowhere, used nowhere. Affine permits zero consumers, so
    the handler stays ``Available`` at block exit and that is fine."""
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
    obligation transfers to the surrounding scope (yield = consume)."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import _linear_test
        | %lin : _linear_test.LinearMarker = _linear_test.introduce_linear()
    """)
        )
    )


def test_partial_op_consuming_the_linear_itself_ok() -> None:
    """A PARTIAL op (raise) that itself consumes a linear chain leaves
    no Available linear values at block exit — the exit check passes."""
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


def test_capture_into_one_ops_two_children_is_ok() -> None:
    """Both children of a single try op capture the outer handler. With
    the unknown-block-op rule, the parent op transitions the captured
    value to ``MaybeAvailable`` once (dedup'd across children, since
    the captures are surfaced through one op's ``Value.dependencies``
    walk only once via the set-comprehension)."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import error
        | import index
        | %outer : index.Index = error.try<index.Index>() body<%ho: error.RaiseHandler<index.Index>>():
        |     %inner : index.Index = error.try<index.Index>() body<%hi: error.RaiseHandler<index.Index>>() captures(%ho):
        |         %v : index.Index = 7
        |         %r1 : Never = error.raise<index.Index>(%ho, %v)
        |     except(%err: index.Index) captures(%ho):
        |         %r2 : Never = error.raise<index.Index>(%ho, %err)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_two_sibling_block_ops_capturing_same_affine_is_ok() -> None:
    """Two *sibling* block-holding ops both capture the same outer
    handler. The MaybeAvailable rule is what makes this work:
    ``Available → MaybeAvailable → MaybeAvailable``. Without it,
    the second sibling's capture would trip ``DoubleConsumeError``.

    This is the same shape as the post-lowering goto if/else (two
    ``goto.label`` ops both capturing ``%exit``) — the test exercises
    the rule via ``error.try`` because writing valid raw goto IR by
    hand is fiddlier."""
    verify_linearity(
        parse(
            strip_prefix("""
        | import builtin
        | import error
        | import index
        | %outer : index.Index = error.try<index.Index>() body<%ho: error.RaiseHandler<index.Index>>():
        |     %a : index.Index = error.try<index.Index>() body<%h1: error.RaiseHandler<index.Index>>() captures(%ho):
        |         %v1 : index.Index = 1
        |         %r1 : Never = error.raise<index.Index>(%ho, %v1)
        |     except(%err1: index.Index):
        |         %z1 : index.Index = 0
        |     %b : index.Index = error.try<index.Index>() body<%h2: error.RaiseHandler<index.Index>>() captures(%ho):
        |         %v2 : index.Index = 2
        |         %r2 : Never = error.raise<index.Index>(%ho, %v2)
        |     except(%err2: index.Index):
        |         %z2 : index.Index = 0
        |     %sum : index.Index = builtin.chain(%a, %b)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
        )
    )


def test_goto_if_else_through_lowering_is_ok() -> None:
    """End-to-end: a high-level ``control_flow.if`` lowers to a goto
    region with two label ops each capturing ``%exit``. Marking
    ``goto.Label`` Affine + the MaybeAvailable rule together let
    ``ControlFlowToGoto.verify_postconditions`` accept this IR."""
    from dgen.passes.compiler import Compiler, IdentityPass
    from dgen.passes.control_flow_to_goto import ControlFlowToGoto

    value = parse(
        strip_prefix("""
        | import algebra
        | import control_flow
        | import index
        | import number
        | %five : index.Index = 5
        | %ten : index.Index = 10
        | %cond : number.Boolean = algebra.less_than(%five, %ten)
        | %r : index.Index = control_flow.if(%cond, [], []) then_body():
        |     %ta : index.Index = 1
        | else_body():
        |     %tb : index.Index = 2
    """)
    )
    Compiler([ControlFlowToGoto()], IdentityPass()).compile(value)


# -- Verifier: fail cases --------------------------------------------------


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
    with pytest.raises(DoubleConsumeError, match="consumed twice"):
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
    with pytest.raises(DoubleConsumeError, match="consumed twice"):
        verify_linearity(value)


def test_linear_leak_in_inner_block_raises() -> None:
    """An inner block captures a linear value, doesn't consume it, and
    isn't its own block result. The inner block's own exit check fires.
    (Flat-block leaks are structurally impossible: anything in
    ``block.values`` is reachable from ``block.result``, hence already
    consumed by some op.)"""
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
    with pytest.raises(LinearLeakError, match="AVAILABLE at block exit"):
        verify_linearity(value)


# -- LinearityError hierarchy ----------------------------------------------


def test_error_classes_share_base() -> None:
    assert issubclass(DoubleConsumeError, LinearityError)
    assert issubclass(LinearLeakError, LinearityError)
