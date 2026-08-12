"""Tests for aggregate taint in linearity and totality.

An aggregate's multiplicity is the least permissive of its components,
so a tuple carrying an Origin is linear and a tuple carrying a raise
handler is affine. Totality taints through components too: possessing
an aggregate that contains a Handler<Diverge> means the handler can be
projected out, so a dependency on the aggregate may diverge.
"""

from __future__ import annotations

import pytest

from dgen.asm.parser import parse
from dgen.block import BlockArgument, BlockParameter
from dgen.builtins import pack
from dgen.dialects import error, memory
from dgen.dialects.builtin import ChainOp, Tuple
from dgen.dialects.index import Index
from dgen.ir.verification import (
    DoubleConsumeError,
    Linearity,
    LinearLeakError,
    linearity,
    verify_linearity,
)
from dgen.testing import strip_prefix
from dgen.type import Totality


def _value_of(t: object) -> BlockArgument:
    return BlockArgument(name="v", type=t)


def _tuple_of(*types: object) -> Tuple:
    return Tuple(types=pack(list(types)))


# ---------------------------------------------------------------------------
# Multiplicity join
# ---------------------------------------------------------------------------


def test_tuple_with_origin_is_linear():
    t = _tuple_of(memory.Reference(element_type=Index()), memory.Origin())
    assert linearity(_value_of(t)) is Linearity.LINEAR


def test_tuple_with_handler_is_affine():
    t = _tuple_of(error.RaiseHandler(error_type=Index()), Index())
    assert linearity(_value_of(t)) is Linearity.AFFINE


def test_tuple_of_unrestricted_stays_unrestricted():
    t = _tuple_of(Index(), memory.Reference(element_type=Index()))
    assert linearity(_value_of(t)) is Linearity.UNRESTRICTED


def test_nested_tuple_taints_through():
    inner = _tuple_of(Index(), memory.Origin())
    t = _tuple_of(Index(), inner)
    assert linearity(_value_of(t)) is Linearity.LINEAR


def test_linear_beats_affine_in_the_join():
    t = _tuple_of(error.RaiseHandler(error_type=Index()), memory.Origin())
    assert linearity(_value_of(t)) is Linearity.LINEAR


# ---------------------------------------------------------------------------
# Totality containment
# ---------------------------------------------------------------------------


def test_dependency_on_handler_carrying_tuple_is_partial():
    h = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    x = BlockArgument(name="x", type=Index())
    bundle = pack([h, x])
    probe = ChainOp(lhs=x, rhs=bundle, type=Index())
    assert probe.totality is Totality.PARTIAL


def test_dependency_on_plain_tuple_is_total():
    x = BlockArgument(name="x", type=Index())
    bundle = pack([x, x])
    probe = ChainOp(lhs=x, rhs=bundle, type=Index())
    assert probe.totality is Totality.TOTAL


# ---------------------------------------------------------------------------
# The allocation tuple is single-consume end to end
# ---------------------------------------------------------------------------


def test_double_unpack_of_allocation_rejected():
    """Unpacking the same allocation twice would duplicate its origin."""
    value = parse(
        strip_prefix("""
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %r1 : Nil = unpack(%alloc) body(%ref1: memory.Reference<index.Index>, %o1: memory.Origin):
        |     %d1 : Nil = memory.destroy(%o1)
        | %r2 : Nil = unpack(%alloc) body(%ref2: memory.Reference<index.Index>, %o2: memory.Origin):
        |     %d2 : Nil = memory.destroy(%o2)
        | %res : Nil = chain(%r1, %r2)
    """)
    )
    with pytest.raises(DoubleConsumeError):
        verify_linearity(value)


def test_allocation_tuple_leak_rejected():
    """A captured allocation tuple that is never consumed is a leak."""
    value = parse(
        strip_prefix("""
        | import control_flow
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %cond : index.Index = 1
        | %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |     %z : index.Index = 0
        | else_body() captures(%alloc):
        |     %z2 : index.Index = 1
    """)
    )
    with pytest.raises(LinearLeakError):
        verify_linearity(value)
