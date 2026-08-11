"""Tests for per-op block-execution contracts in ``verify_linearity``.

Contracted ops charge linear captures precisely in the parent's Γ
instead of parking them at ``MAYBE_AVAILABLE``.

- ``unpack`` (ExactlyOnce) consumes a linear capture at the op.
- ``control_flow.if`` (Alternatives) consumes when every completing
  alternative captures. It leaves the thread open when only diverging
  alternatives capture, and rejects a capture by only some completing
  alternatives as conditional consumption.
- ``error.try`` (BodyWithHandler) consumes for the cleanup-scope
  pattern where body and except both capture. Other shapes stay
  permissive.
- ``control_flow.for``/``while`` (ZeroOrMore) reject linear captures.

Ops declare these as traits in their ``.dgen`` definitions. The
verifier reaches them through the ``Op.verify_block_linearity``
protocol, which bespoke block-holding ops may override instead. See
the custom-op tests at the bottom.

``memory.Origin`` is the linear type used throughout. Origins come from
allocation tuples, so each scenario opens with an ``unpack``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

import dgen
from dgen.asm.parser import parse
from dgen.dialect import Dialect
from dgen.ir.verification import (
    BlockLinearityContext,
    DoubleConsumeError,
    LinearLeakError,
    UndeclaredBlockContractError,
    ZeroOrMoreCaptureError,
    verify_linearity,
)
from dgen.testing import strip_prefix


def _verify(ir: str) -> None:
    verify_linearity(parse(strip_prefix(ir)))


# ---------------------------------------------------------------------------
# control_flow.if — ALTERNATIVES
# ---------------------------------------------------------------------------


IF_BOTH_CONSUME = """
    | import control_flow
    | import index
    | import memory
    | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    | %r : Nil = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
    |     %cond : index.Index = 1
    |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%o):
    |         %d1 : Nil = memory.destroy(%o)
    |     else_body() captures(%o):
    |         %d2 : Nil = memory.destroy(%o)
"""


def test_if_both_alternatives_consume_is_legal():
    """Every completing alternative consumes the capture. No error."""
    _verify(IF_BOTH_CONSUME)


def test_if_charges_consumed_so_reuse_is_double_consume():
    """After both alternatives consume, a later direct consume is caught.

    Under the pre-contract MAYBE_AVAILABLE treatment this reuse was
    silently accepted.
    """
    with pytest.raises(DoubleConsumeError):
        _verify(
            IF_BOTH_CONSUME
            + """
    |     %d3 : Nil = memory.destroy(%o)
    |     %res : Nil = chain(%if, %d3)
"""
        )


def test_if_conditional_consumption_rejected():
    """Captured and consumed by one completing alternative only. The
    other completing alternative leaks it."""
    with pytest.raises(LinearLeakError):
        _verify("""
            | import control_flow
            | import index
            | import memory
            | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
            | %r : Nil = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
            |     %cond : index.Index = 1
            |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%o):
            |         %d1 : Nil = memory.destroy(%o)
            |     else_body():
            |         %z : index.Index = 0
        """)


def test_if_diverging_alternative_leaves_thread_open():
    """A capture consumed only inside a diverging alternative does not
    charge the parent. The normal path's consume afterwards is legal
    per divergence-aware branch composition (docs/origins.md)."""
    _verify("""
        | import control_flow
        | import error
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %r : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
        |     %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%o):
        |         %cond : index.Index = 1
        |         %val : index.Index = control_flow.if(%cond, [], []) then_body() captures(%o, %h):
        |             %d1 : Nil = memory.destroy(%o)
        |             %code : index.Index = 7
        |             %code2 : index.Index = chain(%code, %d1)
        |             %raised : Never = error.raise<index.Index>(%h, %code2)
        |         else_body():
        |             %five : index.Index = 5
        |         %d2 : Nil = memory.destroy(%o)
        |         %out : index.Index = chain(%val, %d2)
        |     except(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = chain(%err, %one)
    """)


# ---------------------------------------------------------------------------
# error.try — BODY_WITH_HANDLER (cleanup scope)
# ---------------------------------------------------------------------------


TRY_CLEANUP_SCOPE = """
    | import error
    | import index
    | import memory
    | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    | %r : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
    |     %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%o):
    |         %seven : index.Index = 7
    |         %d1 : Nil = memory.destroy(%o)
    |         %bodyv : index.Index = chain(%seven, %d1)
    |     except(%err: index.Index) captures(%o):
    |         %d2 : Nil = memory.destroy(%o)
    |         %rec : index.Index = chain(%err, %d2)
"""


def test_try_cleanup_scope_is_legal():
    """Body consumes on the completing path and except consumes on the
    divergent path. The canonical cleanup-scope pattern."""
    _verify(TRY_CLEANUP_SCOPE)


def test_try_cleanup_scope_charges_consumed():
    """Both children capture, so the try consumes. Parent reuse is
    caught."""
    with pytest.raises(DoubleConsumeError):
        _verify(
            TRY_CLEANUP_SCOPE
            + """
    |     %d3 : Nil = memory.destroy(%o)
    |     %res : index.Index = chain(%t, %d3)
"""
        )


def test_try_body_only_capture_stays_permissive():
    """A body-only capture may be discharged at-site before a raise.
    The try cannot charge it precisely and must not reject it."""
    _verify("""
        | import error
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %r : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
        |     %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%o):
        |         %seven : index.Index = 7
        |         %d1 : Nil = memory.destroy(%o)
        |         %bodyv : index.Index = chain(%seven, %d1)
        |     except(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = chain(%err, %one)
    """)


# ---------------------------------------------------------------------------
# unpack — EXACTLY_ONCE
# ---------------------------------------------------------------------------


UNPACK_CONSUMES_CAPTURE = """
    | import index
    | import memory
    | %a1 : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    | %a2 : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    | %r : index.Index = unpack(%a1) body(%ref1: memory.Reference<index.Index>, %oA: memory.Origin) captures(%a2):
    |     %inner : index.Index = unpack(%a2) body(%ref2: memory.Reference<index.Index>, %oB: memory.Origin) captures(%oA):
    |         %d1 : Nil = memory.destroy(%oA)
    |         %d2 : Nil = memory.destroy(%oB)
    |         %dd : Nil = chain(%d1, %d2)
    |         %five : index.Index = 5
    |         %res : index.Index = chain(%five, %dd)
"""


def test_unpack_capture_consume_is_legal():
    _verify(UNPACK_CONSUMES_CAPTURE)


def test_unpack_charges_consumed_so_reuse_is_double_consume():
    """The body runs exactly once and consumes its capture. A later
    direct consume in the parent is a double-consume."""
    with pytest.raises(DoubleConsumeError):
        _verify(
            UNPACK_CONSUMES_CAPTURE
            + """
    |     %d3 : Nil = memory.destroy(%oA)
    |     %final : index.Index = chain(%inner, %d3)
"""
        )


# ---------------------------------------------------------------------------
# Custom block-holding ops override Op.verify_block_linearity
# ---------------------------------------------------------------------------


_test_dialect = Dialect("linearity_contract_test")


@_test_dialect.op("scope")
@dataclass(eq=False)
class _ScopeOp(dgen.Op):
    """Test-only block-holding op with no declared block-execution
    trait. It implements the protocol directly, composing a standard
    context implementation."""

    body: dgen.Block
    type: dgen.Type
    __blocks__: ClassVar[tuple[str, ...]] = ("body",)

    def verify_block_linearity(self, ctx: BlockLinearityContext) -> None:
        ctx.exactly_once()


SCOPE_CONSUMES_CAPTURE = """
    | import index
    | import linearity_contract_test
    | import memory
    | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    | %r : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
    |     %s : index.Index = linearity_contract_test.scope() body() captures(%o):
    |         %zero : index.Index = 0
    |         %d : Nil = memory.destroy(%o)
    |         %inner : index.Index = chain(%zero, %d)
"""


def test_custom_op_protocol_override_is_legal():
    """A bespoke op's override charges its capture like exactly-once."""
    _verify(SCOPE_CONSUMES_CAPTURE)


def test_custom_op_protocol_override_charges_consumed():
    """The override's charge is real. Reusing the capture after the op
    is a double-consume."""
    with pytest.raises(DoubleConsumeError):
        _verify(
            SCOPE_CONSUMES_CAPTURE
            + """
    |     %d2 : Nil = memory.destroy(%o)
    |     %res : index.Index = chain(%s, %d2)
"""
        )


# ---------------------------------------------------------------------------
# control_flow.for / while — ZeroOrMore
# ---------------------------------------------------------------------------


def test_loop_linear_capture_rejected():
    """A linear value captured into a loop body is unsound in both
    directions. Zero iterations leak it and two double-consume it. The
    ZeroOrMore contract rejects it outright."""
    with pytest.raises(ZeroOrMoreCaptureError):
        _verify("""
            | import control_flow
            | import index
            | import memory
            | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
            | %r : Nil = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin):
            |     %loop : Nil = control_flow.for<index.Index(0), index.Index(5)>([]) body(%iv: index.Index) captures(%o):
            |         %d : Nil = memory.destroy(%o)
        """)


def test_loop_unrestricted_capture_still_fine():
    """Unrestricted captures into loops are untouched by the contract."""
    _verify("""
        | import control_flow
        | import index
        | import memory
        | %alloc : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
        | %loop : Nil = control_flow.for<index.Index(0), index.Index(5)>([]) body(%iv: index.Index) captures(%alloc):
        |     %v : index.Index = 1
        |     %s : Nil = memory.buffer_store(%alloc, %alloc, index.Index(0), %v)
    """)


@_test_dialect.op("undeclared_scope")
@dataclass(eq=False)
class _UndeclaredScopeOp(dgen.Op):
    """Test-only block-holding op with no trait and no override."""

    body: dgen.Block
    type: dgen.Type
    __blocks__: ClassVar[tuple[str, ...]] = ("body",)


def test_undeclared_block_holding_op_fails():
    """A block-holding op with no trait and no override is rejected."""
    with pytest.raises(UndeclaredBlockContractError):
        _verify("""
            | import index
            | import linearity_contract_test
            | %u : index.Index = linearity_contract_test.undeclared_scope() body():
            |     %zero : index.Index = 0
        """)
