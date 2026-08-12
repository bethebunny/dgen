"""Tests for per-op block-execution contracts in ``verify_linearity``.

Contracted ops (``_BLOCK_CONTRACTS``) charge linear captures precisely in
the parent's Γ instead of parking them at ``MAYBE_AVAILABLE``:

- ``unpack`` (EXACTLY_ONCE): a linear capture is consumed at the op.
- ``control_flow.if`` (ALTERNATIVES): consumed when every completing
  alternative captures it; left available when only diverging
  alternatives capture it; rejected when captured by only some
  completing alternatives (conditional consumption).
- ``error.try`` (BodyWithHandler): consumed for the cleanup-scope
  pattern (body and except both capture); otherwise permissive.

Ops declare these as traits in their ``.dgen`` definitions; the
verifier reaches them through the ``Op.verify_block_linearity``
protocol, which bespoke block-holding ops may override instead (see
the custom-op test at the bottom).

``memory.Reference`` is the linear type used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

import dgen
from dgen.asm.parser import parse
from dgen.dialect import Dialect
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.ir.verification import (
    BlockLinearityContext,
    DoubleConsumeError,
    LinearLeakError,
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
    | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    | %cond : index.Index = 1
    | %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%ref):
    |     %d1 : Nil = memory.deallocate(%ref)
    | else_body() captures(%ref):
    |     %d2 : Nil = memory.deallocate(%ref)
"""


def test_if_both_alternatives_consume_is_legal():
    """Every completing alternative consumes the capture — no error."""
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
    | %d3 : Nil = memory.deallocate(%ref)
    | %result : Nil = chain(%if, %d3)
"""
        )


def test_if_conditional_consumption_rejected():
    """Captured and consumed by one completing alternative only — the
    other completing alternative leaks it."""
    with pytest.raises(LinearLeakError):
        _verify("""
            | import control_flow
            | import index
            | import memory
            | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
            | %cond : index.Index = 1
            | %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%ref):
            |     %d1 : Nil = memory.deallocate(%ref)
            | else_body():
            |     %z : index.Index = 0
        """)


def test_if_diverging_alternative_leaves_thread_open():
    """A capture consumed only inside a diverging alternative does not
    charge the parent — the normal path's consume afterwards is legal
    (divergence-aware branch composition, docs/origins.md)."""
    _verify("""
        | import control_flow
        | import error
        | import index
        | import memory
        | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%ref):
        |     %cond : index.Index = 1
        |     %val : index.Index = control_flow.if(%cond, [], []) then_body() captures(%ref, %h):
        |         %d1 : Nil = memory.deallocate(%ref)
        |         %code : index.Index = 7
        |         %code2 : index.Index = chain(%code, %d1)
        |         %raised : Never = error.raise<index.Index>(%h, %code2)
        |     else_body():
        |         %five : index.Index = 5
        |     %d2 : Nil = memory.deallocate(%ref)
        |     %out : index.Index = chain(%val, %d2)
        | except(%err: index.Index):
        |     %one : index.Index = 1
        |     %rec : index.Index = chain(%err, %one)
    """)


# ---------------------------------------------------------------------------
# error.try — BODY_WITH_HANDLER (cleanup scope)
# ---------------------------------------------------------------------------


TRY_CLEANUP_SCOPE = """
    | import error
    | import index
    | import memory
    | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%ref):
    |     %seven : index.Index = 7
    |     %d1 : Nil = memory.deallocate(%ref)
    |     %bodyv : index.Index = chain(%seven, %d1)
    | except(%err: index.Index) captures(%ref):
    |     %d2 : Nil = memory.deallocate(%ref)
    |     %rec : index.Index = chain(%err, %d2)
"""


def test_try_cleanup_scope_is_legal():
    """Body consumes on the completing path, except on the divergent
    path — the canonical cleanup-scope pattern."""
    _verify(TRY_CLEANUP_SCOPE)


def test_try_cleanup_scope_charges_consumed():
    """Both children capture → the try consumes; parent reuse is caught."""
    with pytest.raises(DoubleConsumeError):
        _verify(
            TRY_CLEANUP_SCOPE
            + """
    | %d3 : Nil = memory.deallocate(%ref)
    | %result : index.Index = chain(%t, %d3)
"""
        )


def test_try_body_only_capture_stays_permissive():
    """A body-only capture may be discharged at-site before a raise —
    the try cannot charge it precisely and must not reject it."""
    _verify("""
        | import error
        | import index
        | import memory
        | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%ref):
        |     %seven : index.Index = 7
        |     %d1 : Nil = memory.deallocate(%ref)
        |     %bodyv : index.Index = chain(%seven, %d1)
        | except(%err: index.Index):
        |     %one : index.Index = 1
        |     %rec : index.Index = chain(%err, %one)
    """)


# ---------------------------------------------------------------------------
# unpack — EXACTLY_ONCE
# ---------------------------------------------------------------------------


UNPACK_CONSUMES_CAPTURE = """
    | import index
    | import memory
    | %r1 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    | %r2 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    | %v : index.Index = 3
    | %r1b : memory.Reference<index.Index> = memory.store(%r1, %v)
    | %loaded : Tuple<[index.Index, memory.Reference<index.Index>]> = memory.load(%r1b)
    | %out : index.Index = unpack(%loaded) body(%x: index.Index, %r1c: memory.Reference<index.Index>) captures(%r2):
    |     %d1 : Nil = memory.deallocate(%r1c)
    |     %d2 : Nil = memory.deallocate(%r2)
    |     %dd : Nil = chain(%d1, %d2)
    |     %res : index.Index = chain(%x, %dd)
"""


def test_unpack_capture_consume_is_legal():
    _verify(UNPACK_CONSUMES_CAPTURE)


def test_unpack_charges_consumed_so_reuse_is_double_consume():
    """The body runs exactly once and consumes its capture — a later
    direct consume in the parent is a double-consume."""
    with pytest.raises(DoubleConsumeError):
        _verify(
            UNPACK_CONSUMES_CAPTURE
            + """
    | %d3 : Nil = memory.deallocate(%r2)
    | %final : index.Index = chain(%out, %d3)
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
    trait; it implements the protocol directly, composing a standard
    context implementation."""

    body: dgen.Block
    type: dgen.Type
    __blocks__: ClassVar[tuple[str, ...]] = ("body",)

    def verify_block_linearity(self, ctx: BlockLinearityContext) -> None:
        ctx.exactly_once()


def _scope_over_consumed_ref() -> tuple[dgen.Value, dgen.Value]:
    """A _ScopeOp whose body captures and deallocates a Reference;
    returns (scope op, the reference)."""
    ref = memory.StackAllocateOp(
        element_type=Index(), type=memory.Reference(element_type=Index())
    )
    dealloc = memory.DeallocateOp(ptr=ref, type=Nil())
    inner = ChainOp(lhs=Index().constant(0), rhs=dealloc, type=Index())
    scope = _ScopeOp(body=dgen.Block(result=inner, captures=[ref]), type=Index())
    return scope, ref


def test_custom_op_protocol_override_is_legal():
    """A bespoke op's override charges its capture like exactly-once."""
    scope, _ = _scope_over_consumed_ref()
    verify_linearity(scope)


def test_custom_op_protocol_override_charges_consumed():
    """The override's charge is real: reusing the capture after the op
    is a double-consume, where the conservative default (no trait, no
    override) would have silently accepted it."""
    scope, ref = _scope_over_consumed_ref()
    second = memory.DeallocateOp(ptr=ref, type=Nil())
    root = ChainOp(lhs=scope, rhs=second, type=Index())
    with pytest.raises(DoubleConsumeError):
        verify_linearity(root)


# ---------------------------------------------------------------------------
# control_flow.for / while — ZeroOrMore
# ---------------------------------------------------------------------------


def test_loop_linear_capture_rejected():
    """A linear value captured into a loop body is unsound in both
    directions (zero iterations leak it, two double-consume it) — the
    ZeroOrMore contract rejects it outright."""
    with pytest.raises(ZeroOrMoreCaptureError):
        _verify("""
            | import control_flow
            | import index
            | import memory
            | %ref : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
            | %loop : Nil = control_flow.for<index.Index(0), index.Index(5)>([]) body(%iv: index.Index) captures(%ref):
            |     %d : Nil = memory.deallocate(%ref)
            | %r : Nil = chain(%loop, %loop)
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
