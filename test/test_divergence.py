"""Tests for generic divergence detection (``Value.totality``).

The classification rules — defined in ``docs/divergence.md`` — are:

- An op is *Partial* iff one of its operands or block captures has a type
  that declares the ``Handler<Diverge>`` trait.
- An op is *Divergent* iff it is Partial *and* its result type is ``Never``.
- An op is *Total* otherwise.

The tests below exercise the trait declarations on the existing handler
families (``RaiseHandler``, ``goto.Label``) and the per-value classification
property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dgen import Block, Op, Type, Value
from dgen.asm.parser import parse
from dgen.block import BlockParameter
from dgen.builtins import Totality
from dgen.dialects import builtin, error, goto
from dgen.dialects.builtin import ChainOp, Diverge, Handler, Never
from dgen.dialects.index import Index
from dgen.ir.traversal import all_values
from dgen.testing import strip_prefix
from dgen.type import Fields


# A non-registered op used by the synthetic divergence-shape tests below.
# Registering would require side-effects in a dialect; for unit purposes
# we set ``dialect``/``asm_name`` directly so any ASM print path that
# happens to fire still has the bookkeeping it needs.
@dataclass(eq=False, kw_only=True)
class _BranchLikeOp(Op):
    target: Value
    type: Type = Never()
    __operands__: ClassVar[Fields] = (("target", Type),)


_BranchLikeOp.dialect = goto.goto
_BranchLikeOp.asm_name = "_branch_like"


@dataclass(eq=False, kw_only=True)
class _WrapBlockOp(Op):
    body: Block
    type: Type = Index()
    __blocks__: ClassVar[tuple[str, ...]] = ("body",)


_WrapBlockOp.dialect = goto.goto
_WrapBlockOp.asm_name = "_wrap_block"


# -- Trait declarations on the handler families ------------------------------


def test_diverge_declares_effect_trait() -> None:
    """``Diverge`` is itself an ``Effect`` so it composes with ``Handler<E>``."""
    assert Diverge().has_trait(builtin.Effect())


def test_raise_handler_also_declares_handler_diverge() -> None:
    """``RaiseHandler<E>`` declares both ``Handler<Raise<E>>`` and
    ``Handler<Diverge>`` so generic divergence queries find it."""
    h = error.RaiseHandler(error_type=Index())
    assert h.has_trait(Handler(effect_type=error.Raise(error_type=Index())))
    assert h.has_trait(Handler(effect_type=Diverge()))


def test_label_declares_handler_diverge() -> None:
    """``goto.Label`` is the evidence that a non-local control transfer
    may happen, and so declares ``Handler<Diverge>``."""
    assert goto.Label().has_trait(Handler(effect_type=Diverge()))


# -- Totality on simple values ----------------------------------------------


def test_constant_is_total() -> None:
    """A bare constant has no operands or blocks — always TOTAL."""
    assert Index().constant(7).totality is Totality.TOTAL


def test_type_value_is_total() -> None:
    """A ``Type`` instance has no operands or blocks — always TOTAL."""
    assert Index().totality is Totality.TOTAL
    assert Diverge().totality is Totality.TOTAL


# -- Totality on the raise/try family ---------------------------------------


def test_raise_op_is_divergent() -> None:
    """``raise`` takes a ``RaiseHandler`` operand (Handler<Diverge> evidence)
    and has result type ``Never`` — the canonical Divergent op."""
    handler = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=handler,
        error=Index().constant(0),
    )
    assert raise_op.totality is Totality.DIVERGENT


def test_try_op_is_total() -> None:
    """A ``try`` constructs its handler internally and binds it as a body
    *parameter* (not capture), and its own result type is the user's ``T``,
    not ``Never``. So the try itself is TOTAL — the divergence is contained."""
    value = parse(
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
    assert isinstance(value, error.TryOp)
    assert value.totality is Totality.TOTAL


def test_inner_try_capturing_outer_handler_is_partial() -> None:
    """When an inner try's except captures the *outer* try's handler in
    order to re-raise, the inner try op becomes Partial: one of its block
    captures has type ``Handler<Diverge>``."""
    value = parse(
        strip_prefix("""
        | import error
        | import index
        | %outer : index.Index = error.try<index.Index>() body<%ho: error.RaiseHandler<index.Index>>():
        |     %inner : index.Index = error.try<index.Index>() body<%hi: error.RaiseHandler<index.Index>>():
        |         %ok : index.Index = 5
        |     except(%err: index.Index) captures(%ho):
        |         %r : Never = error.raise<index.Index>(%ho, %err)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    # Find the inner try (the one nested inside the outer body block).
    inner = next(
        v for v in all_values(value) if isinstance(v, error.TryOp) and v is not value
    )
    assert inner.totality is Totality.PARTIAL


# -- Totality on the goto family --------------------------------------------


def test_branch_with_label_operand_is_divergent() -> None:
    """An op that takes a ``Label`` (Handler<Diverge>) as an *operand* and
    whose result is ``Never`` is Divergent. Today's ``goto.branch`` carries
    the label as a *parameter* (compile-time), so for the operand-shaped
    case the test uses ``_BranchLikeOp`` defined at module scope."""
    label = BlockParameter(name="L", type=goto.Label())
    assert _BranchLikeOp(target=label).totality is Totality.DIVERGENT


def test_op_with_label_capture_but_normal_result_is_partial() -> None:
    """An op that owns a block capturing a ``Label`` but whose own result
    type is *not* ``Never`` is Partial — divergence is possible inside the
    block, but the op itself can still produce a value."""
    label = BlockParameter(name="L", type=goto.Label())
    seven = Index().constant(7)
    inner = ChainOp(lhs=seven, rhs=label, type=Index())
    body = Block(result=inner, captures=[label])
    assert _WrapBlockOp(body=body).totality is Totality.PARTIAL


# -- Round-trip from parsed IR ----------------------------------------------


def test_raise_op_in_parsed_ir_is_divergent() -> None:
    """The classification works on values pulled out of parsed IR: walk
    a try's body, find the raise, and confirm it lights up as Divergent."""
    value = parse(
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
    raise_op = next(v for v in all_values(value) if isinstance(v, error.RaiseOp))
    assert raise_op.totality is Totality.DIVERGENT
