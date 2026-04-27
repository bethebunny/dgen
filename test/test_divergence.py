"""Tests for generic divergence detection (``Value.totality``).

The classification rules — defined in the "Generic Divergence Detection"
section of ``docs/control-flow.md`` — are:

- An op is *Partial* iff one of its operands, parameters, or owned-block
  captures has a type that declares the ``Handler<Diverge>`` trait.
- An op is *Divergent* iff it is Partial *and* its result type is ``Never``.
- An op is *Total* otherwise.
"""

from __future__ import annotations

from dgen.asm.parser import parse
from dgen.block import BlockParameter
from dgen.builtins import pack
from dgen.dialects import builtin, error, goto
from dgen.dialects.builtin import Diverge, Handler
from dgen.dialects.index import Index
from dgen.ir.traversal import all_values
from dgen.testing import strip_prefix
from dgen.type import Totality


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
    """A bare constant has no operands, parameters, or blocks — always TOTAL."""
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


def test_branch_with_label_parameter_is_partial() -> None:
    """``goto.branch<target: Label>`` carries its target as a *parameter*.
    With the parameter check the branch op is detected as having a
    ``Handler<Diverge>`` in scope, so it's at least Partial. It would be
    DIVERGENT if the branch's result type were ``Never`` — see TODO.md
    (Type system / effects) for the planned change."""
    label = BlockParameter(name="L", type=goto.Label())
    op = goto.BranchOp(target=label, arguments=pack())
    assert op.totality is Totality.PARTIAL


def test_conditional_branch_is_partial() -> None:
    """``goto.conditional_branch`` carries *two* Label parameters; either
    one is enough to flag it Partial."""
    t = BlockParameter(name="T", type=goto.Label())
    f = BlockParameter(name="F", type=goto.Label())
    op = goto.ConditionalBranchOp(
        true_target=t,
        false_target=f,
        condition=Index().constant(0),
        true_arguments=pack(),
        false_arguments=pack(),
    )
    assert op.totality is Totality.PARTIAL


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
