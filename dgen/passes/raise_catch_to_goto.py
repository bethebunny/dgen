"""Lower ``error.catch`` and ``error.raise`` to the goto dialect.

The ``catch`` op evaluates to a ``RaiseHandler<error_type>`` value. The
handler flows through the enclosing dataflow like any other SSA value; its
``on_raise`` block is the continuation taken when a ``raise`` using that
handler fires.

Lowering:

- ``catch<T>() on_raise(%err: T): <body>`` becomes a ``ChainOp`` that
  threads a ``goto.label`` (derived from ``on_raise``) into the enclosing
  region while still producing a value typed ``RaiseHandler<T>`` for any
  downstream raise to point at. The handler value itself carries no runtime
  data (``RaiseHandler`` has ``layout Void``); it's a compile-time marker.
- Each ``raise<handler>(error)`` becomes a ``goto.branch`` targeting the
  label derived from the catch that produced ``handler``, passing ``error``
  as the block argument.

Because the pass framework visits values in topological order, every catch
is lowered before any raise that consumes its handler. Raise resolution
looks through the handler operand to find the catch's ``on_raise`` label
(stashed on the replacement ``ChainOp``).

v1 requires ``on_raise``'s result to diverge (type ``Never`` — another raise
to an outer handler, or other escape). ``emit_label_op`` inserts an
``unreachable`` terminator if the block body doesn't provide one, so a
misuse fails at JIT compile time rather than silently producing invalid IR.
"""

from __future__ import annotations

import dgen
from dgen.builtins import ConstantOp, pack
from dgen.dialects import error, goto
from dgen.dialects.builtin import ChainOp
from dgen.memory import Memory
from dgen.passes.pass_ import Pass, lowering_for


def _on_raise_label(handler: dgen.Value) -> goto.LabelOp:
    """Return the ``goto.label`` that a lowered catch threaded through its chain.

    Shape produced by ``lower_catch`` below: ``ChainOp(handler_const, label)``.
    """
    assert isinstance(handler, ChainOp), (
        f"expected lowered catch, got {type(handler).__name__}"
    )
    label = handler.rhs
    assert isinstance(label, goto.LabelOp), (
        f"expected on_raise label at chain.rhs, got {type(label).__name__}"
    )
    return label


class RaiseCatchToGoto(Pass):
    """Lower catch/raise pairs to goto.label + goto.branch.

    Catches erase to a chain threading a compile-time handler constant and a
    ``goto.label`` for ``on_raise``; raises become ``goto.branch`` to that
    label. No region wraps the body — the handler's live range is implicit
    in the dataflow, not a syntactic block.
    """

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._counter = 0

    @lowering_for(error.CatchOp)
    def lower_catch(self, op: error.CatchOp) -> dgen.Value | None:
        cid = self._counter
        self._counter += 1

        # Move the on_raise block into a fresh goto.label. Captures and
        # arguments carry over unchanged; ``emit_label_op`` skips to an
        # ``{name}_exit`` label so the enclosing block's linear flow is
        # preserved.
        on_raise = op.on_raise
        label_body = dgen.Block(
            args=list(on_raise.args),
            captures=list(on_raise.captures),
            result=on_raise.result,
        )
        on_raise_label = goto.LabelOp(
            name=f"on_raise{cid}",
            initial_arguments=pack([]),
            body=label_body,
        )

        # Handler value: a compile-time constant of RaiseHandler type. It
        # has ``layout Void`` so the runtime representation is empty. The
        # chain is how we keep the label alive in the enclosing block's
        # use-def graph without adding it as a separate capture.
        handler_type = op.type
        handler_const = ConstantOp(
            value=Memory.from_value(handler_type, None),
            type=handler_type,
        )
        return ChainOp(lhs=handler_const, rhs=on_raise_label, type=handler_type)

    @lowering_for(error.RaiseOp)
    def lower_raise(self, op: error.RaiseOp) -> dgen.Value | None:
        # By topological order the catch has already been lowered; the
        # handler operand is now the ``ChainOp(handler_const, on_raise_label)``
        # that replaced it. Pull the label out and emit a direct branch.
        on_raise_label = _on_raise_label(op.handler)
        return goto.BranchOp(
            target=on_raise_label,
            arguments=pack([op.error]),
        )
