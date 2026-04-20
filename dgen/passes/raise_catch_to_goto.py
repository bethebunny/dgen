"""Lower ``error.catch`` and ``error.raise`` to the goto dialect.

A ``catch`` op runs ``body`` with a fresh ``RaiseHandler<error_type>`` bound to
the block's ``handler`` parameter. Every ``error.raise`` in the body whose
handler resolves to that parameter transfers control to ``on_raise``.

The lowering mirrors ``ControlFlowToGoto.lower_if``:

    error.catch<T> body<%h: RaiseHandler<T>>(): <body> on_raise(%err: T): <on_raise>

becomes::

    goto.region([]) catch<%self, %exit>(%result: ResultT):
        goto.label([]) on_raise captures(%self):
            <on_raise body with %err bound to its arg>
            goto.branch<%self>([<on_raise.result>])
        <catch body with every raise<%h>(err) rewritten to goto.branch<on_raise_label>([err])>
        goto.branch<%self>([<body.result>])

Handler resolution is compile-time. A ``raise`` whose handler is a block
parameter *other* than ``catch.body``'s handler is left alone — it targets an
enclosing catch that will rewrite it when the pass visits that op. An
unresolved handler at the end of the pass is a user error (v1 design requires
all raises to be handled within the function).
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import goto
from dgen.dialects.builtin import ChainOp, Never, Nil
from dgen.dialects.error import RaiseOp
from dgen.builtins import pack
from dgen.error import CatchOp
from dgen.ir.traversal import all_values
from dgen.passes.pass_ import Pass, lowering_for


def _rewrite_raises(
    block: dgen.Block,
    handler: BlockParameter,
    on_raise_label: goto.LabelOp,
) -> bool:
    """Rewrite every ``raise<handler>(err)`` in *block* to a branch to *on_raise_label*.

    Recurses into nested blocks, but stops at ``CatchOp`` bodies — an inner
    catch is responsible for its own handler, and a raise inside an inner
    catch's *on_raise* block still uses the outer handler (by capture).
    Returns True iff at least one rewrite happened anywhere under *block*.

    When a rewrite happens inside a nested block, ``on_raise_label`` becomes a
    new dependency of that block; it's added to the block's captures so the
    closed-block invariant still holds. The caller (one scope up) does the
    same for its own block until the scope that owns ``on_raise_label`` is
    reached.

    The check ``op.handler is handler`` is an object-identity test on the
    compile-time parameter value — catch produces exactly one BlockParameter
    per instance and threads it through as the raise's handler parameter.
    """
    rewrote = False
    for value in list(block.values):
        if isinstance(value, RaiseOp) and value.handler is handler:
            block.replace_uses_of(
                value,
                goto.BranchOp(
                    target=on_raise_label,
                    arguments=pack([value.error]),
                ),
            )
            rewrote = True
            continue
        if isinstance(value, CatchOp):
            # Don't descend into the nested catch's body (it binds its own
            # handler). on_raise may still raise via an outer handler — if
            # so, on_raise_label escapes as a dependency of the inner catch's
            # on_raise block, and (when that inner catch is lowered) of its
            # body as well. Both must declare it as a capture so the
            # closed-block invariant holds at every intermediate scope.
            if _rewrite_raises(value.on_raise, handler, on_raise_label):
                rewrote = True
                if on_raise_label not in value.on_raise.captures:
                    value.on_raise.captures.append(on_raise_label)
                if on_raise_label not in value.body.captures:
                    value.body.captures.append(on_raise_label)
            continue
        if isinstance(value, dgen.Op):
            for _, child in value.blocks:
                if _rewrite_raises(child, handler, on_raise_label):
                    rewrote = True
                    if on_raise_label not in child.captures:
                        child.captures.append(on_raise_label)
    return rewrote


def _drop_handler_captures(block: dgen.Block, handler: BlockParameter) -> None:
    """Remove *handler* from *block*'s and every nested block's captures list.

    After ``_rewrite_raises`` the handler's only uses — the raise ops — have
    been replaced with goto.branch ops. Declared captures referencing the
    handler are now stale and would fail closed-block verification once the
    enclosing catch is lowered away.
    """
    block.captures = [c for c in block.captures if c is not handler]
    for op in block.ops:
        for _, child in op.blocks:
            _drop_handler_captures(child, handler)


class RaiseCatchToGoto(Pass):
    """Lower catch/raise pairs to goto.region + goto.label + goto.branch."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._counter = 0

    @lowering_for(CatchOp)
    def lower_catch(self, op: CatchOp) -> dgen.Value | None:
        cid = self._counter
        self._counter += 1

        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"catch_exit{cid}", type=goto.Label())
        merge_result = BlockArgument(name=f"catch_result{cid}", type=op.type)

        # Handler parameter and error arg must be resolved before we assemble
        # the new blocks (they're attached to op.body / op.on_raise).
        if not op.body.parameters:
            raise AssertionError(
                f"CatchOp %{op.name}: body block missing handler parameter"
            )
        handler = op.body.parameters[0]
        body_diverges = isinstance(op.body.result.type, Never)

        # ---- on_raise label: run on_raise, then branch to merge. ----
        on_raise = op.on_raise
        on_raise_captures = [merge_self, *on_raise.captures]
        on_raise_body = dgen.Block(
            args=list(on_raise.args),
            captures=on_raise_captures,
            result=goto.BranchOp(target=merge_self, arguments=pack([on_raise.result])),
        )
        on_raise_label = goto.LabelOp(
            name=f"on_raise{cid}",
            initial_arguments=pack([]),
            body=on_raise_body,
        )

        # ---- Rewrite raises in body to target the on_raise label. ----
        body_raises = _rewrite_raises(op.body, handler, on_raise_label)
        _drop_handler_captures(op.body, handler)

        # ---- Body terminator ----
        #
        # If the body's natural result is a divergent value (Never — typically
        # the result of a raise), rewriting already replaced it with a branch
        # to on_raise_label, so the body terminates on that path. Emitting
        # another branch would create an unreachable post-terminator block
        # that still gets listed as a phi predecessor in LLVM. Skip it.
        if body_diverges:
            body_terminator = op.body.result
        else:
            body_terminator = goto.BranchOp(
                target=merge_self, arguments=pack([op.body.result])
            )

        # The handler parameter is erased; after rewriting it has no remaining
        # users. Everything else in catch.body's captures flows through.
        #
        # on_raise_label is NOT added to captures — if the body raises, the
        # rewritten branch inside the body references it as a compile-time
        # target, so it's already in the region's use-def subgraph. If the
        # body never raises, on_raise_label is unreachable (dead code),
        # which is fine: the region falls through to merge via
        # body_terminator and the label is elided by codegen.
        region_captures = [c for c in op.body.captures if c is not handler]

        # If the body raises, the on_raise label is needed in the region's
        # own scope (so its reference to ``merge_self`` is valid). Chain it
        # as a sibling of body_terminator. Otherwise the label is unreachable
        # dead code; dropping it keeps LLVM's phi predecessor set clean.
        if body_raises:
            region_dispatch: dgen.Value = ChainOp(
                lhs=body_terminator,
                rhs=on_raise_label,
                type=Nil(),
            )
        else:
            region_dispatch = body_terminator

        region_body = dgen.Block(
            parameters=[merge_self, merge_exit],
            args=[merge_result],
            captures=region_captures,
            # Chain pattern mirrors lower_if: merge_result is the phi value;
            # region_dispatch is the dispatch (side-effecting branch + label
            # declaration) that drives the region.
            result=ChainOp(lhs=merge_result, rhs=region_dispatch, type=op.type),
        )

        return goto.RegionOp(
            name=f"catch{cid}",
            initial_arguments=pack([]),
            type=op.type,
            body=region_body,
        )

    def verify_postconditions(self, value: dgen.Value) -> None:
        """No CatchOp or RaiseOp must remain in the IR after this pass."""
        super().verify_postconditions(value)
        for v in all_values(value):
            if isinstance(v, (CatchOp, RaiseOp)):
                raise AssertionError(
                    f"{type(v).__name__} %{v.name} survived RaiseCatchToGoto "
                    f"— handler likely did not resolve to an enclosing catch"
                )
