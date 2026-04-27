"""Lower ``error.try`` and ``error.raise`` to the goto dialect.

The handler that ``try`` binds as the body block's ``handler`` parameter
is the *evidence* a raise has somewhere to go. ``goto.label`` plays the
exact same role at the next layer down — evidence a branch has somewhere
to go. Lowering replaces the handler with its corresponding label in the
body, which turns every ``raise(handler, error)`` into
``raise(label, error)``; a per-op handler then converts each one into
``goto.branch(label, error)``.

That replacement is the whole substitution. The framework's
``replace_uses_of`` cascade carries it through nested blocks (raise
operands, capture lists, everything). No bespoke recursive rewriter, no
manual capture propagation — same machinery passes already use to
substitute one value for another anywhere in an IR subtree.

The dispatch order works out: a try is at an outer level than the raises
it binds, so the framework visits the try first; ``lower_try`` performs
the handler→label substitution; the framework then recurses into the new
region's body and visits each raise with its handler operand already a
``goto.Label``.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockParameter
from dgen.builtins import pack
from dgen.dialects import error, goto
from dgen.dialects.builtin import Never
from dgen.ir.traversal import all_values
from dgen.passes.control_flow_to_goto import redirect_to_exit
from dgen.passes.pass_ import Pass, lowering_for


class UndischargedEffectError(AssertionError):
    """Raised when a ``raise`` has no enclosing ``try`` to discharge its effect.

    v1 requires every effect op to be discharged within the enclosing
    function. A ``RaiseOp`` whose handler operand isn't a ``goto.LabelOp``
    after this pass runs means no enclosing try produced it — either the
    raise sits outside any try's scope, or the handler was constructed by
    something other than ``try``.
    """


class RaiseCatchToGoto(Pass):
    """Lower try/raise pairs to goto.region + goto.label + goto.branch."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._counter = 0

    def verify_preconditions(self, root: dgen.Value) -> None:
        """Body and except result types must be compatible with the try's
        declared type. ``Never`` is universally compatible (a diverging
        branch contributes no value to the merge). Mirrors
        ``ControlFlowToGoto.verify_preconditions`` for ``IfOp``."""
        super().verify_preconditions(root)
        for value in all_values(root):
            if not isinstance(value, error.TryOp):
                continue
            self._check_branch_type(value, value.body.result.type, "body")
            self._check_branch_type(value, value.except_.result.type, "except")

    @staticmethod
    def _check_branch_type(
        op: error.TryOp, branch_type: dgen.Type, branch_name: str
    ) -> None:
        if isinstance(branch_type, Never):
            return
        if branch_type is op.type or type(branch_type) is type(op.type):
            return
        raise TypeError(
            f"TryOp %{op.name}: {branch_name} block result type "
            f"{branch_type} does not match try's declared type {op.type}"
        )

    @lowering_for(error.TryOp)
    def lower_try(self, op: error.TryOp) -> dgen.Value | None:
        cid = self._counter
        self._counter += 1

        handler = op.body.parameters[0]

        # %self is unused for try-merge (no back-edge); %exit carries the
        # merged value via its phi. Mirrors lower_if: body and except
        # both terminate with branch<%exit>(result), and codegen emits
        # the phi at %exit from the two predecessors.
        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"try_exit{cid}", type=goto.Label())

        # except block becomes a goto.label that branches to %exit with
        # its recovery value (or stays as-is if it already diverges).
        redirect_to_exit(op.except_, merge_exit)
        except_label = goto.LabelOp(
            name=f"except{cid}", initial_arguments=pack([]), body=op.except_
        )

        # Substitute handler (effect-layer evidence) with except_label
        # (goto-layer evidence) in the body. The framework's
        # replace_uses_of cascade updates raise.handler operands and
        # nested block capture lists transitively. ``lower_raise`` then
        # converts each raise(label, err) into goto.branch(label, err)
        # when the framework visits it next in topological order.
        op.body.replace_uses_of(handler, except_label)
        op.body.parameters[:] = [merge_self, merge_exit]
        redirect_to_exit(op.body, merge_exit)

        return goto.RegionOp(
            name=f"try{cid}", initial_arguments=pack([]), type=op.type, body=op.body
        )

    @lowering_for(error.RaiseOp)
    def lower_raise(self, op: error.RaiseOp) -> dgen.Value:
        """Convert ``raise(label, err)`` to ``goto.branch(label, err)``.

        The enclosing try (visited first by the framework) replaced the
        handler with its except label, so ``op.handler`` is already a
        ``goto.LabelOp`` by the time we get here. A raise whose handler
        is *not* a label means no enclosing try produced it — the
        effect was never discharged.
        """
        if not isinstance(op.handler, goto.LabelOp):
            raise UndischargedEffectError(
                f"RaiseOp %{op.name}: handler {op.handler!r} does not "
                f"resolve to any enclosing try — the Raise<{op.error_type}> "
                f"effect is not discharged. Every raise must be enclosed "
                f"by a try that binds its handler (v1 design)."
            )
        return goto.BranchOp(target=op.handler, arguments=pack([op.error]))

    def verify_postconditions(self, value: dgen.Value) -> None:
        """No TryOp may survive the pass — every try is lowered to a region.

        Surviving raises are caught earlier (during lowering) by
        ``lower_raise``, which raises ``UndischargedEffectError`` if the
        handler hasn't resolved to a label.
        """
        super().verify_postconditions(value)
        for v in all_values(value):
            if isinstance(v, error.TryOp):
                raise AssertionError(f"TryOp %{v.name} survived RaiseCatchToGoto")
