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
        # merged value via its phi. Mirrors lower_if: the body terminates
        # with branch<%exit>(body.result), the except label terminates
        # with branch<%exit>(except.result), and codegen emits the phi at
        # %exit from the two predecessors.
        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"try_exit{cid}", type=goto.Label())

        # except block becomes a goto.label whose body branches to %exit
        # with its recovery value. A Never-typed except (e.g. re-raise)
        # already terminates on its own — don't add a merge branch.
        except_block = op.except_
        if isinstance(except_block.result.type, Never):
            except_result: dgen.Value = except_block.result
            except_captures = list(except_block.captures)
        else:
            except_result = goto.BranchOp(
                target=merge_exit, arguments=pack([except_block.result])
            )
            except_captures = [merge_exit, *except_block.captures]
        except_label = goto.LabelOp(
            name=f"except{cid}",
            initial_arguments=pack([]),
            body=dgen.Block(
                args=list(except_block.args),
                captures=except_captures,
                result=except_result,
            ),
        )

        # Substitute handler (effect-layer evidence) with except_label
        # (goto-layer evidence). The framework's replace_uses_of cascade
        # updates raise.handler operands and nested block capture lists
        # transitively. ``lower_raise`` then converts each
        # raise(label, err) into goto.branch(label, err) when the framework
        # visits it next in topological order.
        op.body.replace_uses_of(handler, except_label)

        # Body terminator: branch<%exit>(body.result), or the body's own
        # result if it diverges (every path raises). %merge_exit is a
        # parameter of the region body, so the branch references it
        # directly — no capture needed.
        if isinstance(op.body.result.type, Never):
            body_terminator: dgen.Value = op.body.result
        else:
            body_terminator = goto.BranchOp(
                target=merge_exit, arguments=pack([op.body.result])
            )

        return goto.RegionOp(
            name=f"try{cid}",
            initial_arguments=pack([]),
            type=op.type,
            body=dgen.Block(
                parameters=[merge_self, merge_exit],
                captures=list(op.body.captures),
                result=body_terminator,
            ),
        )

    @lowering_for(error.RaiseOp)
    def lower_raise(self, op: error.RaiseOp) -> dgen.Value | None:
        """Convert ``raise(label, err)`` to ``goto.branch(label, err)``.

        The enclosing try (visited first by the framework) replaced the
        handler with its except label, so ``op.handler`` is already a
        ``goto.LabelOp`` by the time we get here. A raise whose handler
        is *not* a label means no enclosing try produced it — that's an
        undischarged effect, caught by ``verify_postconditions``.
        """
        if not isinstance(op.handler, goto.LabelOp):
            return None
        return goto.BranchOp(target=op.handler, arguments=pack([op.error]))

    def verify_postconditions(self, value: dgen.Value) -> None:
        """No TryOp or RaiseOp may survive the pass.

        A surviving ``RaiseOp`` means its handler didn't resolve to any
        enclosing try — the effect was never discharged. Reporting this
        here keeps codegen from silently emitting undefined behavior.
        """
        super().verify_postconditions(value)
        for v in all_values(value):
            if isinstance(v, error.TryOp):
                raise AssertionError(f"TryOp %{v.name} survived RaiseCatchToGoto")
            if isinstance(v, error.RaiseOp):
                raise UndischargedEffectError(
                    f"RaiseOp %{v.name}: handler {v.handler!r} does not "
                    f"resolve to any enclosing try — the Raise<{v.error_type}> "
                    f"effect is not discharged. Every raise must be enclosed "
                    f"by a try that binds its handler (v1 design)."
                )
