"""Pass-level tests for break/continue lowering through ControlFlowToGoto.

Tests verify that BreakOp/ContinueOp markers are resolved to goto.branch ops
and that the resulting IR structure is correct.
"""

from __future__ import annotations

from dgen.dialects import control_flow
from dgen.ir.traversal import all_values
from dgen.passes.control_flow_to_goto import ControlFlowToGoto


def test_break_in_while_lowers_to_goto(lowering_snapshot):
    """WhileOp body with BreakOp produces goto.branch to %exit."""
    result = lowering_snapshot(
        [ControlFlowToGoto()],
        """
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %brk : Never = control_flow.break()
        """,
    )
    for v in all_values(result):
        assert not isinstance(v, control_flow.BreakOp), "BreakOp survived lowering"


def test_continue_in_while_lowers_to_goto(lowering_snapshot):
    """ContinueOp produces goto.branch to %self."""
    result = lowering_snapshot(
        [ControlFlowToGoto()],
        """
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %cont : Never = control_flow.continue()
        """,
    )
    for v in all_values(result):
        assert not isinstance(v, control_flow.ContinueOp), (
            "ContinueOp survived lowering"
        )


def test_while_without_break_unchanged(lowering_snapshot):
    """WhileOp without break/continue lowers identically to before."""
    result = lowering_snapshot(
        [ControlFlowToGoto()],
        """
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %one : index.Index = 1
        |     %next : index.Index = algebra.add(%i, %one)
        """,
    )
    for v in all_values(result):
        assert not isinstance(v, control_flow.WhileOp), "WhileOp survived lowering"
