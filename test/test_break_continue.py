"""Pass-level tests for break/continue lowering through ControlFlowToGoto.

Tests verify that BreakOp/ContinueOp markers are resolved to goto.branch ops
and that the resulting IR structure is correct.

A break/continue references its enclosing loop's body BlockParameter
``%div: control_flow.Loop``. Because ``Loop`` declares ``Handler<Diverge>``
and ``Affine``, the generic divergence detection and linearity machinery
recognize break/continue automatically.
"""

from __future__ import annotations

from dgen.asm.parser import parse
from dgen.dialects import builtin, control_flow
from dgen.dialects.builtin import Diverge, Handler
from dgen.ir.traversal import all_values
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import strip_prefix
from dgen.type import Totality


def _lower(ir_text: str):
    m = parse(strip_prefix(ir_text))
    return Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)


def test_break_in_while_lowers_to_goto(ir_snapshot):
    """WhileOp body with BreakOp produces goto.branch to %exit."""
    result = _lower("""
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body<%div: control_flow.Loop>(%i: index.Index):
        |     %brk : Never = control_flow.break<%div>([])
    """)
    assert result == ir_snapshot
    for v in all_values(result):
        assert not isinstance(v, control_flow.BreakOp), "BreakOp survived lowering"


def test_continue_in_while_lowers_to_goto(ir_snapshot):
    """ContinueOp produces goto.branch to %self."""
    result = _lower("""
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body<%div: control_flow.Loop>(%i: index.Index):
        |     %cont : Never = control_flow.continue<%div>([%i])
    """)
    assert result == ir_snapshot
    for v in all_values(result):
        assert not isinstance(v, control_flow.ContinueOp), (
            "ContinueOp survived lowering"
        )


def test_loop_handler_declares_divergence_traits():
    """``control_flow.Loop`` is the evidence that break/continue have a
    target — it must declare ``Handler<Diverge>`` and ``Affine`` so that
    generic divergence detection and the linearity verifier recognize
    break/continue without bespoke wiring."""
    loop = control_flow.Loop()
    assert loop.has_trait(Handler(effect_type=Diverge()))
    assert loop.has_trait(builtin.Affine())


def test_break_op_is_partial():
    """``break<%div>`` carries a ``Loop`` parameter (Handler<Diverge>
    evidence) so it lights up as PARTIAL under generic divergence
    detection — no per-op wiring."""
    value = parse(
        strip_prefix("""
        | import control_flow
        | import index
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %t : index.Index = 1
        | body<%div: control_flow.Loop>(%i: index.Index):
        |     %brk : Never = control_flow.break<%div>([])
    """)
    )
    brk = next(v for v in all_values(value) if isinstance(v, control_flow.BreakOp))
    assert brk.totality is Totality.PARTIAL


def test_while_without_break_unchanged(ir_snapshot):
    """WhileOp without break/continue lowers identically to before."""
    result = _lower("""
        | import algebra
        | import control_flow
        | import index
        | import number
        | import record
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %one : index.Index = 1
        |     %next : index.Index = algebra.add(%i, %one)
        |     %carry : Tuple<[index.Index]> = record.pack([%next])
    """)
    assert result == ir_snapshot
    for v in all_values(result):
        assert not isinstance(v, control_flow.WhileOp), "WhileOp survived lowering"
