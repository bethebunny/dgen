"""Lower control_flow dialect to goto dialect.

The body label lives inside the header. The header has two block parameters:
%self (for back-edges) and %exit (codegen emits this as the fall-through
after the header block). No separate exit label needed.

ForOp is replaced by chain(%exit, entry_branch) via rewriter.replace_uses.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import algebra, builtin, control_flow, goto
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.module import ConstantOp, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for


def _pack(values: list[dgen.Value]) -> PackOp:
    if not values:
        return PackOp(values=[], type=builtin.List(element_type=Nil()))
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp, rewriter: Rewriter) -> bool:
        lid = self._loop_counter
        self._loop_counter += 1

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name=f"exit{lid}", type=goto.Label())
        header_iv = BlockArgument(name=f"i{lid}", type=Index())
        body_iv = BlockArgument(name=f"j{lid}", type=Index())

        # --- Body label (inside header): remap IV, back-edge via %self ---
        body_rewriter = Rewriter(op.body)
        body_rewriter.replace_uses(op.body.args[0], body_iv)

        one = ConstantOp(value=1, type=Index())
        next_iv = algebra.AddOp(left=body_iv, right=one, type=Index())
        back_br = goto.BranchOp(target=header_self, arguments=_pack([next_iv]))
        body_result = ChainOp(lhs=op.body.result, rhs=back_br, type=Nil())

        body_block = dgen.Block(
            result=body_result,
            args=[body_iv],
            captures=[header_self] + list(op.body.captures),
        )
        body_label = goto.LabelOp(name=f"loop_body{lid}", body=body_block)

        # --- Header: compare, branch to body or %exit ---
        hi = ConstantOp(value=op.upper_bound.__constant__.to_json(), type=Index())
        cmp = algebra.LessThanOp(left=header_iv, right=hi, type=Index())
        cond_br = goto.ConditionalBranchOp(
            condition=cmp,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=_pack([header_iv]),
            false_arguments=_pack([]),
        )
        header_label = goto.LabelOp(
            name=f"loop_header{lid}",
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self, header_exit],
                args=[header_iv],
            ),
        )

        # --- Entry branch + replacement ---
        lo = ConstantOp(value=op.lower_bound.__constant__.to_json(), type=Index())
        entry_br = goto.BranchOp(target=header_label, arguments=_pack([lo]))

        rewriter.replace_uses(op, entry_br)

        # Recurse into the body label's block to handle nested ForOps.
        self._run_block(body_label.body)

        return True
