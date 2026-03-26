"""Lower control_flow dialect to goto dialect (flat labels).

Converts structured ForOps into flat goto labels, arg projections, and branches.
Labels are zero-dependency source nodes in the use-def graph. The label's "body"
is the subgraph of ops that transitively depend on it via ArgOp projections.

No block splitting, no value_map, no cloning. The ForOp is replaced by its
exit label, and post-loop ops naturally depend on it.
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, builtin, goto, index
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


def _pack(values: list[dgen.Value]) -> PackOp:
    if not values:
        return PackOp(values=[], type=builtin.List(element_type=Nil()))
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0
        self._depth = 0

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        """Run until no ForOps remain (handles nested loops)."""
        from dgen.dialects import control_flow

        while True:
            for func in module.functions:
                self._run_block(func.body)
            if not any(
                isinstance(op, control_flow.ForOp)
                for func in module.functions
                for op in func.body.ops
            ):
                break
        return module

    from dgen.dialects import control_flow

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp, rewriter: Rewriter) -> bool:
        lid = self._loop_counter
        self._loop_counter += 1

        # --- Create labels (zero-dependency source nodes) ---
        # Annotate with nesting depth for codegen scheduling.
        d = self._depth
        header = goto.LabelOp(name=f"loop_header{lid}")
        header._depth = d  # type: ignore[attr-defined]
        body_label = goto.LabelOp(name=f"loop_body{lid}")
        body_label._depth = d + 1  # type: ignore[attr-defined]
        exit_label = goto.LabelOp(name=f"loop_exit{lid}")
        exit_label._depth = d  # type: ignore[attr-defined]

        # --- Header: project IV, compare, branch ---
        header_iv = goto.ArgOp(name=f"i{lid}", label=header, type=Index())
        hi = ConstantOp(value=op.upper_bound.__constant__.to_json(), type=Index())
        cmp = algebra.LessThanOp(left=header_iv, right=hi, type=Index())
        cond_br = goto.ConditionalBranchOp(
            condition=cmp,
            true_target=body_label,
            false_target=exit_label,
            true_arguments=_pack([header_iv]),
            false_arguments=_pack([]),
        )

        # --- Body: project IV, run body ops, increment, back-edge ---
        body_iv = goto.ArgOp(name=f"j{lid}", label=body_label, type=Index())

        # Remap the ForOp body's loop IV to the body_iv ArgOp.
        body_rewriter = Rewriter(op.body)
        body_rewriter.replace_uses(op.body.args[0], body_iv)

        # Increment depth for nested ForOps processed in subsequent iterations.
        self._depth = d + 2  # body is d+1, so nested header starts at d+2

        # Append back-edge: increment IV, branch to header.
        one = ConstantOp(value=1, type=Index())
        next_iv = algebra.AddOp(left=body_iv, right=one, type=Index())
        back_br = goto.BranchOp(target=header, arguments=_pack([next_iv]))

        # The body's result (a store or chain of stores) needs to be chained
        # with the back-edge branch to keep both alive.
        body_terminal = ChainOp(lhs=op.body.result, rhs=back_br, type=Nil())

        # --- Entry: branch to header with initial IV ---
        lo = ConstantOp(value=op.lower_bound.__constant__.to_json(), type=Index())
        entry_br = goto.BranchOp(target=header, arguments=_pack([lo]))

        # --- Wire up ---
        # The ForOp produced Nil. chain(cond_br, body_terminal) keeps the
        # header and body alive. chain that with entry_br to keep the entry
        # branch alive. The exit_label replaces the ForOp — post-loop ops
        # that depended on the ForOp now depend on exit_label.
        alive = ChainOp(lhs=cond_br, rhs=body_terminal, type=Nil())
        alive = ChainOp(lhs=alive, rhs=entry_br, type=Nil())

        # Replace ForOp with exit_label. Post-loop code (e.g. print) that
        # depended on the ForOp via chain(alloc, for_op) now depends on
        # chain(alloc, exit_label). The exit label's "body" is the subgraph
        # of ops that depend on it.
        # But we also need the loop structure to be alive — chain it.
        replacement = ChainOp(lhs=exit_label, rhs=alive, type=Nil())
        rewriter.replace_uses(op, replacement)
        return True
