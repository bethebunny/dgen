"""Lower control_flow dialect to goto dialect.

Loops (ForOp, WhileOp) and conditionals (IfOp) are lowered to goto regions
and labels.

## Region vs Label

``goto.region`` executes inline in use-def order (fall-through entry). It
emits itself as a basic block, and when unterminated falls through to its
exit label.

``goto.label`` is a pure jump target — only reachable via explicit branch.
It emits as a separate basic block with no fall-through entry.

## ForOp lowering

    control_flow.for<lo, hi>([init]) body(%iv):
        <body ops>

becomes:

    goto.region([lo]) body<%self, %exit>(%iv):
        %cmp = less_than(%iv, hi)
        goto.label([]) body(%jv) captures(%self):
            <body ops, iv remapped to jv>
            %next = chain(add(%jv, 1), <body result>)
            goto.branch<%self>([%next])
        goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])

Key points:
- The header is a ``region`` (falls through from use-def position)
- The body is a ``label`` (only entered via conditional_branch)
- `%self` parameter enables back-edges (breaks use-def cycles)
- `%exit` parameter: codegen emits this as a fall-through label after the header
- `chain(increment, body_result)` ensures the increment runs AFTER the body.
  This is necessary because `add(%jv, 1)` doesn't naturally depend on the body
  result — without the chain, the increment could be scheduled before inner loops.

## WhileOp lowering

Similar structure but simpler: the condition and body are user-provided blocks.
No explicit chain is needed for the body because the body result IS the
next-iteration values — the back-edge branch arguments reference them
transitively.

## IfOp lowering

    control_flow.if(%cond, [then_args], [else_args]) then_body(...): ... else_body(...): ...

becomes:

    goto.region([]) if_dispatch<%merge>():
        goto.label([then_args]) if_then(body_args):
            <then body>
            goto.branch<%merge>([then_result])
        goto.label([else_args]) if_else(body_args):
            <else body>
            goto.branch<%merge>([else_result])
        goto.conditional_branch<%if_then, %if_else>(%cond, [then_args], [else_args])

For value-producing ifs, a merge region follows the dispatch:

    goto.region([]) if_merge(%result: type):
        chain(%result, dispatch)

The dispatch region uses an exit parameter ``%merge`` — codegen emits it as a
fall-through label after the dispatch body. Then/else branches carry their
results to this label, which becomes a phi node.

For void ifs (type=Nil), the dispatch region is returned directly.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import algebra, control_flow, goto
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean
from dgen.module import ConstantOp, PackOp, pack
from dgen.passes.pass_ import Pass, Rewriter, lowering_for


class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp) -> dgen.Value | None:
        from dgen.dialects.builtin import Nil

        lid = self._loop_counter
        self._loop_counter += 1

        is_void = isinstance(op.type, Nil)

        # --- Merge region: receives results from then/else as block args ---
        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_result_arg = (
            None if is_void else BlockArgument(name=f"if_result{lid}", type=op.type)
        )

        # --- Then label: runs then body, branches to merge ---
        then_result_args: list[dgen.Value] = (
            []
            if is_void
            else (
                list(op.then_body.result)
                if isinstance(op.then_body.result, PackOp)
                else [op.then_body.result]
            )
        )
        then_br = goto.BranchOp(
            target=merge_self,
            arguments=pack(then_result_args),
        )
        then_body = dgen.Block(
            result=then_br,
            args=list(op.then_body.args),
            captures=[merge_self] + list(op.then_body.captures),
        )
        then_label = goto.LabelOp(
            name=f"if_then{lid}",
            initial_arguments=op.then_arguments,
            body=then_body,
        )

        # --- Else label: runs else body, branches to merge ---
        else_result_args: list[dgen.Value] = (
            []
            if is_void
            else (
                list(op.else_body.result)
                if isinstance(op.else_body.result, PackOp)
                else [op.else_body.result]
            )
        )
        else_br = goto.BranchOp(
            target=merge_self,
            arguments=pack(else_result_args),
        )
        else_body = dgen.Block(
            result=else_br,
            args=list(op.else_body.args),
            captures=[merge_self] + list(op.else_body.captures),
        )
        else_label = goto.LabelOp(
            name=f"if_else{lid}",
            initial_arguments=op.else_arguments,
            body=else_body,
        )

        # --- Conditional dispatch ---
        cond_br = goto.ConditionalBranchOp(
            condition=op.condition,
            true_target=then_label,
            false_target=else_label,
            true_arguments=op.then_arguments,
            false_arguments=op.else_arguments,
        )

        # --- Single region containing everything ---
        merge_captures: list[dgen.Value] = [
            op.condition,
            op.then_arguments,
            op.else_arguments,
        ]
        merge_captures.extend(op.then_body.captures)
        merge_captures.extend(op.else_body.captures)
        # Deduplicate while preserving order
        seen: set[int] = set()
        unique_captures: list[dgen.Value] = []
        for cap in merge_captures:
            if id(cap) not in seen:
                seen.add(id(cap))
                unique_captures.append(cap)

        merge_args = [] if is_void else [merge_result_arg]

        if is_void:
            body_result = cond_br
        else:
            body_result = ChainOp(lhs=merge_result_arg, rhs=cond_br, type=op.type)

        merge = goto.RegionOp(
            name=f"if{lid}",
            initial_arguments=pack([]),
            type=op.type,
            body=dgen.Block(
                result=body_result,
                parameters=[merge_self],
                args=merge_args,
                captures=unique_captures,
            ),
        )

        # Recurse into then/else bodies to handle nested control flow.
        self._run_block(then_label.body)
        self._run_block(else_label.body)

        return merge

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name=f"exit{lid}", type=goto.Label())
        header_iv = BlockArgument(name=f"i{lid}", type=Index())
        body_iv = BlockArgument(name=f"j{lid}", type=Index())

        # --- Body label (inside header): remap IV, back-edge via %self ---
        body_rewriter = Rewriter(op.body)
        body_rewriter.replace_uses(op.body.args[0], body_iv)

        # The increment must depend on the inner loop body having run.
        # op.body.result will become the inner header label after replace_uses,
        # so chain(increment, body_result) ensures the increment comes after.
        one = ConstantOp(value=1, type=Index())
        next_iv_raw = algebra.AddOp(left=body_iv, right=one, type=Index())
        next_iv = ChainOp(lhs=next_iv_raw, rhs=op.body.result, type=Index())
        back_br = goto.BranchOp(target=header_self, arguments=pack([next_iv]))

        body_block = dgen.Block(
            result=back_br,
            args=[body_iv],
            captures=[header_self] + list(op.body.captures),
        )
        body_label = goto.LabelOp(
            name=f"loop_body{lid}",
            initial_arguments=pack([]),
            body=body_block,
        )

        # --- Header: compare, branch to body or %exit ---
        hi = ConstantOp(value=op.upper_bound.__constant__.to_json(), type=Index())
        cmp = algebra.LessThanOp(left=header_iv, right=hi, type=Boolean())
        cond_br = goto.ConditionalBranchOp(
            condition=cmp,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=pack([header_iv]),
            false_arguments=pack([]),
        )

        lo = ConstantOp(value=op.lower_bound.__constant__.to_json(), type=Index())
        header_label = goto.RegionOp(
            name=f"loop_header{lid}",
            initial_arguments=pack([lo]),
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self, header_exit],
                args=[header_iv],
                captures=list(op.body.captures),
            ),
        )

        # Recurse into the body label's block to handle nested ForOps.
        self._run_block(body_label.body)

        # Replace ForOp with the header label. The label is an expression
        # block — it runs when control reaches it. No entry branch needed.
        return header_label

    @lowering_for(control_flow.WhileOp)
    def lower_while(self, op: control_flow.WhileOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        # Block args for header and body, one per loop-carried variable.
        header_args = [
            BlockArgument(name=f"wh{lid}_{a.name}", type=a.type)
            for a in op.condition.args
        ]
        body_args = [
            BlockArgument(name=f"wb{lid}_{a.name}", type=a.type) for a in op.body.args
        ]

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name=f"exit{lid}", type=goto.Label())

        # --- Body label: remap body block args, append back-edge ---
        body_rewriter = Rewriter(op.body)
        for orig, new in zip(op.body.args, body_args):
            body_rewriter.replace_uses(orig, new)

        # Body result is the next-iteration values. Wrap in a pack for the
        # back-edge branch arguments. The branch already depends on body_result
        # transitively via the arguments operand.
        body_result = op.body.result
        next_args: list[dgen.Value] = (
            list(body_result) if isinstance(body_result, PackOp) else [body_result]
        )
        back_br = goto.BranchOp(target=header_self, arguments=pack(next_args))

        body_block = dgen.Block(
            result=back_br,
            args=body_args,
            captures=[header_self] + list(op.body.captures),
        )
        body_label = goto.LabelOp(
            name=f"while_body{lid}",
            initial_arguments=pack([]),
            body=body_block,
        )

        # --- Header: remap condition block args, append conditional branch ---
        cond_rewriter = Rewriter(op.condition)
        for orig, new in zip(op.condition.args, header_args):
            cond_rewriter.replace_uses(orig, new)

        cond_result = op.condition.result
        cond_br = goto.ConditionalBranchOp(
            condition=cond_result,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=pack(header_args),
            false_arguments=pack([]),
        )

        header_label = goto.RegionOp(
            name=f"while_header{lid}",
            initial_arguments=op.initial_arguments,
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self, header_exit],
                args=header_args,
                captures=list(op.condition.captures) + list(op.body.captures),
            ),
        )

        # Recurse into body to handle nested loops.
        self._run_block(body_label.body)

        return header_label
