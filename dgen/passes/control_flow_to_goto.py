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

    goto.region([]) if<%self>(%result: type):
        goto.label([then_args]) if_then(body_args):
            <then body>
            goto.branch<%self>([then_result])
        goto.label([else_args]) if_else(body_args):
            <else body>
            goto.branch<%self>([else_result])
        goto.conditional_branch<%if_then, %if_else>(%cond, [then_args], [else_args])
        chain(%result, cond_br)

Same structure as loops: ``%self`` is the merge target. The initial entry
dispatches via cond_br; re-entries from then/else hit the phi for ``%result``.
Codegen's ``emit_region_op`` splits this into an entry block (dispatch) and
a merge block (phi) when block args exist without initial_arguments.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import algebra, control_flow, goto
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean
from dgen.module import ConstantOp, PackOp, pack
from dgen.passes.pass_ import Pass, lowering_for


class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0

    def verify_preconditions(self, module: dgen.module.Module) -> None:
        super().verify_preconditions(module)
        from dgen.graph import all_values

        for top_level in module.ops:
            for value in all_values(top_level):
                if not isinstance(value, control_flow.IfOp):
                    continue
                then_type = value.then_body.result.type
                else_type = value.else_body.result.type
                if then_type is not value.type and type(then_type) is not type(
                    value.type
                ):
                    raise TypeError(
                        f"IfOp then-branch result type {then_type} "
                        f"does not match declared type {value.type}"
                    )
                if else_type is not value.type and type(else_type) is not type(
                    value.type
                ):
                    raise TypeError(
                        f"IfOp else-branch result type {else_type} "
                        f"does not match declared type {value.type}"
                    )

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"if_exit{lid}", type=goto.Label())
        merge_result = BlockArgument(name=f"if_result{lid}", type=op.type)

        # Save original captures before mutation.
        then_orig_captures = list(op.then_body.captures)
        else_orig_captures = list(op.else_body.captures)

        # Both branches carry their body result to %self.
        # Reuse original blocks in-place.
        then_br = goto.BranchOp(
            target=merge_self, arguments=pack([op.then_body.result])
        )
        op.then_body.captures = [merge_self, *then_orig_captures]
        op.then_body.result = then_br

        then_label = goto.LabelOp(
            name=f"if_then{lid}",
            initial_arguments=pack([]),
            body=op.then_body,
        )

        else_br = goto.BranchOp(
            target=merge_self, arguments=pack([op.else_body.result])
        )
        op.else_body.captures = [merge_self, *else_orig_captures]
        op.else_body.result = else_br

        else_label = goto.LabelOp(
            name=f"if_else{lid}",
            initial_arguments=pack([]),
            body=op.else_body,
        )

        cond_br = goto.ConditionalBranchOp(
            condition=op.condition,
            true_target=then_label,
            false_target=else_label,
            true_arguments=pack([]),
            false_arguments=pack([]),
        )

        region = goto.RegionOp(
            name=f"if{lid}",
            initial_arguments=pack([]),
            type=op.type,
            body=dgen.Block(
                result=ChainOp(lhs=merge_result, rhs=cond_br, type=op.type),
                parameters=[merge_self, merge_exit],
                args=[merge_result],
                captures=[
                    op.condition,
                    *then_orig_captures,
                    *else_orig_captures,
                ],
            ),
        )

        return region

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name=f"exit{lid}", type=goto.Label())
        header_iv = BlockArgument(name=f"i{lid}", type=Index())

        # --- Body label (inside header): reuse op.body, add back-edge ---
        # Reuse the original block arg as the body IV. No mutation needed.
        iv = op.body.args[0]
        original_captures = list(op.body.captures)

        # The increment must depend on the inner loop body having run.
        # chain(increment, body_result) ensures the increment comes after.
        one = ConstantOp(value=1, type=Index())
        next_iv_raw = algebra.AddOp(left=iv, right=one, type=Index())
        next_iv = ChainOp(lhs=next_iv_raw, rhs=op.body.result, type=Index())
        back_br = goto.BranchOp(target=header_self, arguments=pack([next_iv]))

        # Modify op.body in-place: add header_self to captures, set new result.
        # This keeps all new ops (%next, back_br) in the same scope as the
        # original body ops, so nested lowering replacements reach them.
        op.body.captures = [header_self] + original_captures
        op.body.result = back_br

        body_label = goto.LabelOp(
            name=f"loop_body{lid}",
            initial_arguments=pack([]),
            body=op.body,
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
                captures=original_captures,
            ),
        )

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
        for orig, new in zip(op.body.args, body_args):
            op.body.replace_uses_of(orig, new)

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
        for orig, new in zip(op.condition.args, header_args):
            op.condition.replace_uses_of(orig, new)

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

        return header_label
