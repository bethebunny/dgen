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
from dgen.dialects.builtin import ChainOp, Never
from dgen.ir.traversal import all_values
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean
from dgen.builtins import ConstantOp, PackOp, pack
from dgen.passes.pass_ import Pass, lowering_for


def _resolve_jump_markers(
    block: dgen.Block,
    self_param: BlockParameter,
    exit_param: BlockParameter,
) -> set[BlockParameter]:
    """Replace BreakOp/ContinueOp with goto.BranchOp targeting exit/self.

    Recurses into child blocks but skips nested WhileOp/ForOp bodies —
    inner loops resolve their own markers when they are lowered.
    Returns the set of parameters that needed to be captured.
    """
    needed: set[BlockParameter] = set()
    for v in block.values:
        if isinstance(v, control_flow.BreakOp):
            block.replace_uses_of(
                v, goto.BranchOp(target=exit_param, arguments=pack([]))
            )
            needed.add(exit_param)
        elif isinstance(v, control_flow.ContinueOp):
            block.replace_uses_of(
                v, goto.BranchOp(target=self_param, arguments=pack([]))
            )
            needed.add(self_param)
        elif not isinstance(v, (control_flow.WhileOp, control_flow.ForOp)):
            for _, child_block in v.blocks:
                child_needed = _resolve_jump_markers(
                    child_block, self_param, exit_param
                )
                for param in child_needed:
                    if param not in child_block.captures:
                        child_block.captures.append(param)
                needed |= child_needed
    return needed


class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0

    def verify_preconditions(self, root: dgen.Value) -> None:
        super().verify_preconditions(root)
        for value in all_values(root):
            if not isinstance(value, control_flow.IfOp):
                continue
            then_type = value.then_body.result.type
            else_type = value.else_body.result.type
            if isinstance(then_type, Never) or isinstance(else_type, Never):
                continue
            if then_type is not value.type and type(then_type) is not type(value.type):
                raise TypeError(
                    f"IfOp then-branch result type {then_type} "
                    f"does not match declared type {value.type}"
                )
            if else_type is not value.type and type(else_type) is not type(value.type):
                raise TypeError(
                    f"IfOp else-branch result type {else_type} "
                    f"does not match declared type {value.type}"
                )

    @staticmethod
    def _make_branch_label(
        name: str,
        body: dgen.Block,
        merge_self: BlockParameter,
    ) -> goto.LabelOp:
        """Build a label for one branch of an IfOp.

        If the body diverges (result type is Never), the body already
        terminates elsewhere — don't add a merge branch. Otherwise,
        append a branch to the merge point.
        """
        if isinstance(body.result.type, Never):
            result = body.result
            captures = list(body.captures)
        else:
            result = goto.BranchOp(
                target=merge_self, arguments=pack([body.result])
            )
            captures = [merge_self, *body.captures]
        return goto.LabelOp(
            name=name,
            initial_arguments=pack([]),
            body=dgen.Block(result=result, captures=captures),
        )

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"if_exit{lid}", type=goto.Label())
        merge_result = BlockArgument(name=f"if_result{lid}", type=op.type)

        then_label = self._make_branch_label(f"if_then{lid}", op.then_body, merge_self)
        else_label = self._make_branch_label(f"if_else{lid}", op.else_body, merge_self)

        cond_br = goto.ConditionalBranchOp(
            condition=op.condition,
            true_target=then_label,
            false_target=else_label,
            true_arguments=pack([]),
            false_arguments=pack([]),
        )

        return goto.RegionOp(
            name=f"if{lid}",
            initial_arguments=pack([]),
            type=op.type,
            body=dgen.Block(
                result=ChainOp(lhs=merge_result, rhs=cond_br, type=op.type),
                params=[merge_self, merge_exit],
                args=[merge_result],
                captures=[op.condition, *op.then_body.captures, *op.else_body.captures],
            ),
        )

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name=f"exit{lid}", type=goto.Label())
        header_iv = BlockArgument(name=f"i{lid}", type=Index())

        # Body label: reuse op.body's IV, chain(increment, body_result) as
        # the back-edge arg so the increment happens after the body runs.
        iv = op.body.args[0]
        next_iv = ChainOp(
            lhs=algebra.AddOp(left=iv, right=Index().constant(1), type=Index()),
            rhs=op.body.result,
            type=Index(),
        )
        body_block = dgen.Block(
            result=goto.BranchOp(target=header_self, arguments=pack([next_iv])),
            args=[iv],
            captures=[header_self, header_exit, *op.body.captures],
        )
        body_label = goto.LabelOp(
            name=f"loop_body{lid}",
            initial_arguments=pack([]),
            body=body_block,
        )

        _resolve_jump_markers(body_block, header_self, header_exit)

        # Header: compare, branch to body or %exit.
        hi = Index().constant(op.upper_bound.__constant__.to_json())
        cmp = algebra.LessThanOp(left=header_iv, right=hi, type=Boolean())
        cond_br = goto.ConditionalBranchOp(
            condition=cmp,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=pack([header_iv]),
            false_arguments=pack([]),
        )
        lo = ConstantOp.from_constant(
            Index().constant(op.lower_bound.__constant__.to_json())
        )
        return goto.RegionOp(
            name=f"loop_header{lid}",
            initial_arguments=pack([lo]),
            body=dgen.Block(
                result=cond_br,
                params=[header_self, header_exit],
                args=[header_iv],
                captures=list(op.body.captures),
            ),
        )

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
        branch_args = (
            body_result if isinstance(body_result, PackOp) else pack([body_result])
        )
        back_br = goto.BranchOp(target=header_self, arguments=branch_args)

        body_block = dgen.Block(
            result=back_br,
            args=body_args,
            captures=[header_self, header_exit, *op.body.captures],
        )
        body_label = goto.LabelOp(
            name=f"while_body{lid}",
            initial_arguments=pack([]),
            body=body_block,
        )

        _resolve_jump_markers(body_block, header_self, header_exit)

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
                params=[header_self, header_exit],
                args=header_args,
                captures=list(op.condition.captures) + list(op.body.captures),
            ),
        )

        return header_label
