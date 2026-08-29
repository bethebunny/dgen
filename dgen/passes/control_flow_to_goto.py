"""Lower control_flow dialect to goto dialect.

Loops (ForOp, WhileOp) and conditionals (IfOp) are lowered to goto regions
and labels.

## Region vs Label

``goto.region`` executes inline in use-def order (fall-through entry). It
emits itself as a basic block, and when unterminated falls through to its
exit label.

``goto.label`` is a pure jump target — only reachable via explicit branch.
It emits as a separate basic block with no fall-through entry.

## Loop iteration contract: concurrent vs sequential

A loop's iterations are SEQUENTIAL iff its carry threads a dataflow/effect
token — a value produced by iteration *n* and consumed by iteration *n+1*
(e.g. a memory effect token). Threading such a token makes the
cross-iteration ordering explicit in the IR: each iteration's memory effects
depend on the previous iteration's via the carried value, so the iterations
cannot be reordered.

A loop with NO such carry has CONCURRENT iterations: there is no
iteration-to-iteration data dependency, so iterations are independent and may
be executed (or scheduled) in any order. This is the natural reading of the
use-def graph — absence of a carried token means absence of ordering.

Loops that mutate shared memory (e.g. dcc's C ``while``/``for``) MUST thread
a memory effect token through the carry to obtain sequential semantics; the
frontend is responsible for establishing this (see dcc's ThreadLoopMemory,
which threads a ``Nil`` effect token as a loop-carried block argument). The
``Nil`` carry has no runtime representation, so codegen erases its phi — the
token exists purely to encode ordering at the IR level.

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

The body result MUST be a tuple shape (``Array`` or ``Tuple``) so its fields
line up with the header's block args; this assertion is load-bearing and is
NOT relaxed. A loop that wants sequential iterations threads its effect token
as one of these carried values — the frontend wraps the outgoing token in a
1-tuple (``pack([token])``) so it satisfies the tuple contract and feeds the
header's carried arg (see the iteration contract above and dcc's
ThreadLoopMemory).

## IfOp lowering

    control_flow.if(%cond, [then_args], [else_args]) then_body(...): ... else_body(...): ...

becomes:

    %if : type = goto.region([]) body<%self: Label, %exit: Label>():
        goto.label([]) if_then() captures(%exit):
            <then body>
            goto.branch<%exit>([then_result])
        goto.label([]) if_else() captures(%exit):
            <else body>
            goto.branch<%exit>([else_result])
        goto.conditional_branch<%if_then, %if_else>(%cond, [], [])

The region's value (``type``) is the phi at ``%exit``, populated from the
two branches' arguments. ``%self`` is unused — there are no back-edges.
The body has no block args; the merge sits at the exit, not at body
entry. Codegen's ``emit_region_op`` emits the body at ``%{name}:`` and
the exit phi at ``%{exit_param.name}:``; the same machinery emits loop
phis at the body block.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import algebra, builtin, control_flow, goto
from dgen.dialects.builtin import Array, ChainOp, Never, Tuple
from dgen.ir.traversal import all_values
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean
from dgen.builtins import ConstantOp, pack
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


def redirect_to_exit(block: dgen.Block, exit_param: BlockParameter) -> None:
    """Wrap *block*'s result in ``branch<%exit>([result])`` in place.

    Skipped when the block already terminates (Never-typed result, e.g.
    a raise that diverges). ``%exit`` is added to the block's captures
    unless it's already a parameter of the block.

    Mutates the block: callers are expected to be discarding the
    enclosing op anyway (lowering replaces it).
    """
    if isinstance(block.result.type, Never):
        return
    branch = goto.BranchOp(target=exit_param, arguments=pack([block.result]))
    block.replace_uses_of(block.result, branch)
    if exit_param not in block.parameters and exit_param not in block.captures:
        block.captures.append(exit_param)


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
        merge_exit: BlockParameter,
    ) -> goto.LabelOp:
        """Build a label for one branch of an IfOp. The label terminates
        with ``branch<%exit>([body.result])`` to feed the region's exit
        phi, except when the body already diverges."""
        redirect_to_exit(body, merge_exit)
        return goto.LabelOp(name=name, initial_arguments=pack([]), body=body)

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp) -> dgen.Value | None:
        lid = self._loop_counter
        self._loop_counter += 1

        # %self is unused for if-merge (no back-edge); %exit carries the
        # merged value via its phi. Region body has no block args — the
        # value lives at the exit, not at body entry.
        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name=f"if_exit{lid}", type=goto.Label())

        # Snapshot the branch bodies' captures before _make_branch_label
        # mutates them — merge_exit gets appended to each branch's
        # captures, but the region body owns it as a parameter so we
        # mustn't propagate it up here.
        then_captures = list(op.then_body.captures)
        else_captures = list(op.else_body.captures)
        then_label = self._make_branch_label(f"if_then{lid}", op.then_body, merge_exit)
        else_label = self._make_branch_label(f"if_else{lid}", op.else_body, merge_exit)

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
                result=cond_br,
                parameters=[merge_self, merge_exit],
                captures=[op.condition, *then_captures, *else_captures],
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
        # If the body already terminates (Never-typed result, e.g. ends in
        # ``continue`` / ``break``), the chain + back-edge are both dead;
        # use the body's result directly.
        iv = op.body.args[0]
        body_result = op.body.result
        if isinstance(body_result.type, Never):
            body_block_result: dgen.Value = body_result
        else:
            next_iv = ChainOp(
                lhs=algebra.AddOp(left=iv, right=Index().constant(1), type=Index()),
                rhs=body_result,
                type=Index(),
            )
            body_block_result = goto.BranchOp(
                target=header_self, arguments=pack([next_iv])
            )
        body_block = dgen.Block(
            result=body_block_result,
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
            type=builtin.Nil(),
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self, header_exit],
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

        # Body result is the next-iteration tuple of carried values, fed
        # back to the header via the back-edge branch. ``body_result.type``
        # MUST be a tuple shape (``Array`` or ``Tuple``) so its fields
        # line up with the header's block args; the IR is malformed
        # otherwise.
        #
        # If the body already terminates (its result is Never-typed — e.g.
        # the body ends in ``continue`` or ``break``), skip the back-edge:
        # control has already transferred and the back-edge would be dead
        # code, plus a duplicate consume of ``%self``/``%exit``.
        body_result = op.body.result
        if isinstance(body_result.type, Never):
            body_block_result: dgen.Value = body_result
        else:
            assert isinstance(body_result.type, (Array, Tuple)), (
                f"control_flow.while body result must be a tuple of carried "
                f"values; got {body_result.type!r}"
            )
            body_block_result = goto.BranchOp(target=header_self, arguments=body_result)

        body_block = dgen.Block(
            result=body_block_result,
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
            type=builtin.Nil(),
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self, header_exit],
                args=header_args,
                captures=list(op.condition.captures) + list(op.body.captures),
            ),
        )

        return header_label
