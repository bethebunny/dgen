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
- `chain(result=increment, effect=body_result)` keeps `body_result` reachable
  from the block result and makes the back-edge branch (which consumes `%next`)
  depend on it, so the body's effects stay live and are sequenced before the
  back-edge. The chain does NOT order the increment relative to the body — the
  increment depends only on `%jv` (`add(%jv, 1)`) and is free to run in any order
  with respect to the body.

Extra loop carries (beyond the IV) are supported via ``op.initial_arguments``.
The convention is ``op.body.args == [iv, *carries]`` and
``op.initial_arguments == pack([*carry_inits])`` (the IV init is
``lower_bound``, not part of ``initial_arguments``). The header and body each
gain a block arg per carry, the back-edge threads the next carry values
(``op.body.result``, or its ``pack`` elements for multiple carries), and the
region's ``initial_arguments`` becomes ``[lo, *carry_inits]``. With carries
present the increment does NOT need a ``chain``: the back-edge already depends
on the next carry values, which carry the body's effects via dataflow. The
loop op result stays ``Nil`` — exit phis for carries are not materialized
(consumers read carried state from memory after the loop).

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
from dgen.builtins import ConstantOp, pack, unpack
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import types_equivalent


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


def _verify_if_types(op: control_flow.IfOp) -> None:
    then_type = op.then_body.result.type
    else_type = op.else_body.result.type
    if isinstance(then_type, Never) or isinstance(else_type, Never):
        return
    if then_type is not op.type and type(then_type) is not type(op.type):
        raise TypeError(
            f"IfOp then-branch result type {then_type} "
            f"does not match declared type {op.type}"
        )
    if else_type is not op.type and type(else_type) is not type(op.type):
        raise TypeError(
            f"IfOp else-branch result type {else_type} "
            f"does not match declared type {op.type}"
        )


def _check_carry_types(
    kind: str,
    carries: list[BlockArgument],
    label: str,
    types: list[dgen.Value],
) -> None:
    """One carry-consistency check: ``types`` (the element types of the
    loop's ``initial_arguments`` / condition args / next-iteration values)
    must line up 1:1 with ``carries`` in count and type."""
    if len(types) != len(carries):
        raise TypeError(
            f"{kind} carry arity: {len(carries)} carries but {len(types)} {label}"
        )
    for carry, expected in zip(carries, types):
        carry_type = carry.type
        # Unresolved staged type values can't be compared yet; staging
        # resolves them before the pass pipeline runs.
        if not isinstance(carry_type, dgen.Type) or not isinstance(expected, dgen.Type):
            continue
        if not types_equivalent(carry_type, expected):
            raise TypeError(
                f"{kind} carry {carry.name!r}: type {carry_type} does not "
                f"match {label} type {expected}"
            )


def _element_types(tuple_type: dgen.Value, count: int) -> list[dgen.Value]:
    """Element types of a tuple-shaped type holding ``count`` values.

    Works at the type level so it handles any tuple-shaped *value*
    (``builtin.pack``, ``record.pack``, an aggregate Constant, ...) —
    unlike ``unpack``, which only decomposes ``builtin.PackOp`` values.
    """
    if isinstance(tuple_type, Tuple):
        return unpack(tuple_type.types)
    if isinstance(tuple_type, Array):
        return [tuple_type.element_type] * count
    raise TypeError(f"expected a tuple-shaped type; got {tuple_type}")


def _verify_while_carries(op: control_flow.WhileOp) -> None:
    """A WhileOp's loop-carried values must agree in type across its
    ``initial_arguments``, condition/body block args, and the body
    result (a tuple of next-iteration values). Without this, a carry
    whose next value mismatches its declared type lowers to an invalid
    back-edge phi (e.g. a Nil next value for an i64 carry)."""
    carries = op.body.args
    if not carries:
        return  # zero-carry: body-result shape is checked at lowering
    if isinstance(op.body.result.type, Never):
        return  # body diverges (break/continue): no back-edge
    if not isinstance(op.body.result.type, (Array, Tuple)):
        raise TypeError(
            f"WhileOp with carries must have a tuple body result; got "
            f"{op.body.result.type}"
        )
    _check_carry_types(
        "WhileOp",
        carries,
        "initial_arguments",
        [v.type for v in unpack(op.initial_arguments)],
    )
    _check_carry_types(
        "WhileOp", carries, "condition arg", [a.type for a in op.condition.args]
    )
    _check_carry_types(
        "WhileOp",
        carries,
        "body result",
        _element_types(op.body.result.type, len(carries)),
    )


def _verify_for_carries(op: control_flow.ForOp) -> None:
    """A ForOp's carries (block args beyond the induction variable) must
    agree in type with their ``initial_arguments`` and next-iteration
    values. ``body.result`` is the single next value for one carry, or a
    tuple for several (matching ``lower_for``)."""
    carries = op.body.args[1:]
    if not carries:
        return
    if isinstance(op.body.result.type, Never):
        return
    next_types = (
        [op.body.result.type]
        if len(carries) == 1
        else _element_types(op.body.result.type, len(carries))
    )
    _check_carry_types(
        "ForOp",
        carries,
        "initial_arguments",
        [v.type for v in unpack(op.initial_arguments)],
    )
    _check_carry_types("ForOp", carries, "body result", next_types)


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


class ControlFlowToGoto(Pass):
    """Lower control_flow loops and conditionals to goto regions/labels.

    Emitted names ("loop_header", "exit", ...) are readability prefixes,
    not identifiers: value identity carries the IR semantics, and the
    naming layers (codegen's tracker, the ASM formatter) uniquify
    duplicates on demand.
    """

    allow_unregistered_ops = True

    def verify_preconditions(self, root: dgen.Value) -> None:
        super().verify_preconditions(root)
        for value in all_values(root):
            if isinstance(value, control_flow.IfOp):
                _verify_if_types(value)
            elif isinstance(value, control_flow.WhileOp):
                _verify_while_carries(value)
            elif isinstance(value, control_flow.ForOp):
                _verify_for_carries(value)

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp) -> dgen.Value | None:
        # %self is unused for if-merge (no back-edge); %exit carries the
        # merged value via its phi. Region body has no block args — the
        # value lives at the exit, not at body entry.
        merge_self = BlockParameter(name="self", type=goto.Label())
        merge_exit = BlockParameter(name="if_exit", type=goto.Label())

        # Snapshot the branch bodies' captures before _make_branch_label
        # mutates them — merge_exit gets appended to each branch's
        # captures, but the region body owns it as a parameter so we
        # mustn't propagate it up here.
        then_captures = list(op.then_body.captures)
        else_captures = list(op.else_body.captures)
        then_label = _make_branch_label("if_then", op.then_body, merge_exit)
        else_label = _make_branch_label("if_else", op.else_body, merge_exit)

        conditional_branch = goto.ConditionalBranchOp(
            condition=op.condition,
            true_target=then_label,
            false_target=else_label,
            true_arguments=pack([]),
            false_arguments=pack([]),
        )

        return goto.RegionOp(
            name="if",
            initial_arguments=pack([]),
            type=op.type,
            body=dgen.Block(
                result=conditional_branch,
                parameters=[merge_self, merge_exit],
                captures=[op.condition, *then_captures, *else_captures],
            ),
        )

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp) -> dgen.Value | None:
        # ``op.body.args == [iv, *carries]``; the IV's init is
        # ``lower_bound``, not part of ``initial_arguments``.
        iv = op.body.args[0]
        carries = op.body.args[1:]

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name="exit", type=goto.Label())
        header_iv = BlockArgument(name="i", type=Index())
        header_carries = [
            BlockArgument(name=carry.name, type=carry.type) for carry in carries
        ]

        body_result = op.body.result
        if isinstance(body_result.type, Never):
            # Body already terminates (e.g. ends in ``continue`` / ``break``);
            # the back-edge is dead. Use the body's result directly.
            body_block_result: dgen.Value = body_result
        elif not carries:
            # No extra carries: chain(increment, body_result) as the back-edge
            # arg so the increment happens after the body runs. ``add(%iv, 1)``
            # doesn't naturally depend on the body result, so without the chain
            # the increment could be scheduled before inner loops.
            next_iv = ChainOp(
                result=algebra.AddOp(left=iv, right=Index().constant(1), type=Index()),
                effect=body_result,
                type=Index(),
            )
            body_block_result = goto.BranchOp(
                target=header_self, arguments=pack([next_iv])
            )
        else:
            # No chain here: the back-edge already consumes the next carry
            # values, which carry the body's effects via dataflow.
            next_iv = algebra.AddOp(left=iv, right=Index().constant(1), type=Index())
            if len(carries) == 1:
                next_carry_values: list[dgen.Value] = [body_result]
            else:
                next_carry_values = unpack(body_result)
                # ``unpack`` decomposes only builtin.PackOp and aggregate
                # Constants; any other tuple-shaped value (e.g. record.pack)
                # passes the type-level carry verification but cannot be
                # spliced with the incremented IV here.
                if len(next_carry_values) != len(carries):
                    raise TypeError(
                        f"ForOp body result must decompose into "
                        f"{len(carries)} next carry values (a builtin.pack "
                        f"or aggregate Constant); got {body_result.name!r}"
                    )
            body_block_result = goto.BranchOp(
                target=header_self,
                arguments=pack([next_iv, *next_carry_values]),
            )
        body_block = dgen.Block(
            result=body_block_result,
            args=[iv, *carries],
            captures=[header_self, header_exit, *op.body.captures],
        )
        body_label = goto.LabelOp(
            name="loop_body",
            initial_arguments=pack([]),
            body=body_block,
        )

        _resolve_jump_markers(body_block, header_self, header_exit)

        # Header: compare, branch to body or %exit.
        upper_bound = Index().constant(op.upper_bound.__constant__.to_json())
        comparison = algebra.LessThanOp(
            left=header_iv, right=upper_bound, type=Boolean()
        )
        conditional_branch = goto.ConditionalBranchOp(
            condition=comparison,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=pack([header_iv, *header_carries]),
            false_arguments=pack([]),
        )
        lower_bound = ConstantOp.from_constant(
            Index().constant(op.lower_bound.__constant__.to_json())
        )
        return goto.RegionOp(
            name="loop_header",
            initial_arguments=pack([lower_bound, *unpack(op.initial_arguments)]),
            type=builtin.Nil(),
            body=dgen.Block(
                result=conditional_branch,
                parameters=[header_self, header_exit],
                args=[header_iv, *header_carries],
                captures=list(op.body.captures),
            ),
        )

    @lowering_for(control_flow.WhileOp)
    def lower_while(self, op: control_flow.WhileOp) -> dgen.Value | None:
        # Block args for header and body, one per loop-carried variable.
        header_args = [
            BlockArgument(name=arg.name, type=arg.type) for arg in op.condition.args
        ]
        body_args = [
            BlockArgument(name=arg.name, type=arg.type) for arg in op.body.args
        ]

        header_self = BlockParameter(name="self", type=goto.Label())
        header_exit = BlockParameter(name="exit", type=goto.Label())

        # --- Body label: remap body block args, append back-edge ---
        for old, new in zip(op.body.args, body_args):
            op.body.replace_uses_of(old, new)

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
            name="while_body",
            initial_arguments=pack([]),
            body=body_block,
        )

        _resolve_jump_markers(body_block, header_self, header_exit)

        # --- Header: remap condition block args, append conditional branch ---
        for old, new in zip(op.condition.args, header_args):
            op.condition.replace_uses_of(old, new)

        conditional_branch = goto.ConditionalBranchOp(
            condition=op.condition.result,
            true_target=body_label,
            false_target=header_exit,
            true_arguments=pack(header_args),
            false_arguments=pack([]),
        )

        return goto.RegionOp(
            name="while_header",
            initial_arguments=op.initial_arguments,
            type=builtin.Nil(),
            body=dgen.Block(
                result=conditional_branch,
                parameters=[header_self, header_exit],
                args=header_args,
                captures=list(op.condition.captures) + list(op.body.captures),
            ),
        )
