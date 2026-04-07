"""Lower c.CReturnOp to goto.branch targeting function exit.

When a function body contains CReturnOp (early return inside control
flow), this pass wraps the body in a goto.RegionOp with an exit
parameter, and replaces each CReturnOp with a goto.BranchOp targeting
that exit. The normal end-of-function result also branches to exit.

Must run *before* ControlFlowToGoto so that break/continue markers
in the same body are resolved after early returns.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import function, goto
from dgen.dialects.builtin import ChainOp
from dgen.graph import all_values
from dgen.module import pack
from dgen.passes.pass_ import Pass, lowering_for

from dcc2.dialects.c import CReturnOp


class _ResolveCReturn(Pass):
    """Replace CReturnOp markers with goto.branch to function exit."""

    allow_unregistered_ops = True

    def __init__(self, exit_param: BlockParameter) -> None:
        self._exit_param = exit_param

    def _lower_block(self, block: dgen.Block) -> None:
        block.captures = [self._exit_param, *block.captures]
        super()._lower_block(block)

    @lowering_for(CReturnOp)
    def lower_c_return(self, op: CReturnOp) -> dgen.Value | None:
        return goto.BranchOp(target=self._exit_param, arguments=pack([op.value]))


class CReturnToGoto(Pass):
    """Wrap function bodies containing CReturnOp in a goto.RegionOp."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._counter = 0

    @lowering_for(function.FunctionOp)
    def lower_function(self, op: function.FunctionOp) -> dgen.Value | None:
        # Check if body contains any CReturnOp (including in nested blocks).
        has_return = any(isinstance(v, CReturnOp) for v in all_values(op.body.result))
        if not has_return:
            return None

        lid = self._counter
        self._counter += 1

        region_self = BlockParameter(name="self", type=goto.Label())
        exit_param = BlockParameter(name=f"func_exit{lid}", type=goto.Label())

        # Resolve CReturnOp markers — they branch to %self (the merge
        # point) with their return value as the phi argument.
        _ResolveCReturn(region_self)._lower_block(op.body)
        body_captures = [c for c in op.body.captures if c is not region_self]

        result_arg = BlockArgument(name=f"func_result{lid}", type=op.result_type)

        # If the body's normal exit is reachable (its result type
        # matches the function return type), branch to %self with
        # the body result. Otherwise all paths return early and the
        # body result is just a dependency anchor.
        # The normal exit is reachable if the body result type matches
        # the function return type. When all paths return early, the
        # body result is Nil-typed (control flow) and the normal exit
        # is dead code.
        body_terminated = type(op.body.result.type) is not type(op.result_type)
        if body_terminated:
            # All paths return early — body result is just effects.
            region_result = ChainOp(
                lhs=result_arg, rhs=op.body.result, type=op.result_type
            )
        else:
            normal_exit = goto.BranchOp(
                target=region_self, arguments=pack([op.body.result])
            )
            region_result = ChainOp(
                lhs=result_arg, rhs=normal_exit, type=op.result_type
            )

        region = goto.RegionOp(
            name=f"func_region{lid}",
            initial_arguments=pack([]),
            type=op.result_type,
            body=dgen.Block(
                result=region_result,
                parameters=[region_self, exit_param],
                args=[result_arg],
                captures=list(op.body.args) + body_captures,
            ),
        )

        return function.FunctionOp(
            name=op.name,
            result_type=op.result_type,
            type=op.type,
            body=dgen.Block(
                result=region,
                args=list(op.body.args),
            ),
        )
