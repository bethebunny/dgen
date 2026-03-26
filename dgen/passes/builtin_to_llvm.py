"""Lower builtin dialect ops to LLVM dialect ops.

Handles: IfOp, CallOp.
Passes through unchanged: ConstantOp, PackOp, and any LLVM dialect ops.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, control_flow, function, goto, llvm
from dgen.dialects.builtin import Nil, String
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler

_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))


class BuiltinToLLVMLowering(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._if_counter = 0

    @lowering_for(control_flow.IfOp)
    def lower_if(self, op: control_flow.IfOp, rewriter: Rewriter) -> bool:
        if_id = self._if_counter
        self._if_counter += 1

        # Create flat labels
        then_label = goto.LabelOp(name=f"then_{if_id}")
        else_label = goto.LabelOp(name=f"else_{if_id}")
        merge_label = goto.LabelOp(name=f"merge_{if_id}")

        # Merge label produces the if result via ArgOp
        merge_val = goto.ArgOp(
            name=f"merge_val{if_id}", label=merge_label, type=op.type
        )

        # Convert i64 condition to i1 via icmp ne 0
        zero = ConstantOp(value=0, type=builtin.Index())
        cond_i1 = llvm.IcmpOp(
            pred=String().constant("ne"),
            lhs=op.condition,
            rhs=zero,
        )

        assert isinstance(op.then_arguments, PackOp)
        assert isinstance(op.else_arguments, PackOp)
        cond_br = goto.ConditionalBranchOp(
            condition=cond_i1,
            true_target=then_label,
            false_target=else_label,
            true_arguments=op.then_arguments,
            false_arguments=op.else_arguments,
        )

        # Then branch: remap block args, add branch to merge
        then_rewriter = Rewriter(op.then_body)
        if op.then_body.args:
            then_arg = goto.ArgOp(label=then_label, type=op.then_body.args[0].type)
            then_rewriter.replace_uses(op.then_body.args[0], then_arg)
        then_result = op.then_body.result
        if not isinstance(then_result, Nil):
            then_br = goto.BranchOp(
                target=merge_label,
                arguments=PackOp(
                    values=[then_result],
                    type=builtin.List(element_type=then_result.type),
                ),
            )
        else:
            then_br = goto.BranchOp(target=merge_label, arguments=_EMPTY_PACK)

        # Else branch: remap block args, add branch to merge
        else_rewriter = Rewriter(op.else_body)
        if op.else_body.args:
            else_arg = goto.ArgOp(label=else_label, type=op.else_body.args[0].type)
            else_rewriter.replace_uses(op.else_body.args[0], else_arg)
        else_result = op.else_body.result
        if not isinstance(else_result, Nil):
            else_br = goto.BranchOp(
                target=merge_label,
                arguments=PackOp(
                    values=[else_result],
                    type=builtin.List(element_type=else_result.type),
                ),
            )
        else:
            else_br = goto.BranchOp(target=merge_label, arguments=_EMPTY_PACK)

        # Keep everything alive: chain the branches and cond_br
        alive = builtin.ChainOp(lhs=then_br, rhs=else_br, type=Nil())
        alive = builtin.ChainOp(lhs=cond_br, rhs=alive, type=Nil())
        replacement = builtin.ChainOp(lhs=merge_val, rhs=alive, type=op.type)

        rewriter.replace_uses(op, replacement)
        return True

    @lowering_for(function.CallOp)
    def lower_call(self, op: function.CallOp, rewriter: Rewriter) -> bool:
        callee_name = op.callee.name
        assert callee_name is not None
        if isinstance(op.arguments, PackOp):
            args = list(op.arguments.values)
        else:
            args = [op.arguments]
        pack = PackOp(values=args, type=op.arguments.type)
        llvm_call = llvm.CallOp(
            callee=String().constant(callee_name),
            args=pack,
            type=op.type,
        )
        rewriter.replace_uses(op, llvm_call)
        return True
