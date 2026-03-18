"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import toy


class ToyOptimize(Pass):
    allow_unregistered_ops = True

    @lowering_for(toy.TransposeOp)
    def eliminate_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        if isinstance(op.input, toy.TransposeOp):
            return rewriter.replace_uses(op, op.input.input)

    @lowering_for(toy.ReshapeOp)
    def simplify_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        # Collapse sequences of reshapes
        if isinstance(op.input, toy.ReshapeOp):
            return rewriter.replace_uses(
                op, toy.ReshapeOp(input=op.input.input, type=op.type)
            )
