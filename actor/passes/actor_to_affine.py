"""Lower actor pipeline ops by inlining actor bodies.

Each actor body is self-contained: it receives an input buffer,
performs its computation (including any loops), and produces an
output via actor.produce. The pipeline lowering inlines each body
in sequence, threading outputs to inputs.
"""

from __future__ import annotations

from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from dgen.type import Value

from actor.dialects.actor import ActorOp, PipelineOp, ProduceOp


class ActorToAffine(Pass):
    allow_unregistered_ops = True

    @lowering_for(PipelineOp)
    def lower_pipeline(self, op: PipelineOp, rewriter: Rewriter) -> bool:
        current: Value = op.input
        for body_op in op.body.ops:
            if not isinstance(body_op, ActorOp):
                continue
            Rewriter(body_op.body).replace_uses(body_op.body.args[0], current)
            result = body_op.body.result
            assert isinstance(result, ProduceOp)
            current = result.value
        rewriter.replace_uses(op, current)
        return True
