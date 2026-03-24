"""Lower actor pipeline ops by inlining actor bodies.

Each actor body is self-contained: it receives an input buffer,
performs its computation (including any loops), and produces an
output via actor.produce. The pipeline lowering inlines each body
in sequence, threading outputs to inputs.

The lowering uses the generic `inline_block` operation: because all
blocks satisfy the closed-block invariant, inlining is a simple
arg substitution — map block.args → actual values and the result
is valid in the caller's scope.
"""

from __future__ import annotations

import dgen
from dgen.graph import inline_block
from dgen.passes.pass_ import Pass, Rewriter, lowering_for

from actor.dialects.actor import ActorOp, PipelineOp, ProduceOp


class ActorToAffine(Pass):
    allow_unregistered_ops = True

    @lowering_for(PipelineOp)
    def lower_pipeline(self, op: PipelineOp, rewriter: Rewriter) -> bool:
        # Step 1: Inline the pipeline body — substitute pipeline body arg
        # with the pipeline's input operand.
        inline_block(op.body, [op.input])

        # Step 2: Inline each actor body in sequence, threading produce
        # values through. After pipeline inline, actor inputs reference
        # actual values from the function scope.
        #
        # TODO: check fusibility criteria (matching consume/produce rates)
        # before inlining. Mismatched rates need separate loops with an
        # intermediate buffer instead of direct inlining.
        actors = [o for o in op.body.ops if isinstance(o, ActorOp)]
        pipeline_rewriter = Rewriter(op.body)
        for actor in actors:
            result = inline_block(actor.body, [actor.input])
            assert isinstance(result, ProduceOp)
            pipeline_rewriter.replace_uses(actor, result.value)

        # Step 3: Replace the pipeline op in the function body with the
        # pipeline body's result (the last actor's produce value).
        rewriter.replace_uses(op, op.body.result)
        return True
