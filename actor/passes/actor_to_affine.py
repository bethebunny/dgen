"""Lower actor pipeline ops by inlining actor bodies.

Each actor body is self-contained: it receives an input buffer,
performs its computation (including any loops), and produces an
output via actor.produce. The pipeline lowering inlines each body
in sequence, threading outputs to inputs.
"""

from __future__ import annotations

from dgen.block import Block
from dgen.dialects.builtin import ChainOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from dgen.type import Value

from actor.dialects.actor import ActorOp, PipelineOp, ProduceOp


def _extract_chain(body: Block) -> list[ActorOp]:
    """Extract the linear actor chain from a pipeline body, forward order."""
    chain: list[ActorOp] = []
    current: Value = body.result
    while isinstance(current, ActorOp):
        chain.append(current)
        current = current.input
    chain.reverse()
    return chain


def _inline_actor(actor: ActorOp, input_val: Value) -> Value:
    """Inline an actor's body, replacing its block arg with input_val.

    Returns the value fed to the produce op (the actor's output).
    """
    Rewriter(actor.body).replace_uses(actor.body.args[0], input_val)
    result = actor.body.result
    assert isinstance(result, ProduceOp)
    return result.value


class ActorToAffine(Pass):
    allow_unregistered_ops = True

    @lowering_for(PipelineOp)
    def lower_pipeline(self, op: PipelineOp, rewriter: Rewriter) -> bool:
        chain = _extract_chain(op.body)
        stages: list[Value] = []
        current: Value = op.input
        for actor in chain:
            current = _inline_actor(actor, current)
            stages.append(current)
        # Chain all stage outputs at function level so walk_ops discovers
        # every stage's transitive ops (not just the last stage's).
        result = stages[-1]
        for stage in reversed(stages[:-1]):
            result = ChainOp(lhs=result, rhs=stage, type=result.type)
        rewriter.replace_uses(op, result)
        return True
