"""Lower actor dialect ops to affine dialect ops.

Actor bodies are inlined into affine loops. Fusion: when actor A's
produce_rate equals actor B's consume_rate, the two stages share a
single loop. When rates differ, separate loops with an intermediate
buffer are emitted.
"""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument
from dgen.dialects import builtin
from dgen.dialects.builtin import Index, List
from dgen.module import PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from dgen.type import Value
from toy.dialects import affine, shape_constant

from actor.dialects.actor import ActorOp, PipelineOp, ProduceOp


def _alloc(dims: list[int]) -> affine.AllocOp:
    shape = shape_constant(dims)
    return affine.AllocOp(shape=shape, type=affine.MemRef(shape=shape))


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
        assert len(chain) >= 2

        a, b = chain[0], chain[1]
        r_a_produce = a.produce_rate.__constant__.to_json()
        r_b_consume = b.consume_rate.__constant__.to_json()
        assert isinstance(r_a_produce, int) and isinstance(r_b_consume, int)

        if r_a_produce == r_b_consume:
            result = self._lower_fused(op, chain, r_a_produce)
        else:
            result = self._lower_unfused(op, chain)

        rewriter.replace_uses(op, result)
        return True

    def _lower_fused(
        self, pipeline: PipelineOp, chain: list[ActorOp], rate: int
    ) -> Value:
        """Fused: all actors share one loop."""
        output = _alloc([rate])
        iv = BlockArgument(type=Index())
        idx = PackOp(values=[iv], type=List(element_type=Index()))

        current: Value = affine.LoadOp(memref=pipeline.input, indices=idx)
        for actor in chain:
            current = _inline_actor(actor, current)

        store = affine.StoreOp(value=current, memref=output, indices=idx)
        loop = affine.ForOp(
            lo=Index().constant(0),
            hi=Index().constant(rate),
            body=dgen.Block(result=store, args=[iv]),
        )
        inner = builtin.ChainOp(lhs=pipeline.input, rhs=loop, type=output.type)
        return builtin.ChainOp(lhs=output, rhs=inner, type=output.type)

    def _lower_unfused(self, pipeline: PipelineOp, chain: list[ActorOp]) -> Value:
        """Unfused: separate loops with intermediate buffer."""
        a, b = chain[0], chain[1]
        r1 = a.consume_rate.__constant__.to_json()
        r2 = b.consume_rate.__constant__.to_json()
        assert isinstance(r1, int) and isinstance(r2, int)

        buffer = _alloc([r1])
        output = _alloc([r2])

        # Loop 1: input → buffer (actor A)
        iv1 = BlockArgument(type=Index())
        idx1 = PackOp(values=[iv1], type=List(element_type=Index()))
        val1 = affine.LoadOp(memref=pipeline.input, indices=idx1)
        out1 = _inline_actor(a, val1)
        store1 = affine.StoreOp(value=out1, memref=buffer, indices=idx1)
        loop1 = affine.ForOp(
            lo=Index().constant(0),
            hi=Index().constant(r1),
            body=dgen.Block(result=store1, args=[iv1]),
        )

        # Loop 2: buffer → output (actor B)
        iv2 = BlockArgument(type=Index())
        idx2 = PackOp(values=[iv2], type=List(element_type=Index()))
        val2 = affine.LoadOp(memref=buffer, indices=idx2)
        out2 = _inline_actor(b, val2)
        store2 = affine.StoreOp(value=out2, memref=output, indices=idx2)
        loop2 = affine.ForOp(
            lo=Index().constant(0),
            hi=Index().constant(r2),
            body=dgen.Block(result=store2, args=[iv2]),
        )

        c1 = builtin.ChainOp(lhs=pipeline.input, rhs=loop1, type=buffer.type)
        c2 = builtin.ChainOp(lhs=buffer, rhs=c1, type=buffer.type)
        c3 = builtin.ChainOp(lhs=c2, rhs=loop2, type=output.type)
        return builtin.ChainOp(lhs=output, rhs=c3, type=output.type)
