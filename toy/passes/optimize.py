"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

import dgen
from dgen.module import Module, _walk_all_ops
from dgen.passes.pass_ import Pass, lowering_for
from toy.dialects import toy


class ToyOptimize(Pass):
    allow_unregistered_ops = True

    def verify_postconditions(self, module: Module) -> None:
        super().verify_postconditions(module)
        for func in module.functions:
            for op in _walk_all_ops(func):
                if isinstance(op, toy.TransposeOp):
                    assert not isinstance(op.input, toy.TransposeOp), (
                        "double transpose remains after ToyOptimize"
                    )
                if isinstance(op, toy.ReshapeOp):
                    assert not isinstance(op.input, toy.ReshapeOp), (
                        "consecutive reshape remains after ToyOptimize"
                    )

    @lowering_for(toy.TransposeOp)
    def eliminate_transpose(self, op: toy.TransposeOp) -> dgen.Value | None:
        if isinstance(op.input, toy.TransposeOp):
            return op.input.input
        return None

    @lowering_for(toy.ReshapeOp)
    def simplify_reshape(self, op: toy.ReshapeOp) -> dgen.Value | None:
        # Collapse sequences of reshapes
        if isinstance(op.input, toy.ReshapeOp):
            return toy.ReshapeOp(input=op.input.input, type=op.type)
        return None
