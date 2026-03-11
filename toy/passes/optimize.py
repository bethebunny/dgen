"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

from collections.abc import Sequence

import dgen
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import shape_constant, toy


# ===----------------------------------------------------------------------=== #
# ToyOptimize pass
# ===----------------------------------------------------------------------=== #


class ToyOptimize(Pass):
    allow_unregistered_ops = True

    @lowering_for(toy.TransposeOp)
    def eliminate_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        if not isinstance(op.input, toy.TransposeOp):
            return False
        rewriter.replace_uses(op, op.input.input)
        return True

    @lowering_for(toy.ReshapeOp)
    def fold_constants(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        defn = op.input
        if not isinstance(defn, ConstantOp):
            return False
        if not isinstance(op.type, toy.Tensor):
            return False
        target_shape = op.type.shape
        # Skip same-shape folds (simplify_reshape handles those)
        if isinstance(defn.type, toy.Tensor):
            if defn.type.shape == target_shape:
                return False
        new_op = ConstantOp(
            value=defn.memory.to_json(),
            type=toy.Tensor(shape=shape_constant(target_shape.__constant__.to_json())),
        )
        rewriter.replace_op(op, new_op)
        return True

    @lowering_for(toy.ReshapeOp)
    def simplify_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        defn = op.input

        # Reshape of constant with matching shape -> remove
        if isinstance(defn, ConstantOp):
            if isinstance(op.type, toy.Tensor) and isinstance(defn.type, toy.Tensor):
                if op.type.shape == defn.type.shape:
                    rewriter.replace_uses(op, defn)
                    return True

        # Reshape of reshape -> collapse
        if isinstance(defn, toy.ReshapeOp):
            new_op = toy.ReshapeOp(input=defn.input, type=op.type)
            rewriter.replace_op(op, new_op)
            return True

        return False


# ===----------------------------------------------------------------------=== #
# Dead code elimination
# ===----------------------------------------------------------------------=== #


def collect_uses(ops: Sequence[dgen.Op]) -> set[int]:
    """Return set of id()s of Values referenced as parameters and operands."""
    used: set[int] = set()
    for op in ops:
        for _, v in op.parameters:
            used.add(id(v))
        for _, v in op.operands:
            used.add(id(v))
    return used


def eliminate_dead_code(func: builtin.FunctionOp) -> None:
    changed = True
    while changed:
        changed = False
        used = collect_uses(func.body.ops)
        to_remove: list[int] = []
        for i, op in enumerate(func.body.ops):
            if isinstance(op, (toy.PrintOp, builtin.ReturnOp)):
                continue
            if id(op) not in used:
                to_remove.append(i)
                changed = True
        _remove_indices(func.body.ops, to_remove)


def _remove_indices(ops: list[dgen.Op], indices: Sequence[int]) -> None:
    for idx in reversed(indices):
        ops.pop(idx)


# ===----------------------------------------------------------------------=== #
# Pipeline
# ===----------------------------------------------------------------------=== #


def optimize(m: Module) -> Module:
    return ToyOptimize().run(m)
