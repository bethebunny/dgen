"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import dgen
from dgen.dialects import builtin
from toy.dialects import shape_constant
from toy.dialects import toy

# ===----------------------------------------------------------------------=== #
# Helpers
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


def rewrite_uses(
    ops: Sequence[dgen.Op], old_value: dgen.Value, new_value: dgen.Value
) -> None:
    """Replace all parameter and operand references to old_value with new_value."""
    for op in ops:
        for name, param in op.parameters:
            if param is old_value:
                setattr(op, name, new_value)
        for name, operand in op.operands:
            if operand is old_value:
                setattr(op, name, new_value)


# ===----------------------------------------------------------------------=== #
# Transforms
# ===----------------------------------------------------------------------=== #


def eliminate_transpose(func: builtin.FuncOp) -> None:
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.TransposeOp):
            continue
        inner = op.input
        if not isinstance(inner, toy.TransposeOp):
            continue
        # Double transpose: replace uses of outer with inner's input
        rewrite_uses(ops, op, inner.input)
        to_remove.append(i)

    _remove_indices(ops, to_remove)


def fold_constants(func: builtin.FuncOp) -> None:
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        defn = op.input
        if not isinstance(defn, builtin.ConstantOp):
            continue
        if not isinstance(op.type, toy.TensorType):
            continue
        target_shape = op.type.shape
        # Skip same-shape folds (simplify_reshape handles those)
        if isinstance(defn.type, toy.TensorType):
            if defn.type.shape == target_shape:
                continue
        new_op = builtin.ConstantOp(
            value=list(defn.value.unpack()),
            type=toy.TensorType(
                shape=shape_constant(list(target_shape.__constant__.unpack()))
            ),
        )
        # Transfer identity: rewrite uses of old op to new op
        rewrite_uses(ops, op, new_op)
        ops[i] = new_op


def simplify_reshape(func: builtin.FuncOp) -> None:
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        defn = op.input

        # Reshape of constant with matching shape -> remove
        if isinstance(defn, builtin.ConstantOp):
            if isinstance(op.type, toy.TensorType) and isinstance(
                defn.type, toy.TensorType
            ):
                if op.type.shape == defn.type.shape:
                    rewrite_uses(ops, op, defn)
                    to_remove.append(i)
                    continue

        # Reshape of reshape -> collapse
        if isinstance(defn, toy.ReshapeOp):
            new_op = toy.ReshapeOp(
                input=defn.input,
                type=op.type,
            )
            rewrite_uses(ops, op, new_op)
            ops[i] = new_op

    _remove_indices(ops, to_remove)


def eliminate_dead_code(func: builtin.FuncOp) -> None:
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


def optimize(m: builtin.Module) -> builtin.Module:
    functions = []
    for func in m.functions:
        func = deepcopy(func)
        eliminate_transpose(func)
        fold_constants(func)
        simplify_reshape(func)
        eliminate_dead_code(func)
        functions.append(func)
    return builtin.Module(functions=functions)
