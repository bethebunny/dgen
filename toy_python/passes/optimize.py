"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

import dataclasses
from copy import deepcopy

from toy_python.dialects import builtin, toy


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def collect_uses(ops: list[builtin.Op]) -> set[int]:
    """Return set of id()s of Values referenced as operands."""
    used: set[int] = set()
    for op in ops:
        for v in op.operands:
            used.add(id(v))
    return used


def rewrite_uses(ops: list[builtin.Op], old_value: builtin.Value, new_value: builtin.Value):
    """Replace all operand references to old_value with new_value."""
    for op in ops:
        for f in dataclasses.fields(op):
            if f.name == "name":
                continue
            val = getattr(op, f.name)
            if val is old_value:
                object.__setattr__(op, f.name, new_value)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if item is old_value:
                        val[i] = new_value


# ===----------------------------------------------------------------------=== #
# Transforms
# ===----------------------------------------------------------------------=== #


def eliminate_transpose(func: builtin.FuncOp):
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


def fold_constants(func: builtin.FuncOp):
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        defn = op.input
        if not isinstance(defn, toy.ConstantOp):
            continue
        if not isinstance(op.type, toy.RankedTensorType):
            continue
        target_shape = op.type.shape
        # Skip same-shape folds (simplify_reshape handles those)
        if isinstance(defn.type, toy.RankedTensorType):
            if defn.type.shape == target_shape:
                continue
        new_op = toy.ConstantOp(
            value=list(defn.value),
            shape=list(target_shape),
            type=toy.RankedTensorType(shape=list(target_shape)),
        )
        # Transfer identity: rewrite uses of old op to new op
        rewrite_uses(ops, op, new_op)
        ops[i] = new_op


def simplify_reshape(func: builtin.FuncOp):
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        defn = op.input

        # Reshape of constant with matching shape -> remove
        if isinstance(defn, toy.ConstantOp):
            if isinstance(op.type, toy.RankedTensorType) and isinstance(
                defn.type, toy.RankedTensorType
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


def eliminate_dead_code(func: builtin.FuncOp):
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


def _remove_indices(ops: list, indices: list[int]):
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
