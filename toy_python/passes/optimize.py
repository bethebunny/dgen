"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

from copy import deepcopy

from toy_python.dialects import toy


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def get_result_name(op: toy.AnyOp) -> str | None:
    if isinstance(
        op,
        (
            toy.ConstantOp,
            toy.TransposeOp,
            toy.ReshapeOp,
            toy.MulOp,
            toy.AddOp,
            toy.GenericCallOp,
        ),
    ):
        return op.result
    return None


def get_operands(op: toy.AnyOp) -> list[str]:
    if isinstance(op, toy.TransposeOp):
        return [op.input]
    if isinstance(op, toy.ReshapeOp):
        return [op.input]
    if isinstance(op, toy.MulOp):
        return [op.lhs, op.rhs]
    if isinstance(op, toy.AddOp):
        return [op.lhs, op.rhs]
    if isinstance(op, toy.GenericCallOp):
        return list(op.args)
    if isinstance(op, toy.PrintOp):
        return [op.input]
    if isinstance(op, toy.ReturnOp):
        return [op.value] if op.value is not None else []
    return []


def collect_uses(ops: list[toy.AnyOp]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        for name in get_operands(op):
            counts[name] = counts.get(name, 0) + 1
    return counts


def find_def(ops: list[toy.AnyOp], name: str) -> int | None:
    for i, op in enumerate(ops):
        if get_result_name(op) == name:
            return i
    return None


def rewrite_uses(ops: list[toy.AnyOp], old_name: str, new_name: str):
    for i, op in enumerate(ops):
        if isinstance(op, toy.TransposeOp) and op.input == old_name:
            ops[i] = toy.TransposeOp(
                result=op.result, input=new_name, type=op.type
            )
        elif isinstance(op, toy.ReshapeOp) and op.input == old_name:
            ops[i] = toy.ReshapeOp(
                result=op.result, input=new_name, type=op.type
            )
        elif isinstance(op, toy.MulOp) and (
            op.lhs == old_name or op.rhs == old_name
        ):
            ops[i] = toy.MulOp(
                result=op.result,
                lhs=new_name if op.lhs == old_name else op.lhs,
                rhs=new_name if op.rhs == old_name else op.rhs,
                type=op.type,
            )
        elif isinstance(op, toy.AddOp) and (
            op.lhs == old_name or op.rhs == old_name
        ):
            ops[i] = toy.AddOp(
                result=op.result,
                lhs=new_name if op.lhs == old_name else op.lhs,
                rhs=new_name if op.rhs == old_name else op.rhs,
                type=op.type,
            )
        elif isinstance(op, toy.GenericCallOp) and old_name in op.args:
            ops[i] = toy.GenericCallOp(
                result=op.result,
                callee=op.callee,
                args=[new_name if a == old_name else a for a in op.args],
                type=op.type,
            )
        elif isinstance(op, toy.PrintOp) and op.input == old_name:
            ops[i] = toy.PrintOp(input=new_name)
        elif isinstance(op, toy.ReturnOp) and op.value == old_name:
            ops[i] = toy.ReturnOp(value=new_name)


# ===----------------------------------------------------------------------=== #
# Transforms
# ===----------------------------------------------------------------------=== #


def eliminate_transpose(func: toy.FuncOp):
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.TransposeOp):
            continue
        outer_input = op.input
        outer_result = op.result
        def_idx = find_def(ops, outer_input)
        if def_idx is None:
            continue
        if not isinstance(ops[def_idx], toy.TransposeOp):
            continue
        inner_input = ops[def_idx].input
        rewrite_uses(ops, outer_result, inner_input)
        to_remove.append(i)

    _remove_indices(ops, to_remove)


def fold_constants(func: toy.FuncOp):
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        reshape_input = op.input
        reshape_result = op.result
        def_idx = find_def(ops, reshape_input)
        if def_idx is None:
            continue
        if not isinstance(ops[def_idx], toy.ConstantOp):
            continue
        if not isinstance(op.type, toy.RankedTensorType):
            continue
        target_shape = op.type.shape
        # Skip same-shape folds (simplify_reshape handles those)
        if isinstance(ops[def_idx].type, toy.RankedTensorType):
            if ops[def_idx].type.shape == target_shape:
                continue
        ops[i] = toy.ConstantOp(
            result=reshape_result,
            value=list(ops[def_idx].value),
            shape=list(target_shape),
            type=toy.RankedTensorType(shape=list(target_shape)),
        )


def simplify_reshape(func: toy.FuncOp):
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, toy.ReshapeOp):
            continue
        reshape_input = op.input
        reshape_result = op.result
        def_idx = find_def(ops, reshape_input)
        if def_idx is None:
            continue

        # Reshape of constant with matching shape -> remove
        if isinstance(ops[def_idx], toy.ConstantOp):
            if isinstance(op.type, toy.RankedTensorType) and isinstance(
                ops[def_idx].type, toy.RankedTensorType
            ):
                if op.type.shape == ops[def_idx].type.shape:
                    rewrite_uses(ops, reshape_result, reshape_input)
                    to_remove.append(i)
                    continue

        # Reshape of reshape -> collapse
        if isinstance(ops[def_idx], toy.ReshapeOp):
            inner_input = ops[def_idx].input
            ops[i] = toy.ReshapeOp(
                result=reshape_result,
                input=inner_input,
                type=op.type,
            )

    _remove_indices(ops, to_remove)


def eliminate_dead_code(func: toy.FuncOp):
    changed = True
    while changed:
        changed = False
        uses = collect_uses(func.body.ops)
        to_remove: list[int] = []
        for i, op in enumerate(func.body.ops):
            if isinstance(op, (toy.PrintOp, toy.ReturnOp)):
                continue
            name = get_result_name(op)
            if name is None:
                continue
            if name not in uses:
                to_remove.append(i)
                changed = True
        _remove_indices(func.body.ops, to_remove)


def _remove_indices(ops: list, indices: list[int]):
    for idx in reversed(indices):
        ops.pop(idx)


# ===----------------------------------------------------------------------=== #
# Pipeline
# ===----------------------------------------------------------------------=== #


def optimize(m: toy.Module) -> toy.Module:
    functions = []
    for func in m.functions:
        func = deepcopy(func)
        eliminate_transpose(func)
        fold_constants(func)
        simplify_reshape(func)
        eliminate_dead_code(func)
        functions.append(func)
    return toy.Module(functions=functions)
