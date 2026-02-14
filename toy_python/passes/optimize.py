"""Ch3: IR-to-IR optimization passes for the Toy dialect."""

from __future__ import annotations

from copy import deepcopy

from toy_python.dialects.toy_ops import (
    Module,
    FuncOp,
    AnyToyOp,
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    GenericCallOp,
    PrintOp,
    ReturnOp,
    RankedTensorType,
)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def get_result_name(op: AnyToyOp) -> str | None:
    if isinstance(
        op, (ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp, GenericCallOp)
    ):
        return op.result
    return None


def get_operands(op: AnyToyOp) -> list[str]:
    if isinstance(op, TransposeOp):
        return [op.input]
    if isinstance(op, ReshapeOp):
        return [op.input]
    if isinstance(op, MulOp):
        return [op.lhs, op.rhs]
    if isinstance(op, AddOp):
        return [op.lhs, op.rhs]
    if isinstance(op, GenericCallOp):
        return list(op.args)
    if isinstance(op, PrintOp):
        return [op.input]
    if isinstance(op, ReturnOp):
        return [op.value] if op.value is not None else []
    return []


def collect_uses(ops: list[AnyToyOp]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        for name in get_operands(op):
            counts[name] = counts.get(name, 0) + 1
    return counts


def find_def(ops: list[AnyToyOp], name: str) -> int | None:
    for i, op in enumerate(ops):
        if get_result_name(op) == name:
            return i
    return None


def rewrite_uses(ops: list[AnyToyOp], old_name: str, new_name: str):
    for i, op in enumerate(ops):
        if isinstance(op, TransposeOp) and op.input == old_name:
            ops[i] = TransposeOp(
                result=op.result, input=new_name, type=op.type
            )
        elif isinstance(op, ReshapeOp) and op.input == old_name:
            ops[i] = ReshapeOp(
                result=op.result, input=new_name, type=op.type
            )
        elif isinstance(op, MulOp) and (
            op.lhs == old_name or op.rhs == old_name
        ):
            ops[i] = MulOp(
                result=op.result,
                lhs=new_name if op.lhs == old_name else op.lhs,
                rhs=new_name if op.rhs == old_name else op.rhs,
                type=op.type,
            )
        elif isinstance(op, AddOp) and (
            op.lhs == old_name or op.rhs == old_name
        ):
            ops[i] = AddOp(
                result=op.result,
                lhs=new_name if op.lhs == old_name else op.lhs,
                rhs=new_name if op.rhs == old_name else op.rhs,
                type=op.type,
            )
        elif isinstance(op, GenericCallOp) and old_name in op.args:
            ops[i] = GenericCallOp(
                result=op.result,
                callee=op.callee,
                args=[new_name if a == old_name else a for a in op.args],
                type=op.type,
            )
        elif isinstance(op, PrintOp) and op.input == old_name:
            ops[i] = PrintOp(input=new_name)
        elif isinstance(op, ReturnOp) and op.value == old_name:
            ops[i] = ReturnOp(value=new_name)


# ===----------------------------------------------------------------------=== #
# Transforms
# ===----------------------------------------------------------------------=== #


def eliminate_transpose(func: FuncOp):
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, TransposeOp):
            continue
        outer_input = op.input
        outer_result = op.result
        def_idx = find_def(ops, outer_input)
        if def_idx is None:
            continue
        if not isinstance(ops[def_idx], TransposeOp):
            continue
        inner_input = ops[def_idx].input
        rewrite_uses(ops, outer_result, inner_input)
        to_remove.append(i)

    _remove_indices(ops, to_remove)


def fold_constants(func: FuncOp):
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, ReshapeOp):
            continue
        reshape_input = op.input
        reshape_result = op.result
        def_idx = find_def(ops, reshape_input)
        if def_idx is None:
            continue
        if not isinstance(ops[def_idx], ConstantOp):
            continue
        if not isinstance(op.type, RankedTensorType):
            continue
        target_shape = op.type.shape
        # Skip same-shape folds (simplify_reshape handles those)
        if isinstance(ops[def_idx].type, RankedTensorType):
            if ops[def_idx].type.shape == target_shape:
                continue
        ops[i] = ConstantOp(
            result=reshape_result,
            value=list(ops[def_idx].value),
            shape=list(target_shape),
            type=RankedTensorType(shape=list(target_shape)),
        )


def simplify_reshape(func: FuncOp):
    to_remove: list[int] = []
    ops = func.body.ops
    for i, op in enumerate(ops):
        if not isinstance(op, ReshapeOp):
            continue
        reshape_input = op.input
        reshape_result = op.result
        def_idx = find_def(ops, reshape_input)
        if def_idx is None:
            continue

        # Reshape of constant with matching shape -> remove
        if isinstance(ops[def_idx], ConstantOp):
            if isinstance(op.type, RankedTensorType) and isinstance(
                ops[def_idx].type, RankedTensorType
            ):
                if op.type.shape == ops[def_idx].type.shape:
                    rewrite_uses(ops, reshape_result, reshape_input)
                    to_remove.append(i)
                    continue

        # Reshape of reshape -> collapse
        if isinstance(ops[def_idx], ReshapeOp):
            inner_input = ops[def_idx].input
            ops[i] = ReshapeOp(
                result=reshape_result,
                input=inner_input,
                type=op.type,
            )

    _remove_indices(ops, to_remove)


def eliminate_dead_code(func: FuncOp):
    changed = True
    while changed:
        changed = False
        uses = collect_uses(func.body.ops)
        to_remove: list[int] = []
        for i, op in enumerate(func.body.ops):
            if isinstance(op, (PrintOp, ReturnOp)):
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


def optimize(m: Module) -> Module:
    functions = []
    for func in m.functions:
        func = deepcopy(func)
        eliminate_transpose(func)
        fold_constants(func)
        simplify_reshape(func)
        eliminate_dead_code(func)
        functions.append(func)
    return Module(functions=functions)
