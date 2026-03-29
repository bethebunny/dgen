"""Autodiff pass: lower diff.GradOp into concrete Toy IR via reverse-mode AD.

Runs after shape inference (types are resolved) and before ToyToStructured.
For each GradOp(callee=f, arguments=args), this pass:
1. Looks up the function f in the module
2. Clones the function body ops, substituting block args with actual args
3. Walks the cloned ops in reverse to accumulate adjoint (gradient) values
4. Replaces the GradOp with the computed gradient(s)

Supported ops for differentiation:
- toy.AddOp: d_lhs += d_out, d_rhs += d_out
- toy.MulOp: d_lhs += d_out * rhs, d_rhs += d_out * lhs
- toy.TransposeOp: d_input += transpose(d_out)
- toy.ReshapeOp: d_input (passthrough)
- builtin.ConstantOp: zero gradient
- builtin.ChainOp: propagates through lhs
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import dgen
from dgen.dialects.builtin import ChainOp
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, lowering_for
from toy.dialects import shape_constant, toy
from toy.dialects.diff import GradOp

if TYPE_CHECKING:
    from dgen.compiler import Compiler


def _shape(val: dgen.Value) -> list[int]:
    """Extract concrete shape from a Tensor-typed value."""
    assert isinstance(val.type, toy.Tensor), f"Expected Tensor, got {type(val.type)}"
    result = val.type.shape.__constant__.to_json()
    assert isinstance(result, list)
    return result


def _zeros_like(val: dgen.Value) -> ConstantOp:
    """Create a zero constant with the same shape as val."""
    shape = _shape(val)
    return ConstantOp(
        value=[0.0] * math.prod(shape),
        type=toy.Tensor(shape=shape_constant(shape)),
    )


def _ones_like(val: dgen.Value) -> ConstantOp:
    """Create a ones constant with the same shape as val."""
    shape = _shape(val)
    return ConstantOp(
        value=[1.0] * math.prod(shape),
        type=toy.Tensor(shape=shape_constant(shape)),
    )


def _clone_op(op: dgen.Op, val_map: dict[int, dgen.Value]) -> dgen.Op:
    """Clone an op, substituting all operand references via val_map."""

    def subst(v: dgen.Value) -> dgen.Value:
        return val_map.get(id(v), v)

    if isinstance(op, ConstantOp):
        clone = ConstantOp(value=op.value, type=op.type)
    elif isinstance(op, toy.AddOp):
        clone = toy.AddOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=op.type)
    elif isinstance(op, toy.MulOp):
        clone = toy.MulOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=op.type)
    elif isinstance(op, toy.TransposeOp):
        clone = toy.TransposeOp(input=subst(op.input), type=op.type)
    elif isinstance(op, toy.ReshapeOp):
        clone = toy.ReshapeOp(input=subst(op.input), type=op.type)
    elif isinstance(op, ChainOp):
        clone = ChainOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=op.type)
    else:
        raise RuntimeError(f"Cannot differentiate through {type(op).__name__}")
    val_map[id(op)] = clone
    return clone


def _inline_and_differentiate(
    func: FunctionOp,
    args: list[dgen.Value],
) -> tuple[list[dgen.Op], list[dgen.Value]]:
    """Perform reverse-mode autodiff on func applied to args.

    Returns (new_ops, grad_values) where:
    - new_ops: all new ops created (cloned forward + gradient backward)
    - grad_values: gradient for each function parameter, same order as args
    """
    body = func.body
    block_args = body.args

    # Build substitution map: block args -> actual args
    val_map: dict[int, dgen.Value] = {}
    for block_arg, actual_arg in zip(block_args, args):
        val_map[id(block_arg)] = actual_arg

    # Clone forward ops with substitution
    forward_ops = body.ops
    cloned_ops: list[dgen.Op] = []
    for op in forward_ops:
        cloned = _clone_op(op, val_map)
        cloned_ops.append(cloned)

    # Find the cloned result
    result_val = body.result
    cloned_result = val_map.get(id(result_val), result_val)

    # Strip ChainOps to find the actual computed value
    actual_result = cloned_result
    while isinstance(actual_result, ChainOp):
        actual_result = actual_result.lhs

    # Adjoint accumulator
    adjoints: dict[int, dgen.Value] = {}
    new_grad_ops: list[dgen.Op] = []

    def get_adjoint(val: dgen.Value) -> dgen.Value | None:
        return adjoints.get(id(val))

    def accumulate(val: dgen.Value, grad: dgen.Value) -> None:
        existing = adjoints.get(id(val))
        if existing is None:
            adjoints[id(val)] = grad
        else:
            add_op = toy.AddOp(lhs=existing, rhs=grad, type=existing.type)
            new_grad_ops.append(add_op)
            adjoints[id(val)] = add_op

    # Seed: d(output)/d(output) = ones
    seed = _ones_like(actual_result)
    new_grad_ops.append(seed)
    accumulate(actual_result, seed)

    # Reverse pass over cloned ops
    for op in reversed(cloned_ops):
        d_out = get_adjoint(op)
        if d_out is None:
            continue

        if isinstance(op, ConstantOp):
            pass

        elif isinstance(op, toy.AddOp):
            accumulate(op.lhs, d_out)
            accumulate(op.rhs, d_out)

        elif isinstance(op, toy.MulOp):
            # d(a * b)/da = b * d_out, d(a * b)/db = a * d_out
            d_lhs = toy.MulOp(lhs=d_out, rhs=op.rhs, type=op.type)
            new_grad_ops.append(d_lhs)
            accumulate(op.lhs, d_lhs)
            d_rhs = toy.MulOp(lhs=d_out, rhs=op.lhs, type=op.type)
            new_grad_ops.append(d_rhs)
            accumulate(op.rhs, d_rhs)

        elif isinstance(op, toy.TransposeOp):
            d_input = toy.TransposeOp(input=d_out, type=op.input.type)
            new_grad_ops.append(d_input)
            accumulate(op.input, d_input)

        elif isinstance(op, toy.ReshapeOp):
            accumulate(op.input, d_out)

        elif isinstance(op, ChainOp):
            accumulate(op.lhs, d_out)

    # Collect gradient values for each actual argument
    grad_values: list[dgen.Value] = []
    for actual_arg in args:
        grad = get_adjoint(actual_arg)
        if grad is None:
            grad = _zeros_like(actual_arg)
            new_grad_ops.append(grad)
        grad_values.append(grad)

    all_ops = cloned_ops + new_grad_ops
    return all_ops, grad_values


class Autodiff(Pass):
    """Lower diff.GradOp into concrete Toy IR via reverse-mode autodiff."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._func_map: dict[str, FunctionOp] = {}

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        self._func_map = {f.name: f for f in module.functions if f.name is not None}
        for func in module.functions:
            self._run_block(func.body)
        return module

    @lowering_for(GradOp)
    def lower_grad(self, op: GradOp) -> dgen.Value | None:
        callee_name = op.callee.name
        if callee_name is None:
            raise RuntimeError("GradOp callee must have a name")
        callee = self._func_map.get(callee_name)
        if callee is None:
            raise RuntimeError(f"Unknown function in grad: {callee_name}")

        # Extract arguments
        if isinstance(op.arguments, PackOp):
            args = list(op.arguments)
        else:
            args = [op.arguments]

        # Perform autodiff
        _new_ops, grad_values = _inline_and_differentiate(callee, args)

        # Return the gradient for the first argument
        return grad_values[0]
