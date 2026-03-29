"""Autodiff pass: lower grad calls into concrete Toy IR via reverse-mode AD.

Runs before shape inference and before ToyToStructured.
Handles CallOp nodes whose callee is a GradOp — i.e. ``grad(f)(x)`` or
``var df = grad(f); df(x)`` — by inlining the gradient computation:

1. Clones the target function's body, substituting block args with actual args
2. Walks the cloned ops in reverse, accumulating adjoint (gradient) values
3. Replaces the CallOp with the computed gradient tensor

All generated ops use InferredShapeTensor types — shape inference resolves
them in a subsequent pass.

Supported ops for differentiation:
- toy.AddOp: d_lhs += d_out, d_rhs += d_out
- toy.MulOp: d_lhs += d_out * rhs, d_rhs += d_out * lhs
- toy.TransposeOp: d_input += transpose(d_out)
- toy.ReshapeOp: d_input (passthrough)
- builtin.ConstantOp: zero gradient
- builtin.ChainOp: propagates through lhs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dgen
from dgen.dialects.builtin import ChainOp
from dgen.dialects.function import CallOp, FunctionOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass, lowering_for
from toy.dialects import shape_constant, toy
from toy.dialects.diff import GradOp

if TYPE_CHECKING:
    from dgen.compiler import Compiler

# Use InferredShapeTensor for all gradient ops — shape inference resolves later
_INFERRED = toy.InferredShapeTensor


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


def _shape_or_none(val: dgen.Value) -> list[int] | None:
    """Extract shape if the value has a concrete Tensor type."""
    if isinstance(val.type, toy.Tensor):
        result = val.type.shape.__constant__.to_json()
        assert isinstance(result, list)
        return result
    return None


def _ones_like(val: dgen.Value) -> ConstantOp:
    """Create ones with the same shape as val, or scalar [1.0] if unknown."""
    import math

    shape = _shape_or_none(val)
    if shape is None:
        shape = [1]
    return ConstantOp(
        value=[1.0] * math.prod(shape),
        type=toy.Tensor(shape=shape_constant(shape)),
    )


def _zeros_like(val: dgen.Value) -> ConstantOp:
    """Create zeros with the same shape as val, or scalar [0.0] if unknown."""
    import math

    shape = _shape_or_none(val)
    if shape is None:
        shape = [1]
    return ConstantOp(
        value=[0.0] * math.prod(shape),
        type=toy.Tensor(shape=shape_constant(shape)),
    )


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
            add_op = toy.AddOp(lhs=existing, rhs=grad, type=_INFERRED())
            new_grad_ops.append(add_op)
            adjoints[id(val)] = add_op

    # Seed: d(output)/d(output) = ones matching the output shape
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
            d_lhs = toy.MulOp(lhs=d_out, rhs=op.rhs, type=_INFERRED())
            new_grad_ops.append(d_lhs)
            accumulate(op.lhs, d_lhs)
            d_rhs = toy.MulOp(lhs=d_out, rhs=op.lhs, type=_INFERRED())
            new_grad_ops.append(d_rhs)
            accumulate(op.rhs, d_rhs)

        elif isinstance(op, toy.TransposeOp):
            d_input = toy.TransposeOp(input=d_out, type=_INFERRED())
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
            # This parameter doesn't affect the output — gradient is zero
            grad = _zeros_like(actual_arg)
            new_grad_ops.append(grad)
        grad_values.append(grad)

    all_ops = cloned_ops + new_grad_ops
    return all_ops, grad_values


class Autodiff(Pass):
    """Lower grad calls into concrete Toy IR via reverse-mode autodiff.

    Handles CallOp nodes whose callee is a GradOp (the symbolic gradient
    of a function). The GradOp itself becomes dead after the CallOp is
    replaced and is automatically removed from the block's ops.
    """

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._func_map: dict[str, FunctionOp] = {}

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        self._func_map = {f.name: f for f in module.functions if f.name is not None}
        for func in module.functions:
            self._run_block(func.body)
        return module

    def _infer_shapes(self, func: FunctionOp) -> None:
        """Run a quick shape inference on the function body.

        Resolves InferredShapeTensor types to concrete Tensor types so the
        autodiff pass can create correctly-shaped gradient ops.
        """
        from toy.passes.shape_inference import ShapeInference

        si = ShapeInference()
        si._func_map = self._func_map
        si._run_block(func.body)

    @lowering_for(CallOp)
    def lower_grad_call(self, op: CallOp) -> dgen.Value | None:
        """Replace CallOp(callee=GradOp(f), args) with inlined gradient."""
        if not isinstance(op.callee, GradOp):
            return None  # Not a gradient call — leave for other passes

        grad_op = op.callee
        callee_name = grad_op.callee.name
        if callee_name is None:
            raise RuntimeError("GradOp callee must have a name")
        callee = self._func_map.get(callee_name)
        if callee is None:
            raise RuntimeError(f"Unknown function in grad: {callee_name}")

        # Extract arguments from the CallOp
        if isinstance(op.arguments, PackOp):
            args = list(op.arguments)
        else:
            args = [op.arguments]

        # Propagate argument types to the callee and run shape inference
        # so the cloned ops have concrete types for correct gradient shapes
        for block_arg, actual_arg in zip(callee.body.args, args):
            if isinstance(actual_arg.type, toy.Tensor):
                block_arg.type = actual_arg.type
        self._infer_shapes(callee)

        # Perform autodiff
        _new_ops, grad_values = _inline_and_differentiate(callee, args)

        # Return the gradient for the first argument
        return grad_values[0]
