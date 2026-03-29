"""Autodiff pass: lower diff.GradOp into a synthesized gradient FunctionOp.

For each GradOp referencing function ``f``, this pass:
1. Differentiates ``f``'s body via reverse-mode AD
2. Wraps the result in a new ``FunctionOp`` named ``f_grad``
3. Adds it to the module
4. Replaces the GradOp with a name reference to ``f_grad``

After this pass, ``grad(f)(x)`` is a normal ``CallOp(callee="f_grad", args)``
that flows through shape inference, staging, and codegen like any other call.

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
from dgen.block import BlockArgument
from dgen.dialects.builtin import ChainOp
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp, Module
from toy.dialects import shape_constant, toy
from toy.dialects.diff import GradOp

if TYPE_CHECKING:
    pass

_INFERRED = toy.InferredShapeTensor


def _clone_op(op: dgen.Op, val_map: dict[int, dgen.Value]) -> dgen.Op:
    """Clone an op, substituting all operand references via val_map."""

    def subst(v: dgen.Value) -> dgen.Value:
        return val_map.get(id(v), v)

    if isinstance(op, ConstantOp):
        clone = ConstantOp(value=op.value, type=op.type)
    elif isinstance(op, toy.AddOp):
        clone = toy.AddOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=_INFERRED())
    elif isinstance(op, toy.MulOp):
        clone = toy.MulOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=_INFERRED())
    elif isinstance(op, toy.TransposeOp):
        clone = toy.TransposeOp(input=subst(op.input), type=_INFERRED())
    elif isinstance(op, toy.ReshapeOp):
        clone = toy.ReshapeOp(input=subst(op.input), type=_INFERRED())
    elif isinstance(op, ChainOp):
        clone = ChainOp(lhs=subst(op.lhs), rhs=subst(op.rhs), type=op.type)
    else:
        raise RuntimeError(f"Cannot differentiate through {type(op).__name__}")
    val_map[id(op)] = clone
    return clone


def _build_grad_function(
    name: str,
    func: FunctionOp,
) -> FunctionOp:
    """Synthesize a gradient FunctionOp from a primal function.

    The gradient function has the same parameters as the primal.
    It clones the primal's forward body, then walks in reverse to
    accumulate adjoint values. The result is the gradient w.r.t. the
    first parameter.
    """
    body = func.body
    primal_args = body.args

    # Create fresh block args for the gradient function
    grad_args = [BlockArgument(name=a.name, type=_INFERRED()) for a in primal_args]

    # Map primal block args → gradient block args
    val_map: dict[int, dgen.Value] = {}
    for primal_arg, grad_arg in zip(primal_args, grad_args):
        val_map[id(primal_arg)] = grad_arg

    # Clone forward ops with substitution
    for op in body.ops:
        _clone_op(op, val_map)

    # Find the cloned result
    cloned_result = val_map.get(id(body.result), body.result)

    # Strip ChainOps to find the actual computed value
    actual_result = cloned_result
    while isinstance(actual_result, ChainOp):
        actual_result = actual_result.lhs

    # Adjoint accumulator
    adjoints: dict[int, dgen.Value] = {}

    def accumulate(val: dgen.Value, grad: dgen.Value) -> None:
        existing = adjoints.get(id(val))
        if existing is None:
            adjoints[id(val)] = grad
        else:
            adjoints[id(val)] = toy.AddOp(lhs=existing, rhs=grad, type=_INFERRED())

    # Seed: d(output)/d(output) = ones matching the output shape.
    # fill_like broadcasts a scalar to match a template tensor's shape.
    one_scalar = ConstantOp(value=[1.0], type=toy.Tensor(shape=shape_constant([1])))
    seed = toy.FillLikeOp(fill=one_scalar, template=actual_result, type=_INFERRED())
    accumulate(actual_result, seed)

    # Reverse pass over cloned ops (topological order from block.ops)
    cloned_ops = [val_map[id(op)] for op in body.ops if id(op) in val_map]
    for op in reversed(cloned_ops):
        d_out = adjoints.get(id(op))
        if d_out is None:
            continue

        if isinstance(op, ConstantOp):
            pass

        elif isinstance(op, toy.AddOp):
            accumulate(op.lhs, d_out)
            accumulate(op.rhs, d_out)

        elif isinstance(op, toy.MulOp):
            d_lhs = toy.MulOp(lhs=d_out, rhs=op.rhs, type=_INFERRED())
            accumulate(op.lhs, d_lhs)
            d_rhs = toy.MulOp(lhs=d_out, rhs=op.lhs, type=_INFERRED())
            accumulate(op.rhs, d_rhs)

        elif isinstance(op, toy.TransposeOp):
            accumulate(op.input, toy.TransposeOp(input=d_out, type=_INFERRED()))

        elif isinstance(op, toy.ReshapeOp):
            accumulate(op.input, d_out)

        elif isinstance(op, ChainOp):
            accumulate(op.lhs, d_out)

    # The gradient for the first parameter is the result
    grad_result = adjoints.get(id(grad_args[0]))
    if grad_result is None:
        zero_scalar = ConstantOp(
            value=[0.0], type=toy.Tensor(shape=shape_constant([1]))
        )
        grad_result = toy.FillLikeOp(
            fill=zero_scalar, template=grad_args[0], type=_INFERRED()
        )

    result_type = _INFERRED()
    return FunctionOp(
        name=name,
        result=result_type,
        body=dgen.Block(result=grad_result, args=grad_args),
        type=FunctionType(result=result_type),
    )


def lower_grad(module: Module, func_map: dict[str, FunctionOp]) -> Module:
    """Expand all GradOps in the module into synthesized gradient functions.

    For each GradOp referencing function ``f``:
    1. Synthesize ``f_grad`` as a new FunctionOp
    2. Add it to the module
    3. Replace the GradOp with ``Value(name="f_grad")``

    Helper functions that are only used as grad targets (never called
    directly) are removed from the module to avoid shape-inference
    failures on their unresolved parameter types.

    Returns a new Module with GradOps eliminated and gradient functions added.
    """
    new_functions: list[dgen.Op] = []
    grad_replacements: dict[int, dgen.Value] = {}
    grad_targets: set[str] = set()  # functions used only as grad sources

    # First pass: find all GradOps and synthesize gradient functions
    for func in module.functions:
        for op in func.body.ops:
            if not isinstance(op, GradOp):
                continue
            callee_name = op.callee.name
            if callee_name is None:
                raise RuntimeError("GradOp callee must have a name")
            callee = func_map.get(callee_name)
            if callee is None:
                raise RuntimeError(f"Unknown function in grad: {callee_name}")
            grad_targets.add(callee_name)

            grad_name = f"{callee_name}_grad"
            grad_func = _build_grad_function(grad_name, callee)
            new_functions.append(grad_func)
            # GradOp → name reference to the new function
            grad_replacements[id(op)] = dgen.Value(name=grad_name, type=op.type)

    if not grad_replacements:
        return module

    # Second pass: replace GradOp references in all blocks
    def _replace_in_block(block: dgen.Block) -> None:
        for op in block.ops:
            # Replace in parameters (e.g. CallOp.callee)
            for param_name, param_val in op.parameters:
                replacement = grad_replacements.get(id(param_val))
                if replacement is not None:
                    setattr(op, param_name, replacement)
            # Replace in operands
            for op_name, op_val in op.operands:
                replacement = grad_replacements.get(id(op_val))
                if replacement is not None:
                    setattr(op, op_name, replacement)
            # Recurse into child blocks
            for _, child_block in op.blocks:
                _replace_in_block(child_block)
        # Replace in block result
        replacement = grad_replacements.get(id(block.result))
        if replacement is not None:
            block.result = replacement

    for func in module.functions:
        _replace_in_block(func.body)

    # Collect functions called directly (not just via grad)
    from dgen.dialects.function import CallOp

    directly_called: set[str] = set()
    for func in module.functions:
        for op in func.body.ops:
            if isinstance(op, CallOp) and hasattr(op.callee, "name") and op.callee.name:
                directly_called.add(op.callee.name)

    # Keep module ops, but drop helper functions only used as grad targets
    kept_ops = [
        op
        for op in module.ops
        if not (
            isinstance(op, FunctionOp)
            and op.name in grad_targets
            and op.name not in directly_called
        )
    ]
    return Module(ops=kept_ops + new_functions)
