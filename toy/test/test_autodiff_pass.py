"""Tests for the Autodiff pass: IR in, IR out.

Constructs IR directly, runs the Autodiff pass, and checks the
synthesized gradient function structure.
"""

import dgen
from dgen.block import BlockArgument
from dgen.compiler import Compiler, IdentityPass, verify_passes
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp, Module
from toy.dialects import shape_constant, toy
from toy.dialects.diff import GradOp
from toy.passes.autodiff import Autodiff


def _run_autodiff(module: Module) -> Module:
    """Run just the Autodiff pass with verification disabled."""
    token = verify_passes.set(False)
    try:
        result = Autodiff().run(module, Compiler(passes=[], exit=IdentityPass()))
    finally:
        verify_passes.reset(token)
    return result


def _make_func(name: str, body_fn):
    """Build a FunctionOp with one parameter, body_fn(arg) -> result op."""
    arg = BlockArgument(name="x", type=toy.InferredShapeTensor())
    result = body_fn(arg)
    return FunctionOp(
        name=name,
        result=toy.InferredShapeTensor(),
        body=dgen.Block(result=result, args=[arg]),
        type=FunctionType(result=toy.InferredShapeTensor()),
    )


def _make_grad_module(func: FunctionOp) -> Module:
    """Build a module with a GradOp applied to func."""
    grad = GradOp(
        function=func,
        type=FunctionType(result=toy.InferredShapeTensor()),
    )
    # A trivial main that just holds the GradOp as its result
    main = FunctionOp(
        name="main",
        result=grad.type,
        body=dgen.Block(result=grad, args=[]),
        type=FunctionType(result=grad.type),
    )
    return Module(ops=[func, main])


def _find_func(module: Module, name: str) -> FunctionOp:
    for f in module.functions:
        if f.name == name:
            return f
    raise KeyError(name)


# === f(x) = x + x => f'(x) = 2 ===


def test_grad_add():
    """f(x) = x + x. Gradient should be seed + seed."""
    func = _make_func(
        "f", lambda x: toy.AddOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")
    ops = grad_fn.body.ops
    op_types = [type(op).__name__ for op in ops]

    # Should contain: ConstantOp (seed scalar), AddOp (clone of forward),
    # FillLikeOp (broadcast seed), AddOp (seed + seed)
    assert "ConstantOp" in op_types
    assert "FillLikeOp" in op_types
    assert op_types.count("AddOp") == 2  # forward clone + gradient accumulation

    # The result should be an AddOp (seed + seed)
    assert isinstance(grad_fn.body.result, toy.AddOp)


# === f(x) = x * x => f'(x) = 2x ===


def test_grad_mul():
    """f(x) = x * x. Gradient should be seed*x + seed*x = 2*seed*x."""
    func = _make_func(
        "f", lambda x: toy.MulOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")
    ops = grad_fn.body.ops
    op_types = [type(op).__name__ for op in ops]

    # Forward: MulOp. Backward: FillLikeOp, MulOp (d*rhs), MulOp (d*lhs), AddOp
    assert "MulOp" in op_types
    assert "FillLikeOp" in op_types
    assert "AddOp" in op_types
    # 1 forward mul + 2 gradient muls = 3
    assert op_types.count("MulOp") == 3

    # Result should be an AddOp (sum of the two partials)
    assert isinstance(grad_fn.body.result, toy.AddOp)


# === f(x) = x * x + x => f'(x) = 2x + 1 ===


def test_grad_polynomial():
    """f(x) = x*x + x. Gradient has both mul and add adjoint rules."""

    def body(x):
        xx = toy.MulOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
        return toy.AddOp(lhs=xx, rhs=x, type=toy.InferredShapeTensor())

    func = _make_func("f", body)
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")
    ops = grad_fn.body.ops

    # Should have forward ops (MulOp, AddOp) plus gradient ops
    op_types = [type(op).__name__ for op in ops]
    assert "MulOp" in op_types
    assert "FillLikeOp" in op_types

    # Result is an AddOp (accumulation of partials)
    assert isinstance(grad_fn.body.result, toy.AddOp)


# === f(x) = transpose(x) * transpose(x) => grad uses transpose in backward ===


def test_grad_transpose():
    """Transpose adjoint is transpose."""

    def body(x):
        t = toy.TransposeOp(input=x, type=toy.InferredShapeTensor())
        return toy.MulOp(lhs=t, rhs=t, type=toy.InferredShapeTensor())

    func = _make_func("f", body)
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")
    ops = grad_fn.body.ops
    op_types = [type(op).__name__ for op in ops]

    # Backward should contain TransposeOp (adjoint of transpose)
    # Forward: 1 transpose + 1 mul. Backward: 2 muls + 1 transpose + adds
    assert op_types.count("TransposeOp") >= 2  # forward + backward


# === f(x) = x + c (constant) => f'(x) = fill_like(1, ...) ===


def test_grad_constant_passthrough():
    """Adding a constant: gradient w.r.t. x is just the seed (constant has no grad)."""

    def body(x):
        c = ConstantOp(
            value=[1.0, 1.0, 1.0], type=toy.Tensor(shape=shape_constant([3]))
        )
        return toy.AddOp(lhs=x, rhs=c, type=toy.InferredShapeTensor())

    func = _make_func("f", body)
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")

    # Result should be FillLikeOp — seed flows through to x, constant gets nothing
    assert isinstance(grad_fn.body.result, toy.FillLikeOp)


# === f(x) = c * x => f'(x) = c ===


def test_grad_mul_by_constant():
    """Multiplying by constant: gradient is seed * c."""

    def body(x):
        c = ConstantOp(value=[2.0, 3.0], type=toy.Tensor(shape=shape_constant([2])))
        return toy.MulOp(lhs=x, rhs=c, type=toy.InferredShapeTensor())

    func = _make_func("f", body)
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    grad_fn = _find_func(result, "f_grad")

    # Result should be MulOp — seed * c
    assert isinstance(grad_fn.body.result, toy.MulOp)


# === Module structure ===


def test_grad_produces_new_function():
    """The pass adds f_grad to the module."""
    func = _make_func(
        "f", lambda x: toy.AddOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    names = [f.name for f in result.functions]
    assert "f_grad" in names


def test_grad_function_has_same_arity():
    """f_grad has the same number of block args as f."""
    func = _make_func(
        "f", lambda x: toy.AddOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    result = _run_autodiff(module)

    f_grad = _find_func(result, "f_grad")
    assert len(f_grad.body.args) == len(func.body.args)


def test_no_grad_ops_passthrough():
    """Module without GradOps passes through unchanged."""
    func = _make_func(
        "f", lambda x: toy.AddOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = Module(ops=[func])
    result = _run_autodiff(module)
    assert result is module


# === IR snapshot tests ===


def test_grad_add_ir(ir_snapshot):
    func = _make_func(
        "f", lambda x: toy.AddOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    assert _run_autodiff(module) == ir_snapshot


def test_grad_mul_ir(ir_snapshot):
    func = _make_func(
        "f", lambda x: toy.MulOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
    )
    module = _make_grad_module(func)
    assert _run_autodiff(module) == ir_snapshot


def test_grad_polynomial_ir(ir_snapshot):
    def body(x):
        xx = toy.MulOp(lhs=x, rhs=x, type=toy.InferredShapeTensor())
        return toy.AddOp(lhs=xx, rhs=x, type=toy.InferredShapeTensor())

    func = _make_func("f", body)
    module = _make_grad_module(func)
    assert _run_autodiff(module) == ir_snapshot
