"""Autodiff: symbolic differentiation via a grad op and lowering pass.

Defines an 'autodiff' dialect with a single op:
  - grad<f: Function>() -> Function

The GradOp takes a function as a compile-time parameter (it needs to
inspect the body to differentiate it) and returns a new function that
computes the derivative.  The AutodiffLowering pass symbolically
differentiates the function body using reverse-mode AD over the algebra
dialect ops (add, multiply, negate, constants).

This demonstrates the design principle from the function-params change:
  - grad<f> is correctly a parameter (needs to inspect f's body)
  - call(grad_result, args) uses the result as an operand (just needs
    the function pointer)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import dgen
from dgen import Dialect, Op, Type, Value, asm
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.builtins import ConstantOp
from dgen.dialects.algebra import AddOp, MultiplyOp, NegateOp
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.number import Float64
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.pass_ import Pass, lowering_for
from dgen.passes.staging import compute_stages
from dgen.testing import strip_prefix
from dgen.type import Constant, Fields

# ============================================================================
# Dialect
# ============================================================================

autodiff = Dialect("autodiff")


@autodiff.op("grad")
@dataclass(eq=False, kw_only=True)
class GradOp(Op):
    """Compute the gradient of a scalar function.

    The input function must be Function<[Float64], Float64>.
    The result is a new Function<[Float64], Float64> computing the derivative.

    f is a parameter because grad needs to inspect the function body
    to symbolically differentiate it.
    """

    f: Value[Function]
    type: Type
    __params__: ClassVar[Fields] = (("f", Function),)


# ============================================================================
# Reverse-mode AD
# ============================================================================


def _differentiate_body(func: FunctionOp) -> FunctionOp:
    """Build a new FunctionOp that computes df/dx for a scalar function.

    Uses reverse-mode AD:
    1. Walk the function body in topological order (forward pass)
    2. Assign adjoint values in reverse order (backward pass)
    3. The adjoint of the input argument is the derivative

    Supported ops: AddOp, MultiplyOp, NegateOp, ConstantOp.
    """
    body = deepcopy(func.body)
    assert len(body.args) == 1, "grad only supports single-argument functions"
    x = body.args[0]
    result = body.result

    f64 = Float64()
    ops = [v for v in body.values if isinstance(v, dgen.Op)]

    # Adjoint map: value -> its adjoint (dL/d_value).  Seed: d_result = 1.0.
    adjoints: dict[dgen.Value, Value] = {
        result: Constant(type=f64, value=f64.constant(1.0).value)
    }

    def get_adjoint(v: dgen.Value) -> Value:
        return adjoints.get(v, Constant(type=f64, value=f64.constant(0.0).value))

    def accum(v: dgen.Value, contrib: Value) -> None:
        if v in adjoints:
            adjoints[v] = AddOp(left=adjoints[v], right=contrib, type=f64)
        else:
            adjoints[v] = contrib

    # Reverse pass: walk ops from output toward input.
    for op in reversed(ops):
        adj = get_adjoint(op)
        if isinstance(op, AddOp):
            # d/d_left(left + right) = 1, d/d_right(left + right) = 1
            accum(op.left, adj)
            accum(op.right, adj)
        elif isinstance(op, MultiplyOp):
            # d/d_left(left * right) = right * adj
            # d/d_right(left * right) = left * adj
            accum(op.left, MultiplyOp(left=op.right, right=adj, type=f64))
            accum(op.right, MultiplyOp(left=op.left, right=adj, type=f64))
        elif isinstance(op, NegateOp):
            # d/d_input(-input) = -adj
            accum(op.input, NegateOp(input=adj, type=f64))
        elif isinstance(op, ConstantOp):
            pass  # Constants have zero derivative, nothing to propagate.
        else:
            raise NotImplementedError(
                f"Autodiff not implemented for {type(op).__name__}"
            )

    dx = get_adjoint(x)

    # Build the derivative function: same signature, returns dx.
    grad_arg = BlockArgument(name="x", type=f64)
    grad_body = dgen.Block(result=dx, args=[grad_arg])
    # Rewrite: replace the original input with the new block argument.
    grad_body.replace_uses_of(x, grad_arg)
    return FunctionOp(
        name=f"grad_{func.name}" if func.name else None,
        body=grad_body,
        result_type=f64,
        type=func.type,
    )


# ============================================================================
# Lowering pass
# ============================================================================


class AutodiffLowering(Pass):
    """Lower grad ops by symbolically differentiating the target function."""

    allow_unregistered_ops = True

    @lowering_for(GradOp)
    def lower_grad(self, op: GradOp) -> Value | None:
        if not isinstance(op.f, FunctionOp):
            return None
        return _differentiate_body(op.f)


autodiff_compiler: Compiler[Executable] = Compiler(
    passes=[
        AutodiffLowering(),
        ControlFlowToGoto(),
        BuiltinToLLVM(),
        AlgebraToLLVM(),
    ],
    exit=LLVMCodegen(),
)


# ============================================================================
# Tests
# ============================================================================


def test_grad_op_roundtrip():
    """grad op parses and round-trips."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%x, %x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
    """)
    value = parse(ir)
    asm_text = asm.format(value)
    assert "autodiff.grad<%f>()" in asm_text
    reparsed = parse(asm_text)
    assert asm.format(reparsed) == asm_text


def test_grad_identity():
    """grad(x -> x + 0) = x -> 1."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.add(%x, number.Float64(0.0))
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 1.0
    assert exe.run(0.0).to_json() == 1.0


def test_grad_constant():
    """grad(x -> 42) = x -> 0."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.add(number.Float64(42.0), number.Float64(0.0))
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 0.0
    assert exe.run(100.0).to_json() == 0.0


def test_grad_add():
    """grad(x -> x + x) = x -> 2."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.add(%x, %x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 2.0


def test_grad_multiply():
    """grad(x -> x * x) = x -> 2x."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%x, %x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 10.0
    assert exe.run(3.0).to_json() == 6.0
    assert exe.run(0.0).to_json() == 0.0


def test_grad_negate():
    """grad(x -> -x) = x -> -1."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.negate(%x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == -1.0


def test_grad_polynomial():
    """grad(x -> x*x + 3*x) = x -> 2x + 3, evaluated at x=5 gives 13."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %three : number.Float64 = 3.0
        |     %3x : number.Float64 = algebra.multiply(%three, %x)
        |     %r : number.Float64 = algebra.add(%x2, %3x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df):
        |     %result : number.Float64 = function.call(%df, [%x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    # f'(x) = 2x + 3
    assert exe.run(5.0).to_json() == 13.0
    assert exe.run(0.0).to_json() == 3.0
    assert exe.run(-1.0).to_json() == 1.0


def test_grad_stage_computation():
    """grad<f> where f is a FunctionOp is stage 0 (no runtime dependency).

    The grad op has f as a parameter (needs to inspect the body), but
    since f is a FunctionOp (stage 0), no stage bump occurs.  The call
    to the derivative uses it as an operand, so no +1 bump there either.
    """
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%x, %x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
    """)
    value = parse(ir)
    stages = compute_stages(value)
    stage_by_name = {v.name: s for v, s in stages.items() if v.name}
    # f is a FunctionOp → stage 0.  grad<f> has f as a parameter, but
    # since stage(f) == 0 there is no +1 bump: stage(grad) = max(0) = 0.
    assert stage_by_name["f"] == 0
    assert stage_by_name["df"] == 0


def test_grad_composes_with_indirect_call():
    """grad result flows as an operand through an indirect-call apply function."""
    ir = strip_prefix("""
        | import algebra
        | import autodiff
        | import function
        | import number
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%x, %x)
        |
        | %df : function.Function<[number.Float64], number.Float64> = autodiff.grad<%f>()
        |
        | %apply : function.Function<[function.Function<[number.Float64], number.Float64>, number.Float64], number.Float64> = function.function<number.Float64>() body(%g: function.Function<[number.Float64], number.Float64>, %x: number.Float64):
        |     %result : number.Float64 = function.call(%g, [%x])
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%df, %apply):
        |     %result : number.Float64 = function.call(%apply, [%df, %x])
    """)
    value = parse(ir)
    exe = autodiff_compiler.compile(value)
    # f'(x) = 2x
    assert exe.run(5.0).to_json() == 10.0
    assert exe.run(3.0).to_json() == 6.0
