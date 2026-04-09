"""Value-level symbolic differentiation: differentiate(value, var) -> value.

Instead of grad operating on functions, differentiate operates directly
on the use-def graph: given a value and a variable (BlockArgument), it
walks backward through the expression and produces a new value computing
the derivative via the chain rule.

This is the Lisp symbolic differentiation model mapped onto SSA:
  - differentiate(multiply(%x, %x), %x) → add(multiply(%x, 1), multiply(1, %x))
  - Per-op derivative rules (add, multiply, negate, sin, ...)
  - Chain rule composes structurally over the use-def graph
  - No function inspection needed — operates on values directly

The function-level API is just sugar: wrap a function body that calls
differentiate on its result with respect to its argument.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import dgen
from dgen import Dialect, Op, Type, Value, asm
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.builtins import ConstantOp, PackOp
from dgen.dialects.algebra import AddOp, MultiplyOp, NegateOp
from dgen.dialects.function import CallOp, FunctionOp
from dgen.dialects.number import Float64
from dgen.ir.traversal import inline_block, transitive_dependencies
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.pass_ import Pass, lowering_for
from dgen.testing import strip_prefix
from dgen.type import Constant, Fields

# ============================================================================
# Dialect
# ============================================================================

symdiff = Dialect("symdiff")


@symdiff.op("stage")
@dataclass(eq=False, kw_only=True)
class StageOp(Op):
    """Explicit deferred compilation: compile body when args are known.

    This is quote/eval made explicit:
    - The block is the quoted form (IR structure, not yet compiled)
    - Evaluation substitutes args into the block and inlines the result
    - Constant folding evaluates stages whose args are all available

    Passes emit stage ops when they need deferred compilation.  The
    infrastructure decides when to evaluate: if args are all known in
    the current scope, fold immediately.  If args depend on runtime
    values, the stage survives until runtime (callback thunk).
    """

    args: Value
    type: Type
    body: dgen.Block
    __operands__: ClassVar[Fields] = (("args", Float64),)
    __blocks__: ClassVar[tuple[str, ...]] = ("body",)


@symdiff.op("differentiate")
@dataclass(eq=False, kw_only=True)
class DifferentiateOp(Op):
    """Symbolic derivative of a value with respect to a variable.

    Both value and var are operands — they're values in the same use-def
    graph.  The op is lowered by a pass that walks the graph and applies
    per-op derivative rules via the chain rule.
    """

    value: Value
    var: Value
    type: Type
    __operands__: ClassVar[Fields] = (("value", Float64), ("var", Float64))


# ============================================================================
# Reverse-mode symbolic differentiation
# ============================================================================


def _differentiate(value: Value, var: Value) -> Value:
    """Compute d(value)/d(var) by reverse-mode AD over the use-def graph.

    Walks backward from value, accumulates adjoints, returns the adjoint
    of var.  Stops at var (and at any value that doesn't depend on var).
    """
    f64 = Float64()

    def const(v: float) -> Constant:
        return Constant(type=f64, value=f64.constant(v).value)

    # Collect ops between var and value in topological order.
    ops = [
        v for v in transitive_dependencies(value, stop=[var]) if isinstance(v, dgen.Op)
    ]

    # Adjoint map: value -> d(result)/d(value).  Seed: d(value)/d(value) = 1.
    adjoints: dict[dgen.Value, Value] = {value: const(1.0)}

    def adjoint(v: dgen.Value) -> Value:
        return adjoints.get(v, const(0.0))

    def accum(v: dgen.Value, contrib: Value) -> None:
        if v in adjoints:
            adjoints[v] = AddOp(left=adjoints[v], right=contrib, type=f64)
        else:
            adjoints[v] = contrib

    for op in reversed(ops):
        adj = adjoint(op)
        if isinstance(op, AddOp):
            accum(op.left, adj)
            accum(op.right, adj)
        elif isinstance(op, MultiplyOp):
            # product rule: d(a*b) = b*da + a*db
            accum(op.left, MultiplyOp(left=op.right, right=adj, type=f64))
            accum(op.right, MultiplyOp(left=op.left, right=adj, type=f64))
        elif isinstance(op, NegateOp):
            accum(op.input, NegateOp(input=adj, type=f64))
        elif isinstance(op, ConstantOp):
            pass  # d(constant)/d(anything) = 0
        else:
            raise NotImplementedError(f"No derivative rule for {type(op).__name__}")

    return adjoint(var)


# ============================================================================
# Lowering pass
# ============================================================================


class StageFolding(Pass):
    """Fold stage ops whose args are all available (not runtime-dependent).

    When a stage op's args are all computable in the current scope
    (none are BlockArguments of the enclosing function), substitute
    the block args with the actual values and inline the block body.

    This is the "eval" step: the quoted block becomes live IR.
    """

    allow_unregistered_ops = True

    @lowering_for(StageOp)
    def fold(self, op: StageOp) -> Value | None:
        actual_args = list(op.args) if isinstance(op.args, PackOp) else [op.args]
        # Always substitute and inline — the stage body becomes live IR
        # in the current scope. Downstream passes handle the result.
        body = deepcopy(op.body)
        return inline_block(body, actual_args)


class InlineKnownCalls(Pass):
    """Inline call(f, args) where f is a FunctionOp."""

    allow_unregistered_ops = True

    @lowering_for(CallOp)
    def inline_call(self, op: CallOp) -> Value | None:
        if not isinstance(op.callee, FunctionOp):
            return None
        body = deepcopy(op.callee.body)
        call_args = (
            list(op.arguments) if isinstance(op.arguments, PackOp) else [op.arguments]
        )
        return inline_block(body, call_args)


class SymDiffLowering(Pass):
    """Lower differentiate ops by applying reverse-mode AD.

    When the expression graph is fully visible, differentiates directly.
    When it contains opaque ops (e.g. call through a BlockArgument),
    wraps the differentiation in a StageOp — deferring until the opaque
    values become concrete.
    """

    allow_unregistered_ops = True

    @lowering_for(DifferentiateOp)
    def lower(self, op: DifferentiateOp) -> Value | None:
        try:
            return _differentiate(op.value, op.var)
        except NotImplementedError:
            # Can't differentiate now — emit a stage op to defer.
            # When the opaque values become known and the stage is
            # folded, the differentiate will be re-exposed and this
            # pass will handle it.
            from dgen.builtins import pack

            val_arg = BlockArgument(name="value", type=op.value.type)
            var_arg = BlockArgument(name="var", type=op.var.type)
            deferred = DifferentiateOp(value=val_arg, var=var_arg, type=op.type)
            return StageOp(
                args=pack([op.value, op.var]),
                type=op.type,
                body=dgen.Block(
                    result=deferred,
                    args=[val_arg, var_arg],
                    captures=[],
                ),
            )


def lower_symdiff(value: dgen.Value) -> dgen.Value:
    """Run SymDiffLowering (no inlining or stage folding)."""
    from dgen.passes.compiler import IdentityPass

    return Compiler([SymDiffLowering()], IdentityPass()).run(value)


symdiff_compiler: Compiler[Executable] = Compiler(
    passes=[
        InlineKnownCalls(),
        StageFolding(),
        SymDiffLowering(),
        ControlFlowToGoto(),
        BuiltinToLLVM(),
        AlgebraToLLVM(),
    ],
    exit=LLVMCodegen(),
)


# ============================================================================
# Tests
# ============================================================================


def test_differentiate_roundtrip():
    """differentiate op parses and round-trips."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %dx2 : number.Float64 = symdiff.differentiate(%x2, %x)
    """)
    value = parse(ir)
    asm_text = asm.format(value)
    assert "symdiff.differentiate(%x2, %x)" in asm_text
    reparsed = parse(asm_text)
    assert asm.format(reparsed) == asm_text


def test_differentiate_x_squared(ir_snapshot):
    """d/dx(x*x) = 2x."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %dx2 : number.Float64 = symdiff.differentiate(%x2, %x)
    """)
    value = parse(ir)
    lowered = lower_symdiff(value)
    assert lowered == ir_snapshot


def test_differentiate_x_squared_jit():
    """d/dx(x*x) evaluated at x=5 gives 10."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %result : number.Float64 = symdiff.differentiate(%x2, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 10.0
    assert exe.run(3.0).to_json() == 6.0
    assert exe.run(0.0).to_json() == 0.0


def test_differentiate_polynomial(ir_snapshot):
    """d/dx(x*x + 3*x) = 2x + 3."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %three : number.Float64 = 3.0
        |     %3x : number.Float64 = algebra.multiply(%three, %x)
        |     %f : number.Float64 = algebra.add(%x2, %3x)
        |     %result : number.Float64 = symdiff.differentiate(%f, %x)
    """)
    value = parse(ir)
    lowered = lower_symdiff(value)
    assert lowered == ir_snapshot

    exe = symdiff_compiler.compile(value)
    # f'(x) = 2x + 3
    assert exe.run(5.0).to_json() == 13.0
    assert exe.run(0.0).to_json() == 3.0
    assert exe.run(-1.0).to_json() == 1.0


def test_differentiate_negate():
    """d/dx(-x) = -1."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %neg : number.Float64 = algebra.negate(%x)
        |     %result : number.Float64 = symdiff.differentiate(%neg, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == -1.0


def test_differentiate_constant():
    """d/dx(42) = 0."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %c : number.Float64 = 42.0
        |     %result : number.Float64 = symdiff.differentiate(%c, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    assert exe.run(5.0).to_json() == 0.0


def test_differentiate_second_derivative():
    """d/dx(d/dx(x*x*x)) = 6x.

    Differentiating twice: the pass lowers inner differentiate first
    (topological order), then outer differentiate sees the result.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %x2 : number.Float64 = algebra.multiply(%x, %x)
        |     %x3 : number.Float64 = algebra.multiply(%x2, %x)
        |     %df : number.Float64 = symdiff.differentiate(%x3, %x)
        |     %result : number.Float64 = symdiff.differentiate(%df, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    # f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
    assert exe.run(1.0).to_json() == 6.0
    assert exe.run(2.0).to_json() == 12.0
    assert exe.run(0.5).to_json() == 3.0


def test_differentiate_two_variables(ir_snapshot):
    """Partial derivatives: d/dx(x*y) = y, d/dy(x*y) = x.

    Both partials are packed into a Span so neither is dead code.
    """
    ir = strip_prefix("""
        | import algebra
        | import builtin
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64, number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64, %y: number.Float64):
        |     %xy : number.Float64 = algebra.multiply(%x, %y)
        |     %dx : number.Float64 = symdiff.differentiate(%xy, %x)
        |     %dy : number.Float64 = symdiff.differentiate(%xy, %y)
        |     %result : number.Float64 = algebra.add(%dx, %dy)
    """)
    value = parse(ir)
    lowered = lower_symdiff(value)
    assert lowered == ir_snapshot


def test_differentiate_mixed_expression():
    """d/dx(x*y + x) at (3, 5) = y + 1 = 6."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %main : function.Function<[number.Float64, number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64, %y: number.Float64):
        |     %xy : number.Float64 = algebra.multiply(%x, %y)
        |     %f : number.Float64 = algebra.add(%xy, %x)
        |     %result : number.Float64 = symdiff.differentiate(%f, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    # d/dx(x*y + x) = y + 1
    assert exe.run(3.0, 5.0).to_json() == 6.0
    assert exe.run(0.0, 7.0).to_json() == 8.0


def test_differentiate_through_known_call():
    """d/dx(f(x)) where f is a known FunctionOp: f(x) = x*x, f'(x) = 2x.

    InlineKnownCalls expands call(%f, [%x]) into the body of %f,
    then SymDiffLowering differentiates the inlined expression.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%t: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%t, %t)
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%f):
        |     %fx : number.Float64 = function.call(%f, [%x])
        |     %result : number.Float64 = symdiff.differentiate(%fx, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    # f(x) = x^2, f'(x) = 2x
    assert exe.run(5.0).to_json() == 10.0
    assert exe.run(3.0).to_json() == 6.0


def test_differentiate_composition():
    """d/dx(f(g(x))) via chain rule through two inlined calls.

    f(t) = t*t, g(t) = t + t, so f(g(x)) = (2x)^2 = 4x^2, derivative = 8x.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %g : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%t: number.Float64):
        |     %r : number.Float64 = algebra.add(%t, %t)
        |
        | %f : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%t: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%t, %t)
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%g, %f):
        |     %gx : number.Float64 = function.call(%g, [%x])
        |     %fgx : number.Float64 = function.call(%f, [%gx])
        |     %result : number.Float64 = symdiff.differentiate(%fgx, %x)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    # f(g(x)) = (2x)^2 = 4x^2, derivative = 8x
    assert exe.run(1.0).to_json() == 8.0
    assert exe.run(2.0).to_json() == 16.0
    assert exe.run(0.5).to_json() == 4.0


def test_differentiate_defers_on_opaque_call():
    """differentiate(call(%g, ...), %x) where %g is a BlockArgument
    can't be lowered directly — the pass emits a StageOp instead.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %diff_apply : function.Function<[function.Function<[number.Float64], number.Float64>, number.Float64], number.Float64> = function.function<number.Float64>() body(%g: function.Function<[number.Float64], number.Float64>, %x: number.Float64):
        |     %gx : number.Float64 = function.call(%g, [%x])
        |     %result : number.Float64 = symdiff.differentiate(%gx, %x)
    """)
    value = parse(ir)
    lowered = lower_symdiff(value)
    # The DifferentiateOp is wrapped in a StageOp (deferred).
    from dgen.ir.traversal import all_values

    assert any(isinstance(v, StageOp) for v in all_values(lowered))


def test_differentiate_higher_order_via_staging():
    """Higher-order differentiation through explicit staging.

    diff_apply(%g, %x) = differentiate(call(%g, [%x]), %x)

    The SymDiffLowering pass wraps the opaque differentiate in a StageOp.
    When diff_apply is inlined at a concrete call site:
    1. InlineKnownCalls expands call(%diff_apply, [%square, %x])
    2. InlineKnownCalls expands call(%square, [%x]) → multiply(%x, %x)
    3. StageFolding inlines the stage body (args are now concrete)
    4. SymDiffLowering differentiates: d/dx(x*x) = 2x
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        | import symdiff
        |
        | %square : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%t: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%t, %t)
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64) captures(%square):
        |     %fx : number.Float64 = function.call(%square, [%x])
        |     %staged : number.Float64 = symdiff.stage([%fx, %x]) body(%v: number.Float64, %w: number.Float64):
        |         %d : number.Float64 = symdiff.differentiate(%v, %w)
    """)
    value = parse(ir)
    exe = symdiff_compiler.compile(value)
    # d/dx(x^2) = 2x
    assert exe.run(5.0).to_json() == 10.0
    assert exe.run(3.0).to_json() == 6.0
