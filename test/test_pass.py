"""Tests for the Pass base class, Rewriter, and Pass.run."""

import pytest

from dgen.asm.parser import parse_module
from dgen.dialects import builtin
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import toy
from dgen.testing import strip_prefix


# ---------------------------------------------------------------------------
# Task 5: @lowering_for handler registration
# ---------------------------------------------------------------------------


def test_lowering_for_registers_handler():
    class MyPass(Pass):
        op_domain: set[type] = set()
        op_range: set[type] = set()
        type_domain: set[type] = set()
        type_range: set[type] = set()

        @lowering_for(ConstantOp)
        def handle_constant(self, op: ConstantOp, rewriter: Rewriter) -> bool:
            return False

    assert ConstantOp in MyPass._handlers
    assert len(MyPass._handlers[ConstantOp]) == 1


def test_multiple_handlers_per_op_type():
    class MyPass(Pass):
        op_domain: set[type] = set()
        op_range: set[type] = set()
        type_domain: set[type] = set()
        type_range: set[type] = set()

        @lowering_for(ConstantOp)
        def handler_a(self, op: ConstantOp, rewriter: Rewriter) -> bool:
            return False

        @lowering_for(ConstantOp)
        def handler_b(self, op: ConstantOp, rewriter: Rewriter) -> bool:
            return True

    assert len(MyPass._handlers[ConstantOp]) == 2
    names = [h.__name__ for h in MyPass._handlers[ConstantOp]]
    assert names == ["handler_a", "handler_b"]


# ---------------------------------------------------------------------------
# Task 6: Rewriter with eager replace_uses
# ---------------------------------------------------------------------------


def test_rewriter_eager_replace(ir_snapshot):
    """replace_uses eagerly updates all referencing ops."""
    ir_text = strip_prefix("""
        | import llvm
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %2 : Index = llvm.add(%0, %0)
        |     %3 : Index = chain(%2, %1)
    """)
    m = parse_module(ir_text)
    func = m.functions[0]
    ops_by_name = {op.name: op for op in func.body.ops}
    old = ops_by_name["0"]  # %0 = 1
    new = ops_by_name["1"]  # %1 = 2

    rewriter = Rewriter(func.body)
    rewriter.replace_uses(old, new)

    assert m == ir_snapshot


# ---------------------------------------------------------------------------
# Task 7: Pass.run — walk graph, dispatch handlers
# ---------------------------------------------------------------------------


def test_pass_run_eliminates_double_transpose(ir_snapshot):
    """A pass that eliminates transpose(transpose(x)) -> x."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%2)
    """)

    class ElimTranspose(Pass):
        allow_unregistered_ops = True

        @lowering_for(toy.TransposeOp)
        def eliminate(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
            if not isinstance(op.input, toy.TransposeOp):
                return False
            rewriter.replace_uses(op, op.input.input)
            return True

    m = parse_module(ir_text)
    result = ElimTranspose().run(m)
    assert result == ir_snapshot


def test_pass_unregistered_ops_error():
    """allow_unregistered_ops=False raises on unhandled ops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)

    class StrictPass(Pass):
        allow_unregistered_ops = False

    m = parse_module(ir_text)
    with pytest.raises(TypeError, match="No handler for"):
        StrictPass().run(m)


def test_pass_multiple_handlers_first_wins():
    """Multiple handlers: first one that returns True wins."""
    call_log: list[str] = []

    class MultiPass(Pass):
        allow_unregistered_ops = True

        @lowering_for(ConstantOp)
        def first(self, op: ConstantOp, rewriter: Rewriter) -> bool:
            call_log.append("first")
            return True

        @lowering_for(ConstantOp)
        def second(self, op: ConstantOp, rewriter: Rewriter) -> bool:
            call_log.append("second")
            return True

    ir_text = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 42
    """)
    m = parse_module(ir_text)
    MultiPass().run(m)
    assert call_log == ["first"]


# ---------------------------------------------------------------------------
# Task 8: Compiler.run with verification
# ---------------------------------------------------------------------------


def test_compiler_run_verification_catches_range_violation():
    """Post-condition check detects ops outside the declared range."""
    from dgen.codegen import LLVMCodegen
    from dgen.compiler import Compiler

    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%0)
        |     %2 : Nil = toy.print(%1)
    """)

    class StrictPass(Pass):
        op_domain = {*toy.toy.ops.values(), ConstantOp}
        op_range = {ConstantOp}  # TransposeOp NOT in range
        type_domain: set[type] = set()
        type_range: set[type] = set()
        allow_unregistered_ops = True

    m = parse_module(ir_text)
    compiler = Compiler(passes=[StrictPass()], exit=LLVMCodegen())
    with pytest.raises(AssertionError):
        compiler.run(m, verify=True)
