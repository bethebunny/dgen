"""Tests for the Pass base class, Rewriter, and Pass.run."""

import pytest

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import toy
from toy.test.helpers import strip_prefix


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


def test_rewriter_eager_replace():
    """replace_uses eagerly updates all referencing ops."""
    ir_text = strip_prefix("""
        | import llvm
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %2 : Index = llvm.add(%0, %0)
        |     %_ : Nil = return(%2)
    """)
    m = parse_module(ir_text)
    func = m.functions[0]
    old = func.body.ops[0]  # %0 = 1
    new = func.body.ops[1]  # %1 = 2

    rewriter = Rewriter(func.body)
    rewriter.replace_uses(old, new)

    result = asm.format(m)
    expected = strip_prefix("""
        | import llvm
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %2 : Index = llvm.add(%1, %1)
        |     %_ : Nil = return(%2)
    """)
    assert result == expected


# ---------------------------------------------------------------------------
# Task 7: Pass.run — walk graph, dispatch handlers
# ---------------------------------------------------------------------------


def test_pass_run_eliminates_double_transpose():
    """A pass that eliminates transpose(transpose(x)) -> x."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<[2, 3], F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%2)
        |     %_ : Nil = return(%3)
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
    formatted = asm.format(result)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : Nil = toy.print(%0)
        |     %_ : Nil = return(%3)
    """)
    assert formatted == expected


def test_pass_unregistered_ops_error():
    """allow_unregistered_ops=False raises on unhandled ops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
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
        |     %_ : Nil = return(%0)
    """)
    m = parse_module(ir_text)
    MultiPass().run(m)
    assert call_log == ["first"]


# ---------------------------------------------------------------------------
# Task 8: PassManager
# ---------------------------------------------------------------------------


def test_pass_manager_verification_catches_range_violation():
    """Post-condition check detects ops outside the declared range."""
    from dgen.passes.pass_manager import PassManager

    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %2 : Nil = toy.print(%1)
        |     %_ : Nil = return(%2)
    """)

    class StrictPass(Pass):
        op_domain = {*toy.toy.ops.values(), ConstantOp, builtin.ReturnOp}
        op_range = {ConstantOp, builtin.ReturnOp}  # TransposeOp NOT in range
        type_domain: set[type] = set()
        type_range: set[type] = set()
        allow_unregistered_ops = True

    m = parse_module(ir_text)
    pm = PassManager([StrictPass()], verify=True)
    with pytest.raises(AssertionError):
        pm.run(m)
