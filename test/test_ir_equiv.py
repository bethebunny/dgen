"""Tests for IR graph equivalence checking."""

from dgen.asm.parser import parse_module
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.ir_equiv import Fingerprinter, graph_equivalent, structural_diff
from dgen.module import ConstantOp
from toy.test.helpers import strip_prefix


def test_identical_ops_same_fingerprint():
    """Two independently-constructed identical ops have the same fingerprint."""
    a = ConstantOp(value=42, type=builtin.Index())
    b = ConstantOp(value=42, type=builtin.Index())
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(a) == fingerprinter.fingerprint(b)


def test_different_value_different_fingerprint():
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=2, type=builtin.Index())
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(a) != fingerprinter.fingerprint(b)


def test_different_type_different_fingerprint():
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=1, type=builtin.F64())
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(a) != fingerprinter.fingerprint(b)


def test_op_includes_operands():
    """Ops with different operands fingerprint differently."""
    from dgen.dialects.llvm import AddOp

    x = ConstantOp(value=1, type=builtin.Index())
    y = ConstantOp(value=2, type=builtin.Index())
    z = ConstantOp(value=3, type=builtin.Index())
    add_xy = AddOp(lhs=x, rhs=y)
    add_xz = AddOp(lhs=x, rhs=z)
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(add_xy) != fingerprinter.fingerprint(add_xz)


def test_op_operand_order_matters():
    """add(%x, %y) != add(%y, %x) — operand order is structural."""
    from dgen.dialects.llvm import AddOp

    x = ConstantOp(value=1, type=builtin.Index())
    y = ConstantOp(value=2, type=builtin.Index())
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(AddOp(lhs=x, rhs=y)) != fingerprinter.fingerprint(
        AddOp(lhs=y, rhs=x)
    )


def test_block_arg_fingerprint_by_position():
    """Two block args at the same position with same type have same fingerprint."""
    arg_a = BlockArgument(type=builtin.Index())
    arg_b = BlockArgument(type=builtin.Index())
    fingerprinter = Fingerprinter()
    fingerprinter._arg_positions[id(arg_a)] = 0
    fingerprinter._arg_positions[id(arg_b)] = 0
    assert fingerprinter.fingerprint(arg_a) == fingerprinter.fingerprint(arg_b)


def test_block_arg_different_position_different_fingerprint():
    arg_a = BlockArgument(type=builtin.Index())
    arg_b = BlockArgument(type=builtin.Index())
    fingerprinter = Fingerprinter()
    fingerprinter._arg_positions[id(arg_a)] = 0
    fingerprinter._arg_positions[id(arg_b)] = 1
    assert fingerprinter.fingerprint(arg_a) != fingerprinter.fingerprint(arg_b)


def test_fingerprint_memoized():
    """fingerprint() is called once per object even in a diamond dependency."""
    from dgen.dialects.llvm import AddOp, MulOp

    x = ConstantOp(value=5, type=builtin.Index())
    add = AddOp(lhs=x, rhs=x)
    mul = MulOp(lhs=add, rhs=add)
    fingerprinter = Fingerprinter()
    fingerprinter.fingerprint(mul)
    # x is a shared dependency — fingerprinted once
    assert id(x) in fingerprinter._cache


def test_graph_equivalent_same_ir():
    ir = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    assert graph_equivalent(parse_module(ir), parse_module(ir))


def test_graph_equivalent_different_names():
    """Same computation, different SSA names -> equivalent."""
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %x : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %y : Nil = toy.print(%x)
        |     %_ : Nil = return(%y)
    """)
    assert graph_equivalent(parse_module(a), parse_module(b))


def test_graph_not_equivalent_different_values():
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    assert not graph_equivalent(parse_module(a), parse_module(b))


def test_structural_diff_returns_string():
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ : Nil = return(())
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %_ : Nil = return(())
    """)
    diff = structural_diff(parse_module(a), parse_module(b))
    assert "actual" in diff.lower() or "expected" in diff.lower()
