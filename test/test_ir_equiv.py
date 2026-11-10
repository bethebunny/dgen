"""Tests for IR graph equivalence checking."""

from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.dialects import builtin, number
from dgen.dialects.llvm import AddOp, MulOp
from dgen.ir_diff import structural_diff
from dgen.ir_equiv import Fingerprinter, graph_equivalent
from dgen.module import ConstantOp
from dgen.testing import strip_prefix
from dgen.dialects.ndbuffer import NDBuffer, Shape


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
    b = ConstantOp(value=1, type=number.Float64())
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(a) != fingerprinter.fingerprint(b)


def test_op_includes_operands():
    """Ops with different operands fingerprint differently."""
    x = ConstantOp(value=1, type=builtin.Index())
    y = ConstantOp(value=2, type=builtin.Index())
    z = ConstantOp(value=3, type=builtin.Index())
    add_xy = AddOp(lhs=x, rhs=y)
    add_xz = AddOp(lhs=x, rhs=z)
    fingerprinter = Fingerprinter()
    assert fingerprinter.fingerprint(add_xy) != fingerprinter.fingerprint(add_xz)


def test_op_operand_order_matters():
    """add(%x, %y) != add(%y, %x) — operand order is structural."""
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
    fingerprinter._arg_positions[arg_a] = 0
    fingerprinter._arg_positions[arg_b] = 0
    assert fingerprinter.fingerprint(arg_a) == fingerprinter.fingerprint(arg_b)


def test_block_arg_different_position_different_fingerprint():
    arg_a = BlockArgument(type=builtin.Index())
    arg_b = BlockArgument(type=builtin.Index())
    fingerprinter = Fingerprinter()
    fingerprinter._arg_positions[arg_a] = 0
    fingerprinter._arg_positions[arg_b] = 1
    assert fingerprinter.fingerprint(arg_a) != fingerprinter.fingerprint(arg_b)


def test_fingerprint_memoized():
    """fingerprint() is called once per object even in a diamond dependency."""
    x = ConstantOp(value=5, type=builtin.Index())
    add = AddOp(lhs=x, rhs=x)
    mul = MulOp(lhs=add, rhs=add)
    fingerprinter = Fingerprinter()
    fingerprinter.fingerprint(mul)
    # x is a shared dependency — fingerprinted once
    assert x in fingerprinter._cache


def test_graph_equivalent_same_ir():
    ir = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert graph_equivalent(parse(ir), parse(ir))


def test_graph_equivalent_different_names():
    """Same computation, different SSA names -> equivalent."""
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %x : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %y : Nil = toy.print(%x)
    """)
    assert graph_equivalent(parse(a), parse(b))


def test_graph_not_equivalent_different_values():
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert not graph_equivalent(parse(a), parse(b))


def test_structural_diff_returns_string():
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    diff = structural_diff(parse(a), parse(b))
    assert "-" in diff and "+" in diff


def test_type_constant_with_dynamic_layout_param():
    """Type.__constant__ works for a param type whose layout depends on its own params."""
    rank = ConstantOp(value=2, type=builtin.Index())
    shape = Shape(rank=rank)
    memref_type = NDBuffer(shape=shape, dtype=number.Float64())
    shape_json = {"tag": "ndbuffer.Shape", "rank": 2}
    assert memref_type.__constant__.to_json()["shape"] == shape_json
