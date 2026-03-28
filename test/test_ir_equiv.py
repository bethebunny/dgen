"""Tests for IR graph equivalence checking."""

import pytest

from dgen.asm.parser import parse_module
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
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert graph_equivalent(parse_module(ir), parse_module(ir))


def test_graph_equivalent_different_names():
    """Same computation, different SSA names -> equivalent."""
    a = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %x : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %y : Nil = toy.print(%x)
    """)
    assert graph_equivalent(parse_module(a), parse_module(b))


def test_graph_not_equivalent_different_values():
    a = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert not graph_equivalent(parse_module(a), parse_module(b))


def test_structural_diff_returns_string():
    a = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    diff = structural_diff(parse_module(a), parse_module(b))
    assert "-" in diff and "+" in diff


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TypeValue._resolve_layout constructs param types without arguments to get "
        "their layout (e.g. Shape().__layout__), but Shape.__layout__ reads "
        "self.rank — so Shape() raises TypeError: missing required arg 'rank'. "
        "Fix: _resolve_layout needs a way to get the layout of a param type "
        "without constructing an instance, e.g. a classmethod or a sentinel instance."
    ),
)
def test_type_constant_with_dynamic_layout_param():
    """Type.__constant__ fails for types whose layout depends on a param type with a dynamic layout.

    NDBuffer.__params__ = (("shape", Shape), ("dtype", TypeType)).
    _resolve_layout("ndbuffer.NDBuffer") calls Shape().__layout__ to determine how to
    serialize the "shape" field, but Shape.__layout__ is Array(Index.__layout__,
    self.rank.to_json()), which requires a concrete self.rank.
    """
    rank = ConstantOp(value=2, type=builtin.Index())
    shape = Shape(rank=rank)
    memref_type = NDBuffer(shape=shape, dtype=number.Float64())
    # Triggers TypeValue._resolve_layout("ndbuffer.NDBuffer") -> Shape().__layout__ -> TypeError
    _ = memref_type.__constant__.to_json()
