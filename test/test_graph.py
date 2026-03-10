"""Tests for use-def graph utilities."""

from dgen.dialects import builtin, llvm
from dgen.graph import walk_ops
from dgen.module import ConstantOp


def test_walk_ops_linear_chain():
    """Walk a simple linear dependency chain."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=2, type=builtin.Index())
    c = llvm.AddOp(lhs=a, rhs=b)
    ops = walk_ops(c)
    assert ops[-1] is c
    assert set(ops) == {a, b, c}


def test_walk_ops_diamond():
    """Diamond dependency: a used by both b and c, both used by d."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = llvm.AddOp(lhs=a, rhs=a)
    c = llvm.MulOp(lhs=a, rhs=a)
    d = llvm.AddOp(lhs=b, rhs=c)
    ops = walk_ops(d)
    assert ops[0] is a
    assert ops[-1] is d
    assert len(ops) == 4


def test_walk_ops_skips_block_args():
    """BlockArguments are not ops and should not appear in the result."""
    from dgen.block import BlockArgument

    arg = BlockArgument(type=builtin.Index())
    op = llvm.AddOp(lhs=arg, rhs=arg)
    ops = walk_ops(op)
    assert ops == [op]


def test_walk_ops_does_not_descend_into_blocks():
    """Ops nested inside another op's block are not included."""
    import dgen

    inner = ConstantOp(value=42, type=builtin.Index())
    func = builtin.FunctionOp(
        name="f",
        body=dgen.Block(ops=[inner], args=[]),
        result=builtin.Nil(),
    )
    ops = walk_ops(func)
    assert func in ops
    assert inner not in ops
