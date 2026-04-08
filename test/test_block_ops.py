"""Tests for Block.ops: the canonical op list derived from use-def traversal.

Block.ops returns ops reachable from block.result via transitive_dependencies,
filtered to Op instances, in topological order (dependencies before dependents).

Contract:
- Follows operands (Value-typed fields) and parameters (non-Value fields
  that are Values, e.g. branch targets)
- Does NOT descend into nested blocks (op.blocks). A LabelOp is an op in
  the walk, but its body block's ops are a separate walk.
- Stops at captures (capture boundaries). These are leaves.
- Only returns Op instances (not BlockArguments, BlockParameters, type values,
  or plain Values).
- Returns ops in topological order: dependencies before dependents.
"""

from dgen import Block, Value
from dgen.block import BlockArgument
from dgen.dialects import goto
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.module import ConstantOp, pack


def test_simple_chain():
    """Block.ops follows operands in topological order."""
    a = ConstantOp.from_constant(Index().constant(1), name="a")
    b = ConstantOp.from_constant(Index().constant(2), name="b")
    c = ChainOp(name="c", lhs=a, rhs=b, type=Index())
    block = Block(result=c)
    assert list(block.ops) == [a, b, c]


def test_block_args_not_included():
    """BlockArguments are not ops — Block.ops skips them."""
    x = BlockArgument(name="x", type=Index())
    a = ConstantOp.from_constant(Index().constant(1), name="a")
    c = ChainOp(name="c", lhs=x, rhs=a, type=Index())
    block = Block(result=c, args=[x])
    assert list(block.ops) == [a, c]


def test_captures_stop_walk():
    """Captured values are leaves — Block.ops doesn't traverse past them."""
    a = ConstantOp.from_constant(Index().constant(1), name="a")
    b = ConstantOp.from_constant(Index().constant(2), name="b")
    c = ChainOp(name="c", lhs=a, rhs=b, type=Index())
    # Stop at a — it's a capture, not included in results
    block = Block(result=c, captures=[a])
    assert list(block.ops) == [b, c]


def test_does_not_descend_into_label_body():
    """Block.ops reaches a LabelOp but does NOT walk into its body block."""
    inner_op = ConstantOp.from_constant(Index().constant(42), name="inner")
    label = goto.LabelOp(
        initial_arguments=pack(),
        name="lbl",
        body=Block(result=inner_op),
    )
    outer_op = ChainOp(name="outer", lhs=label, rhs=label, type=Nil())
    block = Block(result=outer_op)

    ops = list(block.ops)
    # label is reached (it's an Op), but inner_op is NOT — it's inside
    # the label's body block, which is a separate walk.
    assert label in ops
    assert inner_op not in ops


def test_label_body_is_separate_walk():
    """A label's body.ops is its own walk, independent of the parent."""
    inner_a = ConstantOp.from_constant(Index().constant(1), name="inner_a")
    inner_b = ConstantOp.from_constant(Index().constant(2), name="inner_b")
    inner_result = ChainOp(name="inner_c", lhs=inner_a, rhs=inner_b, type=Nil())
    label = goto.LabelOp(
        initial_arguments=pack(),
        name="lbl",
        body=Block(result=inner_result),
    )
    outer = ChainOp(name="outer", lhs=label, rhs=label, type=Nil())
    block = Block(result=outer)

    # Parent walk: sees label and outer, not inner ops
    parent_ops = list(block.ops)
    assert label in parent_ops
    assert outer in parent_ops
    assert inner_a not in parent_ops
    assert inner_b not in parent_ops

    # Label body walk: only sees inner ops
    body_ops = list(label.body.ops)
    assert set(body_ops) == {inner_a, inner_b, inner_result}

    # No overlap
    assert not set(parent_ops) & set(body_ops)


def test_follows_parameters():
    """Block.ops follows parameter references (e.g. branch targets)."""
    label = goto.LabelOp(
        initial_arguments=pack(),
        name="target",
        body=Block(result=Value(type=Nil())),
    )
    branch = goto.BranchOp(target=label, arguments=pack())
    block = Block(result=branch)

    ops = list(block.ops)
    # label is reached via the 'target' parameter
    assert label in ops
    assert branch in ops


def test_captures_as_parent_dependencies():
    """Block.ops uses captures as stop set — captured values are leaves."""
    outer_val = ConstantOp.from_constant(Index().constant(99), name="outer")
    inner_op = ChainOp(name="use", lhs=outer_val, rhs=outer_val, type=Index())

    # Without captures: walk reaches outer_val
    block_no_captures = Block(result=inner_op)
    assert outer_val in block_no_captures.ops

    # With captures: walk stops at outer_val
    block_with_captures = Block(result=inner_op, captures=[outer_val])
    assert outer_val not in block_with_captures.ops
    assert inner_op in block_with_captures.ops


def test_parent_sees_child_block_captures():
    """Parent Block.ops visits a child block's captures as dependencies.

    When an op has a block with captures, those captures are parent-scope
    values that the op depends on. Block.ops must include them so they appear
    in the parent block's op list.
    """
    captured = ConstantOp.from_constant(Index().constant(42), name="captured")
    inner_result = ConstantOp.from_constant(Index().constant(1), name="inner")
    label = goto.LabelOp(
        initial_arguments=pack(),
        name="lbl",
        body=Block(
            result=ChainOp(lhs=inner_result, rhs=captured, type=Index()),
            captures=[captured],
        ),
    )
    block = Block(result=label)
    ops = list(block.ops)
    # captured is visited as a dependency of the label (via block captures)
    assert captured in ops
    assert label in ops
