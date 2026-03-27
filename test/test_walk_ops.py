"""Tests for walk_ops: the use-def graph traversal.

walk_ops(root, stop) returns ops in topological order (dependencies before
dependents) by walking the use-def graph backwards from root.

Contract:
- Follows operands (Value-typed fields) and parameters (non-Value fields
  that are Values, e.g. branch targets)
- Does NOT descend into nested blocks (op.blocks). A LabelOp is an op in
  the walk, but its body block's ops are a separate walk.
- Stops at values in the stop set (capture boundaries). These are leaves.
- Only returns Op instances (not BlockArguments, BlockParameters, Constants,
  or plain Values).
- Returns ops in topological order: dependencies before dependents.
"""

import dgen
from dgen import Block, Value
from dgen.block import BlockArgument
from dgen.dialects import builtin, goto
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.graph import walk_ops
from dgen.module import ConstantOp, PackOp


def test_simple_chain():
    """walk_ops follows operands in topological order."""
    a = ConstantOp(name="a", value=1, type=Index())
    b = ConstantOp(name="b", value=2, type=Index())
    c = ChainOp(name="c", lhs=a, rhs=b, type=Index())
    ops = walk_ops(c)
    assert ops == [a, b, c]


def test_block_args_not_included():
    """BlockArguments are not ops — walk_ops skips them."""
    x = BlockArgument(name="x", type=Index())
    a = ConstantOp(name="a", value=1, type=Index())
    c = ChainOp(name="c", lhs=x, rhs=a, type=Index())
    ops = walk_ops(c)
    # x is a BlockArgument, not an Op — not in the result
    assert ops == [a, c]


def test_stop_set():
    """Values in stop set are leaves — walk_ops doesn't traverse past them."""
    a = ConstantOp(name="a", value=1, type=Index())
    b = ConstantOp(name="b", value=2, type=Index())
    c = ChainOp(name="c", lhs=a, rhs=b, type=Index())
    # Stop at a — it becomes a leaf, not included in results
    ops = walk_ops(c, stop={a})
    assert ops == [b, c]


def test_does_not_descend_into_label_body():
    """walk_ops reaches a LabelOp but does NOT walk into its body block."""
    inner_op = ConstantOp(name="inner", value=42, type=Index())
    label = goto.LabelOp(
        initial_arguments=PackOp(values=[], type=builtin.List(element_type=Nil())),
        name="lbl",
        body=Block(result=inner_op),
    )
    outer_op = ChainOp(name="outer", lhs=label, rhs=label, type=Nil())

    ops = walk_ops(outer_op)
    # label is reached (it's an Op), but inner_op is NOT — it's inside
    # the label's body block, which is a separate walk.
    assert label in ops
    assert inner_op not in ops


def test_label_body_is_separate_walk():
    """A label's body.ops is its own walk, independent of the parent."""
    inner_a = ConstantOp(name="inner_a", value=1, type=Index())
    inner_b = ConstantOp(name="inner_b", value=2, type=Index())
    inner_result = ChainOp(name="inner_c", lhs=inner_a, rhs=inner_b, type=Nil())
    label = goto.LabelOp(
        initial_arguments=PackOp(values=[], type=builtin.List(element_type=Nil())),
        name="lbl",
        body=Block(result=inner_result),
    )
    outer = ChainOp(name="outer", lhs=label, rhs=label, type=Nil())

    # Parent walk: sees label, its initial_arguments PackOp, and outer
    parent_ops = walk_ops(outer)
    assert label in parent_ops
    assert outer in parent_ops
    assert inner_a not in parent_ops
    assert inner_b not in parent_ops

    # Label body walk: only sees inner ops
    body_ops = label.body.ops
    assert set(body_ops) == {inner_a, inner_b, inner_result}

    # No overlap
    assert not set(parent_ops) & set(body_ops)


def test_follows_parameters():
    """walk_ops follows parameter references (e.g. branch targets)."""
    label = goto.LabelOp(
        initial_arguments=PackOp(values=[], type=builtin.List(element_type=Nil())),
        name="target",
        body=Block(result=Value(type=Nil())),
    )
    pack = dgen.module.PackOp(values=[], type=builtin.List(element_type=Nil()))
    branch = goto.BranchOp(target=label, arguments=pack)

    ops = walk_ops(branch)
    # label is reached via the 'target' parameter
    assert label in ops
    assert branch in ops


def test_captures_stop_walk():
    """block.ops uses captures as stop set — captured values are leaves."""
    outer_val = ConstantOp(name="outer", value=99, type=Index())
    inner_op = ChainOp(name="use", lhs=outer_val, rhs=outer_val, type=Index())

    # Without captures: walk reaches outer_val
    block_no_captures = Block(result=inner_op)
    assert outer_val in block_no_captures.ops

    # With captures: walk stops at outer_val
    block_with_captures = Block(result=inner_op, captures=[outer_val])
    assert outer_val not in block_with_captures.ops
    assert inner_op in block_with_captures.ops


def test_walk_visits_block_captures():
    """Parent walk_ops visits a child block's captures as dependencies.

    When an op has a block with captures, those captures are parent-scope
    values that the op depends on. walk_ops must visit them so they appear
    in the parent block's ops.
    """
    captured = ConstantOp(name="captured", value=42, type=Index())
    inner_result = ConstantOp(name="inner", value=1, type=Index())
    label = goto.LabelOp(
        initial_arguments=PackOp(values=[], type=builtin.List(element_type=Nil())),
        name="lbl",
        body=Block(
            result=ChainOp(lhs=inner_result, rhs=captured, type=Index()),
            captures=[captured],
        ),
    )
    ops = walk_ops(label)
    # captured is visited as a dependency of the label (via block captures)
    assert captured in ops
    assert label in ops
