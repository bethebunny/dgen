"""Test flat labels: labels as ops in the use-def graph with no body blocks.

Labels are zero-dependency source nodes. Their phi values are projected via
ArgOp. The codegen partitions ops by which label they transitively depend on.
"""

from __future__ import annotations

import dgen
from dgen import Block
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, goto, index, llvm
from toy.dialects import memory
from dgen.dialects.builtin import ChainOp, F64, Nil
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.graph import walk_ops
from dgen.module import ConstantOp, Module, PackOp


def _pack(values: list[dgen.Value]) -> PackOp:
    if not values:
        return PackOp(values=[], type=builtin.List(element_type=Nil()))
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


def test_flat_label_structure():
    """Build a simple loop with flat labels and verify the use-def graph."""

    # %out = alloc
    out = memory.AllocOp(
        shape=memory.Shape(rank=Index().constant(1)),
        type=memory.MemRef(shape=memory.Shape(rank=Index().constant(1))),
    )

    # %header = goto.label()  -- zero deps, source node
    header = goto.LabelOp(name="header")

    # %i = goto.arg(%header)  -- project the loop IV (single-arg label)
    i = goto.ArgOp(name="i", label=header, type=Index())

    # %cmp = algebra.less_than(%i, 4)
    four = ConstantOp(value=4, type=Index())
    cmp = algebra.LessThanOp(left=i, right=four, type=Index())

    # %body = goto.label()
    body_label = goto.LabelOp(name="body")

    # %j = goto.arg(%body)
    j = goto.ArgOp(name="j", label=body_label, type=Index())

    # %next = algebra.add(%j, 1)
    one = ConstantOp(value=1, type=Index())
    next_j = algebra.AddOp(left=j, right=one, type=Index())

    # %back = goto.branch<%header>([%next])  -- back-edge
    back = goto.BranchOp(target=header, arguments=_pack([next_j]))

    # %exit = goto.label()
    exit_label = goto.LabelOp(name="exit")

    # %result = chain(%out, %exit)  -- alloc value, dep on exit label
    result = ChainOp(lhs=out, rhs=exit_label, type=out.type)

    # %print = memory.PrintMemrefOp(%result)
    print_op = memory.PrintMemrefOp(input=result)

    # %cond = goto.conditional_branch<%body, %exit>(%cmp, [%i], [])
    cond = goto.ConditionalBranchOp(
        condition=cmp,
        true_target=body_label,
        false_target=exit_label,
        true_arguments=_pack([i]),
        false_arguments=_pack([]),
    )

    # Chain the terminal branches to keep them alive.
    # %body_term = chain(%back, ???) — back-edge is the body's terminal
    # %header_term = chain(%cond, %body_term) — cond is the header's terminal
    # But %cond depends on %header (via %cmp → %i), and %back depends on
    # %body (via %next → %j). These are the "returns" of each label.
    # They need to be reachable from the block result.

    # %entry = goto.branch<%header>([0])
    zero = ConstantOp(value=0, type=Index())
    entry = goto.BranchOp(name="entry", target=header, arguments=_pack([zero]))

    # Block result: chain everything together
    # The entry branch, conditional branch, and back-edge all need to be alive.
    r1 = ChainOp(lhs=out, rhs=back, type=out.type)  # alloc + body back-edge
    r2 = ChainOp(lhs=r1, rhs=cond, type=out.type)  # + header conditional
    block_result = ChainOp(lhs=r2, rhs=entry, type=out.type)  # + entry branch

    func = FunctionOp(
        name="test",
        result=Nil(),
        type=Function(result=Nil()),
        body=Block(result=block_result),
    )

    # Verify: walk_ops from block_result reaches all ops
    ops = walk_ops(block_result)
    op_names = {type(op).__name__ for op in ops}
    assert "LabelOp" in op_names, "Labels should be reachable"
    assert "ArgOp" in op_names, "ArgOps should be reachable"
    assert "BranchOp" in op_names, "Branches should be reachable"
    assert "ConditionalBranchOp" in op_names

    # Verify: labels have no dependencies (source nodes)
    assert list(header.operands) == []
    assert list(header.parameters) == []
    assert list(body_label.operands) == []
    assert list(exit_label.operands) == []

    # Verify: ArgOp depends on its label (label is an operand)
    arg_operands = dict(i.operands)
    assert arg_operands["label"] is header

    # Verify: back-edge branch targets header (via parameter)
    back_params = dict(back.parameters)
    assert back_params["target"] is header

    # Verify: no cycles in walk_ops (it would infinite-loop or raise)
    # This succeeds if we get here — walk_ops completed.

    # Verify: we can partition ops by label dependency
    # An op "belongs to" a label if it transitively depends on that label's ArgOp
    def label_of(op: dgen.Op) -> goto.LabelOp | None:
        """Which label does this op belong to?

        An op belongs to a label if it transitively depends on that label
        via an ArgOp (phi projection) or directly (e.g. chain(%out, %exit)).
        """
        visited: set[dgen.Value] = set()

        def find(v: dgen.Value) -> goto.LabelOp | None:
            if v in visited:
                return None
            visited.add(v)
            if isinstance(v, goto.LabelOp):
                return v
            if isinstance(v, goto.ArgOp):
                assert isinstance(v.label, goto.LabelOp)
                return v.label
            if not isinstance(v, dgen.Op):
                return None
            for _, operand in v.operands:
                found = find(operand)
                if found is not None:
                    return found
            return None

        return find(op)

    # body ops depend on %body (via %j)
    assert label_of(next_j) is body_label
    assert label_of(back) is body_label

    # header ops depend on %header (via %i)
    assert label_of(cmp) is header
    assert label_of(cond) is header

    # exit ops depend on %exit (via chain)
    assert label_of(print_op) is exit_label

    # entry ops depend on no label
    assert label_of(entry) is None
    assert label_of(out) is None
