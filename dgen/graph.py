"""Use-def graph utilities."""

from __future__ import annotations

import dgen


def unwrap_chain(result: dgen.Value) -> list[dgen.Op]:
    """Unwrap ChainOp nesting to recover local block ops.

    The chain structure is: ChainOp(lhs=op1, rhs=ChainOp(lhs=op2, rhs=terminator)).
    This walks only the chain spine, returning exactly the ops that were
    chained together — without following transitive operands that may cross
    block boundaries.
    """
    from dgen.dialects.builtin import ChainOp

    ops: list[dgen.Op] = []
    current: dgen.Value = result
    while isinstance(current, ChainOp):
        if isinstance(current.lhs, dgen.Op):
            ops.append(current.lhs)
        current = current.rhs
    if isinstance(current, dgen.Op):
        ops.append(current)
    return ops


def walk_ops(root: dgen.Value) -> list[dgen.Op]:
    """Walk the use-def graph from root, return ops in topological order.

    - Only includes Op instances (not plain Values or BlockArguments).
    - Does not descend into an op's nested blocks.
    - Dependencies appear before dependents.
    """
    visited: set[int] = set()
    order: list[dgen.Op] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dgen.Value):
            return
        vid = id(value)
        if vid in visited:
            return
        visited.add(vid)

        if not isinstance(value, dgen.Op):
            return

        # Visit dependencies first (both operands and parameter values)
        for _, operand in value.operands:
            visit(operand)
        for _, param in value.parameters:
            visit(param)

        order.append(value)

    visit(root)
    return order
