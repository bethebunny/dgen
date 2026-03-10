"""Use-def graph utilities."""

from __future__ import annotations

import dgen


def walk_ops(root: dgen.Value) -> list[dgen.Op]:
    """Walk the use-def graph from root, return ops in topological order.

    - Only includes Op instances (not plain Values or BlockArguments).
    - Does not descend into an op's nested blocks.
    - Dependencies appear before dependents.
    """
    visited: set[int] = set()
    order: list[dgen.Op] = []

    def visit(value: dgen.Value) -> None:
        vid = id(value)
        if vid in visited:
            return
        visited.add(vid)

        if not isinstance(value, dgen.Op):
            return

        # Visit dependencies first
        for _, operand in value.operands:
            visit(operand)

        order.append(value)

    visit(root)
    return order
