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


# ---------------------------------------------------------------------------
# Label-body block utilities (two-phase lowering support)
# ---------------------------------------------------------------------------


def placeholder_block() -> dgen.Block:
    """Create a placeholder block for label ops whose bodies aren't known yet."""
    from dgen.dialects.builtin import Nil

    return dgen.Block(result=dgen.Value(type=Nil()))


def chain_body(ops: list[dgen.Op]) -> dgen.Value:
    """Chain all body ops so they're reachable from a single root via use-def.

    The last op is treated as the terminator. All preceding ops are chained
    to it via ChainOp(lhs=op, rhs=rest) so walk_ops visits them in order.
    """
    from dgen.dialects.builtin import ChainOp, Nil

    if not ops:
        return dgen.Value(type=Nil())
    terminator: dgen.Value = ops[-1]
    for op in reversed(ops[:-1]):
        terminator = ChainOp(lhs=op, rhs=terminator, type=op.type)
    return terminator


def group_into_blocks(
    flat_ops: list[dgen.Op],
) -> tuple[list[dgen.Op], list[tuple[dgen.Op, list[dgen.Op]]]]:
    """Split a flat op list at LabelOp boundaries into (entry_ops, label_groups).

    Returns (entry_ops, [(label_op, body_ops), ...]) where each label_op
    is an llvm.LabelOp and body_ops are the ops that follow it until the
    next label or end of list.
    """
    from dgen.dialects.llvm import LabelOp

    entry_ops: list[dgen.Op] = []
    label_groups: list[tuple[dgen.Op, list[dgen.Op]]] = []
    current_label: dgen.Op | None = None
    current_body: list[dgen.Op] = []

    for op in flat_ops:
        if isinstance(op, LabelOp):
            if current_label is not None:
                label_groups.append((current_label, current_body))
            else:
                entry_ops = current_body
            current_label = op
            current_body = []
        else:
            current_body.append(op)

    if current_label is not None:
        label_groups.append((current_label, current_body))
    else:
        entry_ops = current_body

    return entry_ops, label_groups


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
