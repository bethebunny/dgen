"""Phase 1 of codegen: assign ops to basic-block groups by label dependency."""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.dialects import goto

GroupKey = frozenset[goto.LabelOp]


def _key_deps(op: dgen.Op) -> Iterator[dgen.Value]:
    """Dependencies for key computation: operands + block deps (not params)."""
    for _, v in op.operands:
        yield v
    for _, block in op.blocks:
        yield from block.dependencies


def assign_to_blocks(block: dgen.Block) -> dict[GroupKey, list[dgen.Op]]:
    """Assign each op in a block to a group keyed by its label dependencies.

    Scans ``block.ops`` (topo order) and computes a *GroupKey* per op — the
    frozenset of ``LabelOp``s the op transitively depends on through operands
    and block captures.  Labels are barriers: a dependency on a label adds that
    label to the key but does not propagate the label's own key.

    The key propagates forward in O(V+E)::

        key(op) = ∪ key(dep) for each non-label dep in this block
                  ∪ {dep}    for each LabelOp dep in this block

    Returns a dict mapping each distinct key to its ops (in topo order).
    A single-entry dict means the block is already a valid basic block.
    """
    ops = block.ops
    op_set = set(ops)

    keys: dict[dgen.Op, GroupKey] = {}
    for op in ops:
        k: set[goto.LabelOp] = set()
        for dep in _key_deps(op):
            if isinstance(dep, goto.LabelOp) and dep in op_set:
                k.add(dep)
            elif isinstance(dep, dgen.Op) and dep in op_set:
                k |= keys[dep]
        keys[op] = frozenset(k)

    groups: dict[GroupKey, list[dgen.Op]] = {}
    for op in ops:
        groups.setdefault(keys[op], []).append(op)

    return groups
