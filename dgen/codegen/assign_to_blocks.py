"""Phase 1 of codegen: assign ops to basic-block groups by label dependency."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import dgen
from dgen.dialects import goto

GroupKey = frozenset[goto.LabelOp]


@dataclass
class Segment:
    """One separated segment for linearization.

    label:      real LabelOp to recurse into (ops must be empty).
    ops:        inline ops for synthetic or anonymous segments.
    synth_name: LLVM block name when label is None and ops are non-empty.
                If both label and synth_name are None, fold ops into the
                current block (anonymous segment).
    """

    label: goto.LabelOp | None
    ops: list[dgen.Op]
    synth_name: str | None = None


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


def _key_deps(op: dgen.Op) -> Iterator[dgen.Value]:
    """Dependencies for key computation: operands + block deps (not params)."""
    for _, v in op.operands:
        yield v
    for _, block in op.blocks:
        yield from block.dependencies


def _group_by_label_deps(block: dgen.Block) -> dict[GroupKey, list[dgen.Op]]:
    """Assign each op in a block to a group keyed by its label dependencies.

    Scans ``block.ops`` (topo order) and computes a *GroupKey* per op — the
    frozenset of ``LabelOp``s the op transitively depends on through operands
    and block captures.  Labels are barriers: a dependency on a label adds that
    label to the key but does not propagate the label's own key.

    The key propagates forward in O(V+E)::

        key(op) = ∪ key(dep) for each non-label dep in this block
                  ∪ {dep}    for each LabelOp dep in this block

    Returns a dict mapping each distinct key to its ops (in topo order).
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


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------


def assign_to_blocks(
    block: dgen.Block, synth_counter: int = 0
) -> tuple[list[Segment], int]:
    """Split a block's ops into segments by label-dependency.

    Groups ops by their transitive LabelOp dependencies in O(V+E), then
    orders segments: no-dep group first, each label followed by its
    single-dep ops, remaining labels, then multi-dep groups.

    Returns ``(segments, updated_synth_counter)``.
    """
    groups = _group_by_label_deps(block)

    # Single group with no labels → anonymous segment.
    if len(groups) == 1:
        _, ops = next(iter(groups.items()))
        if not any(isinstance(op, goto.LabelOp) for op in ops):
            return [Segment(None, ops)], synth_counter

    def synth(ops_list: list[dgen.Op]) -> Segment:
        nonlocal synth_counter
        name = f"_blk{synth_counter}"
        synth_counter += 1
        return Segment(None, ops_list, synth_name=name)

    # Extract labels from all groups (they may have non-empty keys when
    # their body captures other labels) and collect non-label ops separately.
    labels: list[goto.LabelOp] = []
    non_label_groups: dict[GroupKey, list[dgen.Op]] = {}
    for key, ops in groups.items():
        for op in ops:
            if isinstance(op, goto.LabelOp):
                labels.append(op)
            else:
                non_label_groups.setdefault(key, []).append(op)

    if not labels:
        return [Segment(None, block.ops)], synth_counter

    result: list[Segment] = []
    no_dep = non_label_groups.pop(frozenset(), None)
    if no_dep:
        result.append(synth(no_dep))

    emitted: set[goto.LabelOp] = set()
    for label in labels:
        dep_ops = non_label_groups.pop(frozenset({label}), None)
        if dep_ops is not None:
            result.append(Segment(label, []))
            result.append(synth(dep_ops))
            emitted.add(label)

    for label in labels:
        if label not in emitted:
            result.append(Segment(label, []))

    for dep_ops in non_label_groups.values():
        result.append(synth(dep_ops))

    return result, synth_counter
