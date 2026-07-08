"""Ephemeral use handles over the use-def graph.

``uses_of`` walks a block subtree and yields a thin ``Use`` per edge that
references the queried value. The walk is capture-guided: it descends into
a nested block only when that block captures the value, so the closed-block
invariant doubles as the query index — sub-blocks that don't declare the
value are never visited.

``Use.rebind`` swaps the single edge it names and maintains the closed-block
invariant around it: every block between the query root and the edge gains a
capture of the new value, and captures of the old value are pruned wherever
a block's subtree no longer needs them. The caller guarantees the new value
is in scope at the query root.

Uses are created during iteration and are not stored on values; a handle is
only meaningful until the next mutation of the graph it was walked from.
``rebind_all`` materializes before rebinding for exactly that reason — the
lazy walk follows the very edges rebinding changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import dgen


@dataclass(frozen=True)
class Use:
    """One edge in the use-def graph: ``owner``'s ``field`` references a value.

    ``rebind(new)`` assigns the edge and repairs captures along the path it
    was found through. ``owner`` is the referencing value, or the block
    itself for ``result`` edges.
    """

    owner: dgen.Value | dgen.Block
    field: str
    rebind: Callable[[dgen.Value], None]


def uses_of(
    value: dgen.Value,
    within: dgen.Block,
    into: Callable[[dgen.Value], bool] = lambda op: True,
) -> Iterator[Use]:
    """Every edge referencing *value* in *within*'s subtree.

    Descends into a nested block only when its owner passes *into* and the
    block captures *value* (closed blocks: there is no other way in).
    """
    yield from _block_uses(value, within, (within,), into)


def rebind_all(uses: Iterable[Use], new: dgen.Value) -> int:
    """Rebind every use to *new*; returns how many were rebound.

    Materializes *uses* first: rebinding mutates the edges a lazy
    ``uses_of`` walk follows.
    """
    materialized = list(uses)
    for use in materialized:
        use.rebind(new)
    return len(materialized)


def _block_uses(
    x: dgen.Value,
    block: dgen.Block,
    path: tuple[dgen.Block, ...],
    into: Callable[[dgen.Value], bool],
) -> Iterator[Use]:
    for arg in (*block.args, *block.parameters):
        if arg.type is x:
            yield _use(arg, "type", x, path)
    if block.result is x:
        yield Use(owner=block, field="result", rebind=_result_rebinder(block, x, path))
    for value in block.values:
        for name, operand in value.operands:
            if operand is x:
                yield _use(value, name, x, path)
        for name, parameter in value.parameters:
            if parameter is x:
                yield _use(value, name, x, path)
        if value.type is x:
            yield _use(value, "type", x, path)
        if into(value):
            for _, child in value.blocks:
                if any(capture is x for capture in child.captures):
                    yield from _block_uses(x, child, (*path, child), into)


def _use(
    owner: dgen.Value, field: str, old: dgen.Value, path: tuple[dgen.Block, ...]
) -> Use:
    def rebind(new: dgen.Value) -> None:
        owner.rebind_field(field, new)
        _repair_captures(old, new, path)

    return Use(owner=owner, field=field, rebind=rebind)


def _result_rebinder(
    block: dgen.Block, old: dgen.Value, path: tuple[dgen.Block, ...]
) -> Callable[[dgen.Value], None]:
    def rebind(new: dgen.Value) -> None:
        block.result = new
        _repair_captures(old, new, path)

    return rebind


def _repair_captures(
    old: dgen.Value, new: dgen.Value, path: tuple[dgen.Block, ...]
) -> None:
    # Blocks below the query root capture the new value; the caller
    # guarantees it is in scope at the root itself.
    for block in path[1:]:
        if not any(capture is new for capture in block.captures):
            block.captures.append(new)
    # Prune stale captures of the old value deepest-first, so an inner
    # drop is visible to the outer blocks' checks.
    for block in reversed(path):
        if any(capture is old for capture in block.captures) and not _still_needs(
            block, old
        ):
            block.captures = [c for c in block.captures if c is not old]


def _still_needs(block: dgen.Block, x: dgen.Value) -> bool:
    """Whether *block* still needs to capture *x*: some direct edge in the
    block references it, or a child block captures it (capture chains —
    a child's declared dependency must stay satisfiable, even if the child
    itself holds it stale)."""
    if block.result is x:
        return True
    if any(arg.type is x for arg in (*block.args, *block.parameters)):
        return True
    for value in block.values:
        if value.type is x:
            return True
        if any(operand is x for _, operand in value.operands):
            return True
        if any(parameter is x for _, parameter in value.parameters):
            return True
        for _, child in value.blocks:
            if any(capture is x for capture in child.captures):
                return True
    return False
