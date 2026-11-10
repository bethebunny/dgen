"""Use-def graph utilities."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from contextvars import ContextVar

import dgen


class RandomOrder:
    """A random depth-first order that stays the same while the graph does.

    Each value draws a rank the first time a walk meets it, and walks visit
    dependencies lowest rank first instead of in field order.
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.ranks: dict[dgen.Value, float] = {}

    def rank(self, value: dgen.Value) -> float:
        if value not in self.ranks:
            self.ranks[value] = self.rng.random()
        return self.ranks[value]


# When set, walks take its order, to test that nothing depends on the field order.
random_order: ContextVar[RandomOrder | None] = ContextVar("random_order", default=None)


def inline_block(block: dgen.Block, args: list[dgen.Value]) -> dgen.Value:
    """Inline a closed block by substituting its args with actual values.

    Given a closed block whose complete dependency interface is block.args,
    replace every reference to a block arg with the corresponding value from
    `args` and return the block's result.  The returned value (and all ops
    reachable from it) is valid in the caller's scope.

    This is the fundamental inlining operation enabled by the closed-block
    invariant: because every external dependency is declared in block.args,
    substitution is guaranteed to produce a well-scoped result.
    """
    assert len(args) == len(block.args), (
        f"inline_block: expected {len(block.args)} args, got {len(args)}"
    )
    for old_arg, new_val in zip(block.args, args):
        block.replace_uses_of(old_arg, new_val)
    return block.result


def transitive_dependencies(
    value: dgen.Value, *, stop: Iterable[dgen.Value] = ()
) -> Iterator[dgen.Value]:
    """Iterate over all transitive dependencies in topological order.

    Stop at any elements in the `stop` set."""
    visited: set[dgen.Value] = set(stop)
    if value in visited:
        return
    visited.add(value)
    order = random_order.get()

    def dependencies(value: dgen.Value) -> Iterator[dgen.Value]:
        return (
            iter(value.dependencies)
            if order is None
            else _ranked(value, visited, order)
        )

    # Each open value with its dependency iterator, so depth costs memory, not frames.
    stack = [(value, dependencies(value))]
    while stack:
        top, pending = stack[-1]
        if (dep := next((d for d in pending if d not in visited), None)) is None:
            stack.pop()
            yield top
        else:
            visited.add(dep)
            stack.append((dep, dependencies(dep)))


def _ranked(
    value: dgen.Value, visited: set[dgen.Value], order: RandomOrder
) -> Iterator[dgen.Value]:
    """value's unvisited dependencies by rank, reread at each step like the lazy walk."""
    while pending := [d for d in value.dependencies if d not in visited]:
        yield min(pending, key=order.rank)


def all_values(value: dgen.Value) -> Iterator[dgen.Value]:
    """Iterate over all values in topological order, including traversing into nested blocks."""
    for value in transitive_dependencies(value):
        yield from interior_values(value)
        yield value


def interior_values(value: dgen.Value) -> Iterator[dgen.Value]:
    """Iterate over all values nested in blocks within value."""
    for _, block in value.blocks:
        for value in block.values:
            yield value
            yield from interior_values(value)


def interior_blocks(value: dgen.Value) -> Iterator[dgen.Block]:
    """Blocks nested within this value's block fields, recursively."""
    for _, block in value.blocks:
        yield block
        for v in block.values:
            yield from interior_blocks(v)


def all_blocks(value: dgen.Value) -> Iterator[dgen.Block]:
    """All blocks reachable from value: own blocks + nested blocks of dependencies."""
    for v in transitive_dependencies(value):
        yield from interior_blocks(v)
