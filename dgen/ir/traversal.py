"""Use-def graph utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import dgen


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

    # Iterative post-order DFS using an explicit stack.
    # Each frame is (value, deps_iterator).  When the iterator is exhausted
    # we yield the value (post-order), matching the old recursive semantics.
    stack: list[tuple[dgen.Value, Iterator[dgen.Value]]] = []
    visited.add(value)
    stack.append((value, iter(value.dependencies)))

    while stack:
        node, deps = stack[-1]
        for dep in deps:
            if dep not in visited:
                visited.add(dep)
                stack.append((dep, iter(dep.dependencies)))
                break  # process new node first (DFS)
        else:
            # all deps visited — emit this node (post-order)
            stack.pop()
            yield node


def all_values(value: dgen.Value) -> Iterator[dgen.Value]:
    """Iterate over all values in topological order, including traversing into nested blocks."""
    for v in transitive_dependencies(value):
        yield from interior_values(v)
        yield v


def interior_values(value: dgen.Value) -> Iterator[dgen.Value]:
    """Iterate over all values nested in blocks within value."""
    for _, block in value.blocks:
        for v in block.values:
            yield v
            yield from interior_values(v)


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
