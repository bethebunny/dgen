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
    from dgen.passes.pass_ import Rewriter  # circular: block → graph → pass_

    rewriter = Rewriter(block)
    for old_arg, new_val in zip(block.args, args):
        rewriter.replace_uses(old_arg, new_val)
    return block.result


def placeholder_block() -> dgen.Block:
    """Create a placeholder block for label ops whose bodies aren't known yet."""
    from dgen.dialects.builtin import Nil

    return dgen.Block(result=dgen.Value(type=Nil()))


def transitive_dependencies(
    value: dgen.Value, *, stop: Iterable[dgen.Value] = ()
) -> Iterator[dgen.Value]:
    """Iterate over all transitive dependencies in topological order.

    Stop at any elements in theh `stop` set."""
    visited: set[dgen.Value] = set(stop)

    def visit(value: dgen.Value) -> Iterator[dgen.Value]:
        visited.add(value)
        for dep in value.dependencies:
            # check before recursing to reduce function call overhead
            if dep not in visited:
                yield from visit(dep)
        yield value

    yield from () if value in visited else visit(value)
