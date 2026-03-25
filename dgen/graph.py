"""Use-def graph utilities."""

from __future__ import annotations

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


def walk_ops(root: dgen.Value, *, stop: set[dgen.Value] | None = None) -> list[dgen.Op]:
    """Walk the use-def graph from root, return ops in topological order.

    - Only includes Op instances (not plain Values or BlockArguments).
    - Does not descend into an op's nested blocks.
    - Dependencies appear before dependents.
    - Values in ``stop`` are treated as leaves (capture boundaries).
    """
    visited: set[dgen.Value] = set(stop) if stop else set()
    order: list[dgen.Op] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dgen.Value):
            # Traverse Type params to find SSA-valued references
            if isinstance(value, dgen.Type):
                for param_name, _ in value.__params__:
                    visit(getattr(value, param_name))
            return
        if value in visited:
            return
        visited.add(value)

        # Type is a Value subclass — traverse its params to find SSA-valued refs
        if isinstance(value, dgen.Type):
            for param_name, _ in value.__params__:
                visit(getattr(value, param_name))

        if not isinstance(value, dgen.Op):
            return

        # Visit dependencies first (both operands and parameter values)
        for _, operand in value.operands:
            visit(operand)
        for _, param in value.parameters:
            visit(param)
        # Visit type (may be an SSA value or a Type with SSA-valued params)
        visit(value.type)
        # Visit block parameter and arg types (leaves in the use-def graph)
        for _, block in value.blocks:
            for param in block.parameters:
                visit(param.type)
            for arg in block.args:
                visit(arg.type)

        order.append(value)

    visit(root)
    return order
