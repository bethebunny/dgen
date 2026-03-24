"""IR invariant verification utilities."""

from __future__ import annotations

import dgen
from dgen import asm
from dgen.block import Block, BlockArgument
from dgen.graph import walk_ops
from dgen.module import Module, _walk_all_ops


class VerificationError(Exception):
    """Base class for IR verification errors."""


class ClosedBlockError(VerificationError):
    """An op references a value not in scope for its block."""


class CycleError(VerificationError):
    """The use-def graph contains a cycle."""


def _annotated_module(module: Module, target: dgen.Value) -> str:
    """Format a module as ASM, annotating the line containing target with ^^^."""
    text = asm.format(module)
    target_name = target.name
    if target_name is None:
        return text
    marker = f"%{target_name}"
    lines = text.splitlines()
    result: list[str] = []
    for line in lines:
        result.append(line)
        idx = line.find(marker)
        if idx >= 0:
            arrow = " " * idx + "^" * len(marker)
            result.append(arrow)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# verify_closed_blocks
# ---------------------------------------------------------------------------


def _verify_block(
    block: Block,
    module: Module,
    visited: set[Block],
) -> None:
    if block in visited:
        return
    visited.add(block)

    valid: set[dgen.Value] = set(block.parameters) | set(block.args)
    for op in block.ops:
        valid.add(op)

    if isinstance(block.result, (dgen.Op, BlockArgument)) and block.result not in valid:
        raise ClosedBlockError(
            f"block.result references out-of-scope "
            f"{type(block.result).__name__} %{block.result.name}\n\n"
            + _annotated_module(module, block.result)
        )

    for op in block.ops:
        for name, operand in op.operands:
            if isinstance(operand, (dgen.Op, BlockArgument)) and operand not in valid:
                raise ClosedBlockError(
                    f"{type(op).__name__}.{name} references out-of-scope "
                    f"{type(operand).__name__} %{operand.name}\n\n"
                    + _annotated_module(module, op)
                )
        for name, param in op.parameters:
            if isinstance(param, (dgen.Op, BlockArgument)) and param not in valid:
                raise ClosedBlockError(
                    f"{type(op).__name__}.{name} references out-of-scope "
                    f"{type(param).__name__} %{param.name}\n\n"
                    + _annotated_module(module, op)
                )
        for _, child_block in op.blocks:
            _verify_block(child_block, module, visited)


def verify_closed_blocks(module: Module) -> None:
    """Assert the closed-block invariant holds for all blocks in the module."""
    visited: set[Block] = set()
    for func in module.functions:
        _verify_block(func.body, module, visited)


# ---------------------------------------------------------------------------
# verify_dag
# ---------------------------------------------------------------------------


def verify_dag(module: Module) -> None:
    """Assert the use-def graph is a DAG (no cycles).

    Uses the same traversal as walk_ops but with DFS path tracking:
    if a value is encountered while still on the current path, there
    is a cycle.
    """
    path: set[dgen.Value] = set()
    visited: set[dgen.Value] = set()

    def visit(value: dgen.Value) -> None:
        if not isinstance(value, dgen.Op):
            return
        if value in visited:
            return
        if value in path:
            raise CycleError(
                f"Use-def cycle detected at %{value.name} "
                f"({type(value).__name__})\n\n"
                + _annotated_module(module, value)
            )
        path.add(value)
        for _, operand in value.operands:
            visit(operand)
        for _, param in value.parameters:
            visit(param)
        for _, block in value.blocks:
            visit(block.result)
        path.remove(value)
        visited.add(value)

    for func in module.functions:
        visit(func)


# ---------------------------------------------------------------------------
# verify_all_ready
# ---------------------------------------------------------------------------


def verify_all_ready(module: Module) -> None:
    """Assert every op is ready (no unresolved parameter dependencies)."""
    for func in module.functions:
        for op in _walk_all_ops(func):
            if not op.ready:
                raise VerificationError(
                    f"{type(op).__name__} %{op.name} is not ready "
                    f"(has unresolved parameter dependencies)\n\n"
                    + _annotated_module(module, op)
                )
