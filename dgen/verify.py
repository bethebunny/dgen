"""IR invariant verification utilities.

Two top-level functions:

  verify_closed_blocks(module)  — every value referenced by an op must be
      defined in the same block (either an op in block.ops or a BlockArgument
      in block.args).  Hard-coded exceptions: FunctionOp (call targets) and
      LabelOp (branch targets) are permitted to cross block boundaries.

  verify_all_ready(module)  — every op must satisfy op.ready, meaning all
      compile-time parameters and the op's type are fully resolved (no
      BlockArgument dependencies remain in parameter position).
"""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument
from dgen.dialects.builtin import FunctionOp
from dgen.dialects.llvm import LabelOp
from dgen.module import Module, _walk_all_ops


# ---------------------------------------------------------------------------
# verify_closed_blocks
# ---------------------------------------------------------------------------


def _is_cross_block_permitted(value: dgen.Value) -> bool:
    return isinstance(value, (FunctionOp, LabelOp))


def _check_in_scope(
    value: object,
    valid: set[dgen.Value],
    op: dgen.Op,
    field_name: str,
) -> None:
    """Assert that value (or each element if it's a list) is in scope.

    Only Op instances and BlockArguments are block-scoped.  Pure Constant
    values (non-Op), Types, and other plain Values are compile-time objects
    with no block ownership and are always valid to reference from any block.
    """
    if isinstance(value, list):
        for item in value:
            _check_in_scope(item, valid, op, field_name)
        return
    if not isinstance(value, (dgen.Op, BlockArgument)):
        return
    if _is_cross_block_permitted(value):
        return
    assert value in valid, (
        f"{type(op).__name__}.{field_name} references out-of-scope "
        f"{type(value).__name__}"
    )


def _verify_block(block: Block, visited: set[Block] | None = None) -> None:
    if visited is None:
        visited = set()
    if block in visited:
        return
    visited.add(block)

    valid: set[dgen.Value] = set(block.parameters) | set(block.args)
    for op in block.ops:
        valid.add(op)

    # The block result itself must be in scope if it is block-scoped.
    if isinstance(
        block.result, (dgen.Op, BlockArgument)
    ) and not _is_cross_block_permitted(block.result):
        assert block.result in valid, (
            f"block.result references out-of-scope {type(block.result).__name__}"
        )

    for op in block.ops:
        for name, operand in op.operands:
            _check_in_scope(operand, valid, op, name)
        for name, param in op.parameters:
            _check_in_scope(param, valid, op, name)
        for _, child_block in op.blocks:
            _verify_block(child_block, visited)


def verify_closed_blocks(module: Module) -> None:
    """Assert the closed-block invariant holds for all blocks in the module."""
    for func in module.functions:
        _verify_block(func.body)


# ---------------------------------------------------------------------------
# verify_all_ready
# ---------------------------------------------------------------------------


def verify_all_ready(module: Module) -> None:
    """Assert every op is ready: no BlockArgument dependencies in parameter position.

    Uses op.ready, which recurses through the op's type and __params__ fields.
    BlockArgument.ready is False, so any op whose type or parameters depend on
    a runtime block argument will fail this check.
    """
    for func in module.functions:
        for op in _walk_all_ops(func):
            assert op.ready, (
                f"{type(op).__name__} is not ready (has unresolved parameter dependencies)"
            )
