"""IR invariant verification utilities.

Two top-level functions:

  verify_closed_blocks(module)  — every value referenced by an op must be
      defined in the same block (either an op in block.ops or a BlockArgument
      in block.args).  Hard-coded exceptions: FunctionOp (call targets) and
      LabelOp (branch targets) are permitted to cross block boundaries.

  verify_all_ready(module)  — every op's type must be a resolved Type, and
      every compile-time parameter (op.__params__) must be a Constant or Type.
      This is the postcondition of staging: no unresolved SSA values remain in
      parameter position.
"""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument
from dgen.module import Module, _walk_all_ops
from dgen.type import Constant, Type


# ---------------------------------------------------------------------------
# verify_closed_blocks
# ---------------------------------------------------------------------------


def _is_cross_block_permitted(value: dgen.Value) -> bool:
    from dgen.dialects.builtin import FunctionOp
    from dgen.dialects.llvm import LabelOp

    return isinstance(value, (FunctionOp, LabelOp))


def _check_in_scope(
    value: object,
    valid: set[int],
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
    assert id(value) in valid, (
        f"{type(op).__name__}.{field_name} references out-of-scope "
        f"{type(value).__name__}"
    )


def _verify_block(block: Block) -> None:
    valid: set[int] = {id(arg) for arg in block.args}
    for op in block.ops:
        valid.add(id(op))

    # The block result itself must be in scope if it is block-scoped.
    if isinstance(block.result, (dgen.Op, BlockArgument)) and not _is_cross_block_permitted(block.result):
        assert id(block.result) in valid, (
            f"block.result references out-of-scope {type(block.result).__name__}"
        )

    for op in block.ops:
        for name, operand in op.operands:
            _check_in_scope(operand, valid, op, name)
        for name, param in op.parameters:
            _check_in_scope(param, valid, op, name)
        for _, child_block in op.blocks:
            _verify_block(child_block)


def verify_closed_blocks(module: Module) -> None:
    """Assert the closed-block invariant holds for all blocks in the module."""
    for func in module.functions:
        _verify_block(func.body)


# ---------------------------------------------------------------------------
# verify_all_ready
# ---------------------------------------------------------------------------


def verify_all_ready(module: Module) -> None:
    """Assert every op's type and __params__ are resolved (no unresolved boundaries).

    Mirrors ``staging._unresolved_boundaries``: a parameter is "not ready" if
    and only if it is an Op or BlockArgument AND NOT a Constant, Type, or
    FunctionOp.  Plain Value forward-references (parser placeholders for
    module-level function names) are stage-0 by design and are valid here.
    """
    from dgen.dialects.builtin import FunctionOp

    _exempt = (Constant, Type, FunctionOp)

    for func in module.functions:
        for op in _walk_all_ops(func):
            assert isinstance(op.type, (Constant, Type)), (
                f"{type(op).__name__}.type is not resolved "
                f"(got {type(op.type).__name__})"
            )
            for name, val in op.parameters:
                if isinstance(val, (dgen.Op, BlockArgument)) and not isinstance(val, _exempt):
                    raise AssertionError(
                        f"{type(op).__name__}.{name} is not a resolved parameter "
                        f"(got {type(val).__name__})"
                    )
