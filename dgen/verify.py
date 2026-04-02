"""IR invariant verification utilities."""

from __future__ import annotations

import dgen
from dgen import asm
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialect import Dialect
from dgen.gen.ast import HasTraitConstraint
from dgen.module import Module, _walk_all_ops
from dgen.trait import Trait
from dgen.type import type_constant


class VerificationError(Exception):
    """Base class for IR verification errors."""


class ConstraintError(VerificationError):
    """An op or type violates a declared constraint."""


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

    valid: set[dgen.Value] = (
        set(block.parameters) | set(block.args) | set(block.captures)
    )
    for op in block.ops:
        valid.add(op)

    if (
        isinstance(block.result, (dgen.Op, BlockArgument, BlockParameter))
        and block.result not in valid
    ):
        raise ClosedBlockError(
            f"block.result references out-of-scope "
            f"{type(block.result).__name__} %{block.result.name}\n\n"
            + _annotated_module(module, block.result)
        )

    for op in block.ops:
        for name, operand in op.operands:
            if (
                isinstance(operand, (dgen.Op, BlockArgument, BlockParameter))
                and operand not in valid
            ):
                raise ClosedBlockError(
                    f"{type(op).__name__}.{name} references out-of-scope "
                    f"{type(operand).__name__} %{operand.name}\n\n"
                    + _annotated_module(module, op)
                )
        for name, param in op.parameters:
            if (
                isinstance(param, (dgen.Op, BlockArgument, BlockParameter))
                and param not in valid
            ):
                raise ClosedBlockError(
                    f"{type(op).__name__}.{name} references out-of-scope "
                    f"{type(param).__name__} %{param.name}\n\n"
                    + _annotated_module(module, op)
                )
        for _, child_block in op.blocks:
            # Captures must chain: every capture of a child block must be
            # in scope in the parent. Otherwise replace_uses on the parent
            # can't maintain the child's captures.
            for capture in child_block.captures:
                if (
                    isinstance(capture, (dgen.Op, BlockArgument, BlockParameter))
                    and capture not in valid
                ):
                    raise ClosedBlockError(
                        f"child block captures out-of-scope "
                        f"{type(capture).__name__} %{capture.name}\n\n"
                        + _annotated_module(module, capture)
                    )
            _verify_block(child_block, module, visited)


def verify_closed_blocks(module: Module) -> None:
    """Assert the closed-block invariant holds for all blocks in the module."""
    visited: set[Block] = set()
    for func in module.functions:
        _verify_block(func.body, module, visited)
    # Verify no op appears in multiple blocks.
    _verify_unique_ownership(module)


def _verify_unique_ownership(module: Module) -> None:
    """Assert every op belongs to exactly one block's block.ops."""
    owner: dict[int, str] = {}  # op id → block description

    def _check_block(block: Block, name: str) -> None:
        for op in block.ops:
            op_id = id(op)
            if op_id in owner:
                raise ClosedBlockError(
                    f"{type(op).__name__} %{op.name} appears in both "
                    f"{owner[op_id]} and {name}\n\n" + _annotated_module(module, op)
                )
            owner[op_id] = name
            if isinstance(op, dgen.Op):
                for _, child_block in op.blocks:
                    child_name = op.name or type(op).__name__
                    _check_block(child_block, child_name)

    for func in module.functions:
        _check_block(func.body, func.name or "function")


# ---------------------------------------------------------------------------
# verify_dag
# ---------------------------------------------------------------------------


def verify_dag(module: Module) -> None:
    """Assert the use-def graph is a DAG (no cycles).

    Uses the same traversal as block.ops but with DFS path tracking:
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
                f"({type(value).__name__})\n\n" + _annotated_module(module, value)
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
    """Assert every op is ready (all compile-time data is known).

    An op is ready when its type, all operand types, and all parameters are
    resolved constants. This means it's safe for a pass to inspect any
    compile-time property of the op without encountering unresolved values.
    """
    for func in module.functions:
        for op in _walk_all_ops(func):
            if not op.ready:
                raise VerificationError(
                    f"{type(op).__name__} %{op.name} is not ready "
                    f"(has unresolved parameter dependencies)\n\n"
                    + _annotated_module(module, op)
                )


# ---------------------------------------------------------------------------
# verify_constraints
# ---------------------------------------------------------------------------


def _resolve_trait(trait_name: str, op: dgen.Op) -> type[Trait]:
    """Look up a trait class by name from the dialect registry."""
    for dialect in Dialect._registry.values():
        if trait_name in dialect.traits:
            return dialect.traits[trait_name]
    raise ConstraintError(
        f"unknown trait {trait_name!r} referenced in constraint on "
        f"{type(op).__name__} %{op.name}"
    )


def _resolve_subject_type(subject: str, op: dgen.Op) -> dgen.Type:
    """Resolve a constraint subject name to the type of that operand/param."""
    for name, value in op.operands:
        if name == subject:
            return type_constant(value.type)
    for name, value in op.parameters:
        if name == subject:
            return type_constant(value)
    raise ConstraintError(
        f"constraint references unknown subject {subject!r} on "
        f"{type(op).__name__} %{op.name}"
    )


def _verify_has_trait(
    constraint: HasTraitConstraint, op: dgen.Op, module: Module
) -> None:
    """Verify a single has-trait constraint on an op."""
    subject_type = _resolve_subject_type(constraint.lhs, op)
    trait_class = _resolve_trait(constraint.trait, op)
    if not isinstance(subject_type, trait_class):
        raise ConstraintError(
            f"{type(op).__name__} %{op.name}: "
            f"operand {constraint.lhs!r} has type {type(subject_type).__name__} "
            f"which does not implement trait {constraint.trait}\n\n"
            + _annotated_module(module, op)
        )


def verify_constraints(module: Module) -> None:
    """Check trait constraints on all ops in the module.

    For each op with ``__constraints__``, verifies that trait constraints
    are satisfied: the subject's type must be an instance of the trait class.
    Other constraint kinds (match, expression) are not yet verified.
    """
    for func in module.functions:
        for op in _walk_all_ops(func):
            for constraint in op.__constraints__:
                if isinstance(constraint, HasTraitConstraint):
                    _verify_has_trait(constraint, op, module)
