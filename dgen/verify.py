"""IR invariant verification utilities."""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialect import Dialect
from dgen.gen.ast import HasTraitConstraint
from dgen.graph import all_blocks, all_values
from dgen.module import asm_with_imports
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


def _annotated_asm(root: dgen.Value, target: dgen.Value) -> str:
    """Format root as ASM, annotating the line containing target with ^^^."""
    text = "\n".join(asm_with_imports(root))
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
    root: dgen.Value,
    visited: set[Block],
) -> None:
    if block in visited:
        return
    visited.add(block)

    # Closed-block invariant: the transitive dependencies of block.result,
    # stopping at captures, must contain BlockArguments/BlockParameters from
    # this block only — not from any other block.
    local = set(block.parameters) | set(block.args)
    for value in block.values:
        if isinstance(value, (BlockArgument, BlockParameter)) and value not in local:
            raise ClosedBlockError(
                f"block contains foreign {type(value).__name__} "
                f"%{value.name}\n\n" + _annotated_asm(root, value)
            )

    # Captures must chain: every capture of a child block must be
    # reachable in the parent (in values, local, or parent's own captures).
    parent_scope = local | set(block.captures) | set(block.values)
    for op in block.ops:
        for _, child_block in op.blocks:
            for capture in child_block.captures:
                if capture not in parent_scope:
                    raise ClosedBlockError(
                        f"child block captures out-of-scope "
                        f"{type(capture).__name__} %{capture.name}\n\n"
                        + _annotated_asm(root, capture)
                    )
            _verify_block(child_block, root, visited)


def verify_closed_blocks(root: dgen.Value) -> None:
    """Assert the closed-block invariant holds for all blocks reachable from root."""
    visited: set[Block] = set()
    for block in all_blocks(root):
        _verify_block(block, root, visited)
    _verify_unique_ownership(root)


def _verify_unique_ownership(root: dgen.Value) -> None:
    """Assert every op belongs to exactly one block's block.ops.

    Walks from every reachable FunctionOp as a top-level starting point,
    recursing into nested blocks. If a FunctionOp is referenced from
    another block without being captured, its body ops will be walked twice
    (once under the referencer, once as a top-level), which manifests as
    a duplicate-owner error.
    """
    from dgen.dialects.function import FunctionOp

    owner: dict[int, str] = {}  # op id → block description

    def _describe(op: dgen.Op) -> str:
        return f"{type(op).__name__} %{op.name}"

    def _check_block(block: Block, scope: str) -> None:
        for op in block.ops:
            op_id = id(op)
            if op_id in owner:
                raise ClosedBlockError(
                    f"{_describe(op)} appears in both {owner[op_id]} and {scope}"
                )
            owner[op_id] = scope
            if isinstance(op, dgen.Op):
                for _, child_block in op.blocks:
                    _check_block(child_block, op.name or type(op).__name__)

    # Walk from every reachable FunctionOp as its own scope, plus the root.
    reachable_fns: list[dgen.Op] = [
        v for v in all_values(root) if isinstance(v, FunctionOp)
    ]
    # Ensure root itself is included even if it isn't a FunctionOp.
    if isinstance(root, dgen.Op) and root not in reachable_fns:
        reachable_fns.append(root)
    for fn in reachable_fns:
        for _, block in fn.blocks:
            _check_block(block, fn.name or type(fn).__name__)


# ---------------------------------------------------------------------------
# verify_dag
# ---------------------------------------------------------------------------


def verify_dag(root: dgen.Value) -> None:
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
            # Don't dump ASM — the graph has a cycle, formatting would loop.
            raise CycleError(
                f"Use-def cycle detected at %{value.name} ({type(value).__name__})"
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

    visit(root)


# ---------------------------------------------------------------------------
# verify_all_ready
# ---------------------------------------------------------------------------


def verify_all_ready(root: dgen.Value) -> None:
    """Assert every op reachable from root is ready."""
    for value in all_values(root):
        if isinstance(value, dgen.Op) and not value.ready:
            raise VerificationError(
                f"{type(value).__name__} %{value.name} is not ready "
                f"(has unresolved parameter dependencies)\n\n"
                + _annotated_asm(root, value)
            )


# ---------------------------------------------------------------------------
# verify_constraints
# ---------------------------------------------------------------------------


def _resolve_trait(trait_name: str, op: dgen.Op) -> type[Trait]:
    """Look up a trait class by name from the dialect registry."""
    for dialect in Dialect._registry.values():
        cls = dialect.types.get(trait_name)
        if cls is not None and issubclass(cls, Trait):
            return cls
    raise ConstraintError(
        f"unknown trait {trait_name!r} referenced in constraint on "
        f"{type(op).__name__} %{op.name}"
    )


def _resolve_subject(subject: str, op: dgen.Op) -> dgen.Value:
    """Resolve a constraint subject name to the value of that operand/param."""
    for name, value in op.operands:
        if name == subject:
            return value
    for name, value in op.parameters:
        if name == subject:
            return value
    raise ConstraintError(
        f"constraint references unknown subject {subject!r} on "
        f"{type(op).__name__} %{op.name}"
    )


def _subject_type(subject: dgen.Value) -> dgen.Type:
    """Get the type to check for trait membership.

    For a Type, this is the type itself. For other values (Constant,
    BlockArgument, Op), this is value.type.
    """
    if isinstance(subject, dgen.Type):
        return subject
    return type_constant(subject.type)


def _verify_has_trait(
    constraint: HasTraitConstraint, op: dgen.Op, root: dgen.Value
) -> None:
    """Verify a single has-trait constraint on an op."""
    subject = _resolve_subject(constraint.lhs, op)
    trait_class = _resolve_trait(constraint.trait, op)
    subject_type = _subject_type(subject)
    if not isinstance(subject_type, trait_class):
        raise ConstraintError(
            f"{type(op).__name__} %{op.name}: "
            f"subject {constraint.lhs!r} ({type(subject_type).__name__}) "
            f"does not implement trait {constraint.trait}\n\n"
            + _annotated_asm(root, op)
        )


def verify_constraints(root: dgen.Value) -> None:
    """Check trait constraints on all ops reachable from root."""
    for value in all_values(root):
        if not isinstance(value, dgen.Op):
            continue
        for constraint in value.__constraints__:
            if isinstance(constraint, HasTraitConstraint):
                _verify_has_trait(constraint, value, root)
