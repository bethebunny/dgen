"""IR invariant verification utilities."""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects.function import FunctionOp
from dgen.ir.constraints import TraitConstraint
from dgen.ir.traversal import all_blocks, all_values
from dgen.asm import asm_with_imports
from dgen.type import Linearity, constant, format_value


class VerificationError(Exception):
    """Base class for IR verification errors."""


class ConstraintError(VerificationError):
    """An op or type violates a declared constraint."""


class ClosedBlockError(VerificationError):
    """An op references a value not in scope for its block."""


class CycleError(VerificationError):
    """The use-def graph contains a cycle."""


class LinearityError(VerificationError):
    """An op or block violates the linear/affine resource discipline."""


class DoubleConsumeError(LinearityError):
    """An affine or linear value is consumed by more than one op."""


class LinearLeakError(LinearityError):
    """A linear value remains ``Available`` at a block's exit and is not
    the block's result — its single-consume obligation is unmet."""


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
    """Assert every op belongs to exactly one scope.

    Walks each reachable FunctionOp (plus the root) as an independent
    top-level scope, recursing through its body's nested blocks. If an op
    is reached via two such walks, raise — this manifests when a
    FunctionOp is referenced from another block without being captured,
    since the referencer then drags that FunctionOp's body into its own
    scope as well.
    """
    owner: dict[dgen.Op, str] = {}

    def _check(block: Block, scope: str) -> None:
        for op in block.ops:
            if op in owner:
                raise ClosedBlockError(
                    f"{type(op).__name__} %{op.name} appears in both "
                    f"{owner[op]} and {scope}"
                )
            owner[op] = scope
            for _, child in op.blocks:
                _check(child, op.name or type(op).__name__)

    starts: list[dgen.Value] = [
        v for v in all_values(root) if isinstance(v, FunctionOp)
    ]
    if root not in starts:
        starts.append(root)
    for v in starts:
        for _, block in v.blocks:
            _check(block, v.name or type(v).__name__)


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
    return constant(subject.type)


def _verify_trait_constraint(
    constraint: TraitConstraint, op: dgen.Op, root: dgen.Value
) -> None:
    """Verify a single trait constraint on an op.

    The constraint carries a closure produced by the builder that, given
    *op*, constructs the trait instance to test against (substituting the
    op's own parameter values into the constraint's references). The
    structural match is delegated to :meth:`Value.has_trait`.
    """
    subject = _resolve_subject(constraint.subject, op)
    target = constraint.build_target(op)
    assert isinstance(target, dgen.Type), (
        f"trait constraint did not resolve to a Type instance: {target!r}"
    )
    if not _subject_type(subject).has_trait(target):
        raise ConstraintError(
            f"{type(op).__name__} %{op.name}: "
            f"subject {constraint.subject!r} ({type(_subject_type(subject)).__name__}) "
            f"does not implement trait {format_value(target)}\n\n"
            + _annotated_asm(root, op)
        )


def verify_constraints(root: dgen.Value) -> None:
    """Check trait constraints on all ops reachable from root."""
    for value in all_values(root):
        if not isinstance(value, dgen.Op):
            continue
        for constraint in value.__constraints__:
            if isinstance(constraint, TraitConstraint):
                _verify_trait_constraint(constraint, value, root)


# ---------------------------------------------------------------------------
# verify_linearity
# ---------------------------------------------------------------------------
#
# Implements the per-block typing-context algorithm in docs/linear_types.md.
# Γ : Value → {AVAILABLE, MAYBE_AVAILABLE, CONSUMED} for the substructural
# values in scope at each point during a topological walk of the block.
# Each block is verified independently — child blocks recurse with their
# own Γ_in and the parent's Γ proceeds without consulting the children's
# internal state.
#
# Block-holding ops are conservative: any op with owned blocks is treated
# as having unknown semantics today (see ``_has_known_block_semantics``).
# A capture into such an op transitions the source value to
# ``MAYBE_AVAILABLE`` in the parent's Γ rather than ``CONSUMED`` — the
# inner block may or may not have actually used the value, so neither
# "definitely consumed" nor "definitely still available" is right. At
# the parent's exit check ``MAYBE_AVAILABLE`` is treated permissively
# (no leak). Once an op declares its block-execution contract precisely
# the verifier can transition to ``CONSUMED`` for ops known to consume
# captured values.


_AVAILABLE = "available"
_MAYBE_AVAILABLE = "maybe_available"
_CONSUMED = "consumed"


def is_linear(value: dgen.Value) -> bool:
    """Whether *value* is of a ``Linear``-trait type."""
    return value.linearity is Linearity.LINEAR


def is_affine_or_linear(value: dgen.Value) -> bool:
    """The verifier's main predicate — True for any value subject to
    resource discipline (consume-at-most-once or consume-exactly-once)."""
    return value.linearity is not Linearity.UNRESTRICTED


def _has_known_block_semantics(op: dgen.Op) -> bool:
    """Whether the verifier knows how *op* uses its captured-into-child
    substructural values.

    Today: always ``False`` for any op with owned blocks. This is the
    conservative shim that keeps the verifier sound while we don't have
    per-op linearity contracts. Ops without blocks (``raise``, ``branch``,
    chains, etc.) don't reach this codepath at all.

    See `TODO.md` (Type system / effects) for the path forward — likely a
    trait or method on ``Op`` that ops opt into when they want precise
    treatment.
    """
    return False


def _consume_at(
    gamma: dict[dgen.Value, str],
    value: dgen.Value,
    *,
    by: dgen.Value,
    root: dgen.Value,
) -> None:
    """Transition a substructural value to ``CONSUMED`` in ``gamma``.

    Lookup failure means "not in scope", per the doc — out-of-scope use
    is the closed-block verifier's responsibility, not ours. Re-consume
    of a ``CONSUMED`` value raises ``DoubleConsumeError``.
    ``MAYBE_AVAILABLE → CONSUMED`` is permitted: the verifier doesn't
    know whether the inner block already consumed the value, and the
    conservative reading is "trust the explicit consume here."
    """
    if value not in gamma:
        return
    if gamma[value] is _CONSUMED:
        raise DoubleConsumeError(
            f"{type(value).__name__} %{value.name} ({value.linearity.value}) "
            f"consumed twice; second consume by {type(by).__name__} "
            f"%{by.name}\n\n" + _annotated_asm(root, value)
        )
    gamma[value] = _CONSUMED


def _capture_into_unknown(
    gamma: dict[dgen.Value, str],
    value: dgen.Value,
    *,
    by: dgen.Value,
    root: dgen.Value,
) -> None:
    """Transition a substructural value into ``MAYBE_AVAILABLE`` because
    it was captured into the body of an op with unknown block semantics.

    ``AVAILABLE → MAYBE_AVAILABLE``, ``MAYBE_AVAILABLE → MAYBE_AVAILABLE``,
    ``CONSUMED → reject`` (you can't capture an already-consumed value).
    Multiple sibling unknown ops capturing the same value all stay
    ``MAYBE_AVAILABLE``: this is what makes ``goto.label`` /
    ``goto.conditional_branch`` patterns work — both branches of an if
    capture ``%exit``, and the verifier doesn't double-charge.
    """
    if value not in gamma:
        return
    if gamma[value] is _CONSUMED:
        raise DoubleConsumeError(
            f"{type(value).__name__} %{value.name} ({value.linearity.value}) "
            f"captured into {type(by).__name__} %{by.name} after being "
            f"consumed\n\n" + _annotated_asm(root, value)
        )
    if gamma[value] is _AVAILABLE:
        gamma[value] = _MAYBE_AVAILABLE


def _verify_linearity_block(block: Block, root: dgen.Value) -> None:
    """Verify the typing-context invariants on a single block.

    Initial Γ_in: block parameters, captures, and runtime args, all
    ``AVAILABLE`` (filtered to substructural). Walk ``block.values`` in
    topological order; for each op, transition its substructural
    operand/parameter consumes through ``_consume_at``, and any captures
    into owned-but-unknown-semantics children through
    ``_capture_into_unknown``. Each child block recurses with its own
    Γ_in. At block exit, any ``LINEAR`` value still ``AVAILABLE`` (and
    not ``block.result``) is a leak; ``MAYBE_AVAILABLE`` is permissive.
    """
    gamma: dict[dgen.Value, str] = {}
    for source in (block.args, block.parameters, block.captures):
        for v in source:
            if is_affine_or_linear(v):
                gamma[v] = _AVAILABLE

    for v in block.values:
        if not isinstance(v, dgen.Op):
            continue  # leaves: BlockArg / BlockParam / Constant / type
        # Op consumes its substructural operands and parameters.
        for source in (v.operands, v.parameters):
            for _, dep in source:
                if is_affine_or_linear(dep):
                    _consume_at(gamma, dep, by=v, root=root)
        # Captures into child blocks. Today every block-holding op is
        # treated as unknown-semantics — the captured value transitions
        # to ``MAYBE_AVAILABLE`` rather than ``CONSUMED``. Captures dedup
        # across alternative children of one op (branch-composition).
        if v.blocks and not _has_known_block_semantics(v):
            for cap in {
                c
                for _, child in v.blocks
                for c in child.captures
                if is_affine_or_linear(c)
            }:
                _capture_into_unknown(gamma, cap, by=v, root=root)
        # Each child block verified independently with its own Γ_in.
        for _, child in v.blocks:
            _verify_linearity_block(child, root)
        # Op result, if substructural, becomes Available.
        if is_affine_or_linear(v):
            gamma[v] = _AVAILABLE

    # Exit check: block.result is "yielded" to the surrounding scope —
    # being the block's output IS the consumption. ``MAYBE_AVAILABLE``
    # is treated permissively (the inner block may already have
    # consumed it). Anything still ``AVAILABLE`` and ``LINEAR`` (and
    # not the block result) is a leak.
    for value, state in gamma.items():
        if state is not _AVAILABLE:
            continue
        if value is block.result:
            continue
        if is_linear(value):
            raise LinearLeakError(
                f"linear {type(value).__name__} %{value.name} is "
                f"AVAILABLE at block exit and is not the block result\n\n"
                + _annotated_asm(root, value)
            )


def verify_linearity(root: dgen.Value) -> None:
    """Verify the linear / affine resource discipline on all blocks
    reachable from *root*.

    See ``docs/linear_types.md`` for the formal rules. Top-level walk
    wraps *root* in a synthetic ``Block`` mirroring ``Pass.run`` so the
    same algorithm covers both block bodies and bare values.
    """
    _verify_linearity_block(Block(result=root), root)
