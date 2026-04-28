"""IR invariant verification utilities."""

from __future__ import annotations

import dgen
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects.function import FunctionOp
from dgen.ir.constraints import TraitConstraint
from dgen.ir.traversal import all_blocks, all_values
from dgen.asm import asm_with_imports
from dgen.type import Totality, Type, constant, format_value


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
    """A linear or affine value is consumed by more than one op."""


class LinearLeakError(LinearityError):
    """A linear value is introduced in a block but never consumed and is
    not the block's result — its single-consume obligation is unmet."""


class LinearLeakAtPartialError(LinearityError):
    """A PARTIAL op is reached while linear values remain live in scope.
    Divergence would leak those obligations on the divergent path."""


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


def _consumes(value: dgen.Value) -> list[dgen.Value]:
    """One-hop affine-or-linear dependencies of *value* — the values it
    directly consumes. Type instances are filtered out: a Value's
    ``dependencies`` yields ``self.type`` and types-as-values are
    universe-1 metadata, not resources subject to linearity.
    """
    return [
        d
        for d in value.dependencies
        if not isinstance(d, Type) and d.is_affine_or_linear
    ]


def _verify_linearity_block(block: Block, root: dgen.Value) -> None:
    """Verify the linearity invariants on a single block, then recurse.

    The walk is a single pass over ``block.values`` (which is already in
    post-order topological order from ``transitive_dependencies``):

    1. Each direct consume of an affine-or-linear value bumps a per-value
       refcount; refcount > 1 → ``DoubleConsumeError``.
    2. At any ``PARTIAL`` op, the running ``live_linear`` set (linear
       values introduced earlier and not yet consumed) must already be
       empty *after* accounting for this op's own consumes — otherwise
       divergence would leak those obligations.
    3. At block exit, any ``LINEAR`` value introduced in the block whose
       refcount is still 0 and that isn't ``block.result`` is a leak.

    Captures are treated as ``introduced`` for the inner block: they are
    leaves in ``block.values`` (in the ``stop`` set) but the inner block
    is on the hook for not double-using them. The outer op already
    accounted for the single capture-use via its own ``dependencies``.
    """
    refcount: dict[dgen.Value, int] = {}
    consumers: dict[dgen.Value, list[dgen.Value]] = {}
    introduced: set[dgen.Value] = set()
    live_linear: set[dgen.Value] = set()

    for capture in block.captures:
        if capture.is_affine_or_linear:
            introduced.add(capture)
            refcount[capture] = 0
            if capture.is_linear:
                live_linear.add(capture)

    for v in block.values:
        # 1. Account for what v consumes.
        for d in _consumes(v):
            refcount[d] = refcount.get(d, 0) + 1
            consumers.setdefault(d, []).append(v)
            if refcount[d] > 1:
                raise DoubleConsumeError(
                    f"{type(d).__name__} %{d.name} ({d.linearity.value}) "
                    f"consumed {refcount[d]} times by "
                    f"{', '.join(f'%{c.name}' for c in consumers[d])}\n\n"
                    + _annotated_asm(root, d)
                )
            live_linear.discard(d)

        # 2. PARTIAL drain check (linear only — affine may stay live).
        if isinstance(v, dgen.Op) and v.totality is Totality.PARTIAL and live_linear:
            leaked = ", ".join(f"%{lv.name}" for lv in live_linear)
            raise LinearLeakAtPartialError(
                f"{type(v).__name__} %{v.name} is PARTIAL but linear "
                f"value(s) {leaked} are still live; divergence would "
                f"leak them\n\n" + _annotated_asm(root, v)
            )

        # 3. v itself may introduce a new affine-or-linear resource.
        if v.is_affine_or_linear:
            introduced.add(v)
            refcount.setdefault(v, 0)
            if v.is_linear:
                live_linear.add(v)

    # 4. Leak check at block exit (linear only; affine may stay 0).
    for v in introduced:
        if v.is_linear and refcount[v] == 0 and v is not block.result:
            raise LinearLeakError(
                f"linear {type(v).__name__} %{v.name} is never consumed "
                f"and is not the block result\n\n" + _annotated_asm(root, v)
            )

    # 5. Recurse into owned blocks of every op as independent walks.
    for op in block.ops:
        for _, child in op.blocks:
            _verify_linearity_block(child, root)


def verify_linearity(root: dgen.Value) -> None:
    """Verify the linear / affine resource discipline on all blocks
    reachable from *root*.

    See ``docs/linear_types.md`` for the formal rules. Top-level walk
    wraps *root* in a synthetic ``Block`` mirroring ``Pass.run`` so the
    same algorithm covers both block bodies and bare values.
    """
    _verify_linearity_block(Block(result=root), root)
