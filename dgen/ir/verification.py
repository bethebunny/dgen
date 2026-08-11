"""IR invariant verification utilities."""

from __future__ import annotations

import enum
import weakref

import dgen
from dgen.asm import asm_with_imports
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects.builtin import (
    Affine,
    Alternatives,
    BodyWithHandler,
    ExactlyOnce,
    Linear,
    Never,
)
from dgen.dialects.function import FunctionOp
from dgen.ir.constraints import TraitConstraint
from dgen.ir.traversal import all_blocks, all_values
from dgen.type import Type, constant, format_value

# Singleton trait instances — `linearity()` is on the hot path, and naive
# `Linear()` / `Affine()` constructions on every call dominated profiles.
_LINEAR_TRAIT = Linear()
_AFFINE_TRAIT = Affine()

# Block-execution traits (see builtin.dgen and Op.verify_block_linearity).
_EXACTLY_ONCE_TRAIT = ExactlyOnce()
_ALTERNATIVES_TRAIT = Alternatives()
_BODY_WITH_HANDLER_TRAIT = BodyWithHandler()

# Cache `linearity` keyed on the value's type. `has_trait` does a structural
# `to_json()` comparison per declared trait; profiling showed this accounted
# for ~40% of test time. Linear/Affine declarations are class-level facts
# in this codebase, so a per-type-instance cache is safe. Keying per
# *instance* (not per class) keeps this correct even if trait declarations
# ever become parametric; the remaining hazard would be in-place mutation
# of a type's declared traits, which nothing does — the `traits` property
# is built from the immutable `.dgen` declaration at dialect load.
_LINEARITY_CACHE: weakref.WeakKeyDictionary[dgen.Value, Linearity] = (
    weakref.WeakKeyDictionary()
)


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
# Block-holding ops charge their child-block captures through the
# ``Op.verify_block_linearity`` protocol: the default dispatches on the
# op's declared block-execution trait (``ExactlyOnce`` /
# ``Alternatives`` / ``BodyWithHandler``, see ``builtin.dgen``) via
# ``BlockLinearityContext``, and bespoke ops may override the method.
# An op declaring no trait stays conservative: a capture transitions
# the source value to ``MAYBE_AVAILABLE`` in the parent's Γ rather than
# ``CONSUMED`` — the inner block may or may not have actually used the
# value, so neither "definitely consumed" nor "definitely still
# available" is right. At the parent's exit check ``MAYBE_AVAILABLE``
# is treated permissively (no leak).


class Linearity(enum.Enum):
    """How a value behaves with respect to resource discipline.

    Read off the value's *type*: a value of type ``T`` is ``LINEAR`` if
    ``T has trait Linear``, ``AFFINE`` if ``T has trait Affine``, and
    ``UNRESTRICTED`` otherwise. See ``docs/linear_types.md``.

    - ``UNRESTRICTED`` — no resource discipline; arbitrary use.
    - ``AFFINE``       — at most one consume.
    - ``LINEAR``       — exactly one consume; must be discharged before
      block exit (or be the block result).
    """

    UNRESTRICTED = "unrestricted"
    AFFINE = "affine"
    LINEAR = "linear"


def linearity(value: dgen.Value) -> Linearity:
    """Classify *value* by the linearity discipline of its type.

    ``Type`` instances always read ``UNRESTRICTED`` — types-as-values
    are universe-1 metadata, not resources.
    """
    if isinstance(value, Type):
        return Linearity.UNRESTRICTED
    t = value.type
    cached = _LINEARITY_CACHE.get(t)
    if cached is not None:
        return cached
    if t.has_trait(_LINEAR_TRAIT):
        result = Linearity.LINEAR
    elif t.has_trait(_AFFINE_TRAIT):
        result = Linearity.AFFINE
    else:
        result = Linearity.UNRESTRICTED
    _LINEARITY_CACHE[t] = result
    return result


def is_linear(value: dgen.Value) -> bool:
    """Whether *value* is of a ``Linear``-trait type."""
    return linearity(value) is Linearity.LINEAR


def is_affine_or_linear(value: dgen.Value) -> bool:
    """The verifier's main predicate — True for any value subject to
    resource discipline (consume-at-most-once or consume-exactly-once)."""
    return linearity(value) is not Linearity.UNRESTRICTED


class _State(enum.Enum):
    """Per-value state in the verifier's typing context Γ."""

    AVAILABLE = "available"
    MAYBE_AVAILABLE = "maybe_available"
    CONSUMED = "consumed"


def _diverges(block: Block) -> bool:
    """Whether *block* never returns control to its parent.

    Proxied by a ``Never`` result type — same proxy used by
    ``ControlFlowToGoto``; see the terminator-check TODO in ``TODO.md``.

    An *unresolved* result type (still a plain ``Value``, not yet a
    resolved ``Type``) reads as completing. That direction is safe:
    every rule consulting this treats "completing" as the demanding
    case, so misreading a diverging block as completing can only reject
    more programs (e.g. push ``alternatives`` toward its
    mixed-consumption error), never accept a leak.
    """
    return isinstance(block.result.type, Never)


def _consume_at(
    gamma: dict[dgen.Value, _State],
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
    if gamma[value] is _State.CONSUMED:
        raise DoubleConsumeError(
            f"{type(value).__name__} %{value.name} ({linearity(value).value}) "
            f"consumed twice; second consume by {type(by).__name__} "
            f"%{by.name}\n\n" + _annotated_asm(root, value)
        )
    gamma[value] = _State.CONSUMED


def _capture_into_unknown(
    gamma: dict[dgen.Value, _State],
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
    if gamma[value] is _State.CONSUMED:
        raise DoubleConsumeError(
            f"{type(value).__name__} %{value.name} ({linearity(value).value}) "
            f"captured into {type(by).__name__} %{by.name} after being "
            f"consumed\n\n" + _annotated_asm(root, value)
        )
    if gamma[value] is _State.AVAILABLE:
        gamma[value] = _State.MAYBE_AVAILABLE


class BlockLinearityContext:
    """The API handed to ``Op.verify_block_linearity``.

    Exposes primitives for charging captured values in the enclosing
    block's Γ, plus the standard implementations for the three
    block-execution traits declared in ``builtin.dgen``
    (``ExactlyOnce`` / ``Alternatives`` / ``BodyWithHandler``). The
    default ``Op.verify_block_linearity`` calls :meth:`from_traits`;
    block-holding ops with bespoke semantics override the method and
    compose the primitives instead.

    Throughout, **linear** captures charge precisely while **affine**
    captures keep the permissive ``MAYBE_AVAILABLE`` treatment: an
    affine value (a raise handler, an exit label) is legitimately
    captured by many sibling scopes — at most one of the captured uses
    fires per runtime path, and charging ``CONSUMED`` at the first
    scope would reject the rest.
    """

    def __init__(
        self,
        gamma: dict[dgen.Value, _State],
        op: dgen.Op,
        root: dgen.Value,
    ) -> None:
        self._gamma = gamma
        self._op = op
        self._root = root

    # -- primitives --------------------------------------------------------

    def consume(self, value: dgen.Value) -> None:
        """Charge *value* ``CONSUMED`` (rejecting a double-consume)."""
        _consume_at(self._gamma, value, by=self._op, root=self._root)

    def park(self, value: dgen.Value) -> None:
        """Park *value* at ``MAYBE_AVAILABLE`` (rejecting
        capture-after-consume) — the unknown-semantics treatment."""
        _capture_into_unknown(self._gamma, value, by=self._op, root=self._root)

    def _collected_captures(
        self,
    ) -> tuple[dict[dgen.Value, list[Block]], set[dgen.Value]]:
        """Substructural captures of the op's children:
        ``({linear value: capturing blocks}, {affine values})``."""
        linear_capturing: dict[dgen.Value, list[Block]] = {}
        affine_caps: set[dgen.Value] = set()
        for _, child in self._op.blocks:
            for cap in child.captures:
                if is_linear(cap):
                    linear_capturing.setdefault(cap, []).append(child)
                elif is_affine_or_linear(cap):
                    affine_caps.add(cap)
        return linear_capturing, affine_caps

    # -- standard trait implementations ------------------------------------

    def from_traits(self) -> None:
        """Dispatch on the op's declared block-execution trait
        (``builtin.dgen``); an op declaring none is conservative."""
        op = self._op
        if op.has_trait(_EXACTLY_ONCE_TRAIT):
            self.exactly_once()
        elif op.has_trait(_ALTERNATIVES_TRAIT):
            self.alternatives()
        elif op.has_trait(_BODY_WITH_HANDLER_TRAIT):
            self.body_with_handler()
        else:
            self.conservative()

    def conservative(self) -> None:
        """Unknown block semantics: park every substructural capture."""
        linear_capturing, affine_caps = self._collected_captures()
        for cap in affine_caps | set(linear_capturing):
            self.park(cap)

    def exactly_once(self) -> None:
        """Every owned block runs exactly once. Each capturing block
        consumes its linear captures (its own local verification
        enforces that), so one capturing child consumes at the op and
        two capturing children is a static double-consume."""
        linear_capturing, affine_caps = self._collected_captures()
        for cap in affine_caps:
            self.park(cap)
        for cap, capturing in linear_capturing.items():
            if len(capturing) > 1:
                raise DoubleConsumeError(
                    f"linear {type(cap).__name__} %{cap.name} captured by "
                    f"{len(capturing)} blocks of {type(self._op).__name__} "
                    f"%{self._op.name}, each of which runs\n\n"
                    + _annotated_asm(self._root, cap)
                )
            self.consume(cap)

    def alternatives(self) -> None:
        """Exactly one owned block runs; the blocks never transfer
        control into each other. For each linear capture:

        - Every non-capturing child diverges → every *completing* path
          consumes the value → charge ``CONSUMED`` (one charge, deduped
          across children — branch composition).
        - Every capturing child diverges → the value is consumed only
          on paths that never return → the parent's thread continues
          (divergence-aware branch composition; capture-after-consume
          is still rejected).
        - Otherwise some completing path consumes and another leaks —
          reject.
        """
        children = [child for _, child in self._op.blocks]
        linear_capturing, affine_caps = self._collected_captures()
        for cap in affine_caps:
            self.park(cap)
        for cap, capturing in linear_capturing.items():
            noncapturing = [c for c in children if c not in capturing]
            if all(_diverges(c) for c in noncapturing):
                self.consume(cap)
            elif all(_diverges(c) for c in capturing):
                if self._gamma.get(cap) is _State.CONSUMED:
                    raise DoubleConsumeError(
                        f"linear {type(cap).__name__} %{cap.name} captured "
                        f"into {type(self._op).__name__} %{self._op.name} "
                        f"after being consumed\n\n" + _annotated_asm(self._root, cap)
                    )
            else:
                raise LinearLeakError(
                    f"linear {type(cap).__name__} %{cap.name} is captured "
                    f"by only some completing alternatives of "
                    f"{type(self._op).__name__} %{self._op.name} — it leaks "
                    f"on the alternatives that neither capture it nor "
                    f"diverge\n\n" + _annotated_asm(self._root, cap)
                )

    def body_with_handler(self) -> None:
        """A body block that always starts plus a handler block that
        runs iff the body diverges into it. Unlike ``alternatives``,
        the body may consume a capture *before* diverging (at-site
        discharge), so a body-only capture cannot be charged precisely;
        only the cleanup-scope pattern — every child captures the
        value — charges ``CONSUMED``."""
        children = [child for _, child in self._op.blocks]
        linear_capturing, affine_caps = self._collected_captures()
        for cap in affine_caps:
            self.park(cap)
        for cap, capturing in linear_capturing.items():
            if len(capturing) == len(children):
                self.consume(cap)
            else:
                self.park(cap)


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
    gamma: dict[dgen.Value, _State] = {}
    for source in (block.args, block.parameters, block.captures):
        for v in source:
            if is_affine_or_linear(v):
                gamma[v] = _State.AVAILABLE

    for v in block.values:
        if not isinstance(v, dgen.Op):
            continue  # leaves: BlockArg / BlockParam / Constant / type
        # Op consumes its substructural operands and parameters.
        for source in (v.operands, v.parameters):
            for _, dep in source:
                if is_affine_or_linear(dep):
                    _consume_at(gamma, dep, by=v, root=root)
        # Captures into child blocks charge through the op's
        # ``verify_block_linearity`` protocol — trait-declared contracts
        # charge linear captures precisely; everything else parks
        # captures at ``MAYBE_AVAILABLE``. Captures dedup across
        # alternative children of one op (branch-composition).
        if v.blocks:
            v.verify_block_linearity(BlockLinearityContext(gamma, v, root))
        # Each child block verified independently with its own Γ_in.
        for _, child in v.blocks:
            _verify_linearity_block(child, root)
        # Op result, if substructural, becomes Available.
        if is_affine_or_linear(v):
            gamma[v] = _State.AVAILABLE

    # Exit check: block.result is "yielded" to the surrounding scope —
    # being the block's output IS the consumption. ``MAYBE_AVAILABLE``
    # is treated permissively (the inner block may already have
    # consumed it). Anything still ``AVAILABLE`` and ``LINEAR`` (and
    # not the block result) is a leak.
    for value, state in gamma.items():
        if state is not _State.AVAILABLE:
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
