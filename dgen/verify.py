"""IR invariant verification utilities."""

from __future__ import annotations

import json
import operator
from collections.abc import Callable

import dgen
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialect import Dialect
from dgen.dialects.function import FunctionOp
from dgen.gen.ast import (
    Constraint,
    ExpressionConstraint,
    HasTraitConstraint,
    HasTypeConstraint,
)
from dgen.gen.expression import (
    Attribute,
    Binary,
    Expression,
    Literal,
    Name,
    TypeOf,
    Unary,
)
from dgen.graph import all_blocks, all_values
from dgen.module import asm_with_imports
from dgen.trait import Trait
from dgen.type import Constant, type_constant


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
#
# A constraint is decided when every value it reads is a constant or a type.
# See the dialect files design doc, "Decided and undecided".


class _Undecided(Exception):
    """A constraint reads a value that isn't known yet."""


class _Invalid(Exception):
    """A constraint can't be evaluated on the values it reads."""


_NUMERIC: dict[str, Callable[[float, float], object]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
}


def _known(value: dgen.Value) -> bool:
    return isinstance(value, (Constant, dgen.Type))


def _resolve_subject(subject: str, op: dgen.Value) -> dgen.Value:
    """Resolve a constraint subject name to the value of that operand/param."""
    for name, value in (*op.operands, *op.parameters):
        if name == subject:
            return value
    raise ConstraintError(
        f"constraint references unknown subject {subject!r} on "
        f"{type(op).__name__} %{op.name}"
    )


def _type_of(value: dgen.Value, text: str) -> dgen.Type:
    if not _known(value.type):
        raise _Undecided(f"the type of {text} is not known yet")
    return type_constant(value.type)


def _subject_type(subject: dgen.Value, text: str) -> dgen.Type:
    """The subject itself when it is a type, else its type."""
    return subject if isinstance(subject, dgen.Type) else _type_of(subject, text)


def _resolve_type(name: str, op: dgen.Value) -> type[dgen.Type]:
    """Look up a type or trait class by bare or dialect-qualified name."""
    dialect_name, _, type_name = name.rpartition(".")
    dialects = (
        [Dialect.get(dialect_name)] if dialect_name else Dialect._registry.values()
    )
    for dialect in dialects:
        if (cls := dialect.types.get(type_name)) is not None:
            return cls
    raise ConstraintError(
        f"unknown type {name!r} referenced in constraint on "
        f"{type(op).__name__} %{op.name}"
    )


def _resolve_trait(trait_name: str, op: dgen.Value) -> type[Trait]:
    """Look up a trait class by name from the dialect registry."""
    for dialect in Dialect._registry.values():
        cls = dialect.types.get(trait_name)
        if cls is not None and issubclass(cls, Trait):
            return cls
    raise ConstraintError(
        f"unknown trait {trait_name!r} referenced in constraint on "
        f"{type(op).__name__} %{op.name}"
    )


def _value(expression: Expression, op: dgen.Value) -> dgen.Value:
    """The value a name, `$name` or attribute path reads."""
    match expression:
        case Name(name):
            return _resolve_subject(name, op)
        case TypeOf(name):
            return _type_of(_resolve_subject(name, op), name)
        case Attribute(inner, name):
            owner = _subject_type(_value(inner, op), str(inner))
            if (param := dict(owner.parameters).get(name)) is None:
                raise _Undecided(
                    f"{inner}: {type(owner).__name__} has no parameter {name!r}"
                )
            return param
        case _:
            raise _Invalid(f"{expression} is not a value")


def _json(expression: Expression, op: dgen.Value, reads: dict[str, str]) -> object:
    """Evaluate an expression, recording each value it reads in `reads`."""
    match expression:
        case Literal(value):
            return value
        case Unary("not", operand):
            return not _json(operand, op, reads)
        case Unary(_, operand):
            return -_number(operand, op, reads)
        case Binary("and", lhs, rhs):
            return _json(lhs, op, reads) and _json(rhs, op, reads)
        case Binary("or", lhs, rhs):
            return _json(lhs, op, reads) or _json(rhs, op, reads)
        case Binary("==", lhs, rhs):
            return _json(lhs, op, reads) == _json(rhs, op, reads)
        case Binary("!=", lhs, rhs):
            return _json(lhs, op, reads) != _json(rhs, op, reads)
        case Binary(symbol, lhs, rhs):
            left, right = _number(lhs, op, reads), _number(rhs, op, reads)
            if symbol in ("//", "%") and right == 0:
                raise _Invalid(f"{expression} divides by zero")
            return _NUMERIC[symbol](left, right)
        case _:
            value = _value(expression, op)
            if not _known(value):
                raise _Undecided(
                    f"{expression} is %{value.name}, which is not known until run time"
                )
            data = value.__constant__.to_json()
            reads[str(expression)] = (
                value.format_asm() if isinstance(value, dgen.Type) else json.dumps(data)
            )
            return data


def _number(
    expression: Expression, op: dgen.Value, reads: dict[str, str]
) -> int | float:
    match value := _json(expression, op, reads):
        case bool():
            raise _Invalid(f"{expression} is {value}, not a number")
        case int() | float():
            return value
        case _:
            raise _Invalid(f"{expression} is {json.dumps(value)}, not a number")


def _failure(constraint: Constraint, op: dgen.Value) -> str | None:
    """Why the constraint is false on op, or None if it holds."""
    match constraint:
        case HasTraitConstraint(lhs, trait):
            subject_type = _subject_type(_resolve_subject(lhs, op), lhs)
            if isinstance(subject_type, _resolve_trait(trait, op)):
                return None
            return (
                f"subject {lhs!r} ({type(subject_type).__name__}) "
                f"does not implement trait {trait}"
            )
        case HasTypeConstraint(lhs, type_ref):
            subject_type = _subject_type(_resolve_subject(lhs, op), lhs)
            if isinstance(subject_type, _resolve_type(type_ref.name, op)):
                return None
            return (
                f"requires {constraint} is false "
                f"({lhs} has type {subject_type.format_asm()})"
            )
        case ExpressionConstraint():
            reads: dict[str, str] = {}
            match _json(constraint.expression, op, reads):
                case True:
                    return None
                case False:
                    values = ", ".join(f"{k} = {v}" for k, v in reads.items())
                    return f"requires {constraint} is false ({values})"
                case result:
                    return (
                        f"requires {constraint} is {json.dumps(result)}, not a boolean"
                    )
        case _:
            raise ConstraintError(f"unknown constraint {constraint!r}")


def verify_op_constraints(op: dgen.Value, root: dgen.Value, *, decided: bool) -> None:
    """Check op's constraints; undecided ones fail only when `decided` is set."""
    for constraint in op.__constraints__:
        try:
            failure = _failure(constraint, op)
        except _Undecided as undecided:
            if not decided:
                continue
            failure = f"requires {constraint} is undecided: {undecided}"
        except _Invalid as invalid:
            failure = f"requires {constraint} can't be evaluated: {invalid}"
        if failure is not None:
            raise ConstraintError(
                f"{type(op).__name__} %{op.name}: {failure}\n\n"
                + _annotated_asm(root, op)
            )


def verify_constraints(root: dgen.Value, *, decided: bool = False) -> None:
    """Check the constraints of every op reachable from root."""
    for value in all_values(root):
        if isinstance(value, dgen.Op):
            verify_op_constraints(value, root, decided=decided)
