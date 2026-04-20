"""Tests for trait infrastructure: ``Trait`` marker, ``has_trait``, and
constraint verification.

v1 model: traits are types (they inherit from ``Trait`` which inherits from
``Type``). A type declares traits via a ``_declared_traits(self)`` method
installed by the builder (or hand-written); ``has_trait(trait_instance)``
walks the class MRO and compares structurally via ``to_json()``. No
inheritance from trait classes, no ``isinstance`` dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from typing import ClassVar

import pytest

from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value, layout
from dgen.builtins import pack
from dgen.dialects.function import Function, FunctionOp
from dgen.ir.verification import ConstraintError, verify_constraints
from dgen.ir.constraints import has_trait
from dgen.spec.ast import (
    ExpressionConstraint,
    HasTraitConstraint,
    HasTypeConstraint,
    TypeRef,
)
from dgen.spec.parser import parse
from dgen.type import Fields


# -- Fixtures: a tiny dialect with traits, types, and ops --------------------

_test = Dialect("_trait_test")


@_test.trait("Numeric")
@dataclass(frozen=True, eq=False)
class Numeric(Trait):
    __layout__ = layout.Void()


@_test.trait("Ordered")
@dataclass(frozen=True, eq=False)
class Ordered(Trait):
    __layout__ = layout.Void()


@_test.type("MyInt")
@dataclass(frozen=True, eq=False)
class MyInt(Type):
    __layout__ = layout.Int()

    @property
    def traits(self) -> Iterator[Type]:
        yield Numeric()
        yield Ordered()
        yield from super().traits


@_test.type("MyStr")
@dataclass(frozen=True, eq=False)
class MyStr(Type):
    __layout__ = layout.String()


@_test.op("add_nums")
@dataclass(eq=False, kw_only=True)
class AddNumsOp(Op):
    """Op that declares itself ``Numeric`` — has_trait on the op value works."""

    lhs: Value
    rhs: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("lhs", Type), ("rhs", Type))

    @property
    def traits(self) -> Iterator[Type]:
        yield Numeric()
        yield from super().traits


@_test.op("requires_numeric")
@dataclass(eq=False, kw_only=True)
class RequiresNumericOp(Op):
    """Trait constraint: operand 'input' must have Numeric trait."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (has_trait("input", Numeric()),)


@_test.op("requires_ordered")
@dataclass(eq=False, kw_only=True)
class RequiresOrderedOp(Op):
    """Trait constraint: operand 'input' must have Ordered trait."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (has_trait("input", Ordered()),)


@_test.op("requires_both")
@dataclass(eq=False, kw_only=True)
class RequiresBothOp(Op):
    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (
        has_trait("input", Numeric()),
        has_trait("input", Ordered()),
    )


@_test.op("requires_param_trait")
@dataclass(eq=False, kw_only=True)
class RequiresParamTraitOp(Op):
    kind: Value[TypeType]
    type: Type = MyInt()
    __params__: ClassVar[Fields] = (("kind", TypeType),)
    __constraints__ = (has_trait("kind", Numeric()),)


@_test.op("no_constraints")
@dataclass(eq=False, kw_only=True)
class NoConstraintsOp(Op):
    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)


def _make_function(*body_ops: Op) -> FunctionOp:
    """Build a minimal FunctionOp wrapping ops inside a function body."""
    result = body_ops[-1] if body_ops else MyInt().constant(0)
    return FunctionOp(
        name="test_fn",
        body=Block(result=result, args=[]),
        result_type=MyInt(),
        type=Function(arguments=pack(), result_type=MyInt()),
    )


# -- Trait class registration ------------------------------------------------


def test_trait_class_inherits_from_base() -> None:
    assert issubclass(Numeric, Trait)
    assert issubclass(Ordered, Trait)


def test_trait_registered_in_dialect_types() -> None:
    """Traits are stored in dialect.types alongside regular types — a trait
    is just a type with ``layout Void`` semantics."""
    assert "Numeric" in _test.types
    assert _test.types["Numeric"] is Numeric
    assert "Ordered" in _test.types


# -- has_trait ---------------------------------------------------------------


def test_has_trait_on_types() -> None:
    """``has_trait(Foo())`` on a type matches traits in its declared list."""
    assert MyInt().has_trait(Numeric())
    assert MyInt().has_trait(Ordered())
    assert not MyStr().has_trait(Numeric())


def test_has_trait_on_ops() -> None:
    """Ops can declare their own traits via ``_declared_traits``."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    assert op.has_trait(Numeric())
    assert not op.has_trait(Ordered())


def test_has_trait_result_type() -> None:
    """``op.type.has_trait`` checks the result type, not the op itself."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    assert op.type.has_trait(Numeric())
    assert op.type.has_trait(Ordered())


def test_has_trait_on_constants_via_type() -> None:
    """Constants don't declare traits themselves; ask their type."""
    c = MyInt().constant(42)
    assert c.type.has_trait(Numeric())
    assert c.type.has_trait(Ordered())
    assert not MyStr().constant("x").type.has_trait(Numeric())


# -- .dgen-built traits still register -------------------------------------


def test_dgen_built_trait_inherits_from_trait() -> None:
    """Traits built from .dgen files remain subclasses of ``Trait``."""
    from dgen.dialects import algebra

    assert issubclass(algebra.AddMagma, Trait)
    assert issubclass(algebra.MulMagma, Trait)
    assert issubclass(algebra.TotalOrder, Trait)


def test_dgen_built_trait_in_dialect_types() -> None:
    from dgen.dialects import algebra

    assert "AddMagma" in algebra.algebra.types
    assert issubclass(algebra.algebra.types["AddMagma"], Trait)


# -- Parser: has trait / has type constraints --------------------------------


def test_parse_has_trait_constraint() -> None:
    src = "op foo(x) -> Type:\n    requires x has trait Numeric\n"
    op = parse(src).ops[0]
    assert op.constraints == [
        HasTraitConstraint(lhs="x", trait=TypeRef(name="Numeric"))
    ]


def test_parse_type_body_requires() -> None:
    src = "type Foo<t: Type>:\n    requires t has trait Numeric\n"
    td = parse(src).types[0]
    assert td.constraints == [
        HasTraitConstraint(lhs="t", trait=TypeRef(name="Numeric"))
    ]


# -- verify_constraints: satisfied constraints -------------------------------


def test_verify_operand_trait_satisfied() -> None:
    c = MyInt().constant(42)
    op = RequiresNumericOp(input=c)
    verify_constraints(_make_function(op))


def test_verify_both_traits_satisfied() -> None:
    c = MyInt().constant(42)
    op = RequiresBothOp(input=c)
    verify_constraints(_make_function(op))


def test_verify_no_constraints_passes() -> None:
    c = MyStr().constant("hello")
    op = NoConstraintsOp(input=c)
    verify_constraints(_make_function(op))


def test_verify_param_trait_satisfied() -> None:
    op = RequiresParamTraitOp(kind=MyInt())
    verify_constraints(_make_function(op))


# -- verify_constraints: violated constraints --------------------------------


def test_verify_operand_trait_violated() -> None:
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c)
    with pytest.raises(ConstraintError, match="does not implement trait .*Numeric"):
        verify_constraints(_make_function(op))


def test_verify_ordered_trait_violated() -> None:
    c = MyStr().constant("hello")
    op = RequiresOrderedOp(input=c)
    with pytest.raises(ConstraintError, match="does not implement trait .*Ordered"):
        verify_constraints(_make_function(op))


def test_verify_one_of_two_traits_violated() -> None:
    c = MyStr().constant("hello")
    op = RequiresBothOp(input=c)
    with pytest.raises(ConstraintError, match="does not implement trait .*Numeric"):
        verify_constraints(_make_function(op))


def test_verify_param_trait_violated() -> None:
    op = RequiresParamTraitOp(kind=MyStr())
    with pytest.raises(ConstraintError, match="does not implement trait .*Numeric"):
        verify_constraints(_make_function(op))


def test_verify_error_names_op() -> None:
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="bad_op")
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad_op"):
        verify_constraints(_make_function(op))


def test_verify_error_names_operand_and_type() -> None:
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="v0")
    with pytest.raises(ConstraintError, match="subject 'input'.*does not implement"):
        verify_constraints(_make_function(op))


# -- verify_constraints: unknown subject / trait -----------------------------


def test_verify_unknown_subject_raises() -> None:
    @dataclass(eq=False, kw_only=True)
    class BadSubjectOp(Op):
        input: Value
        type: Type = MyInt()
        __operands__: ClassVar[Fields] = (("input", Type),)
        __constraints__ = (has_trait("nonexistent", Numeric()),)

    _test.op("bad_subject")(BadSubjectOp)

    op = BadSubjectOp(input=MyInt().constant(42))
    with pytest.raises(ConstraintError, match="unknown subject 'nonexistent'"):
        verify_constraints(_make_function(op))


# Note: a "trait name doesn't exist" error is now a build-time concern
# (``_trait_fn`` raises when ``_resolve_type`` can't find the name) — see
# the spec/builder tests rather than runtime verification.


# -- verify_constraints: mixed ops in a fn -----------------------------------


def test_verify_mixed_ops_first_bad() -> None:
    good = RequiresNumericOp(input=MyInt().constant(1), name="good")
    bad = RequiresNumericOp(input=MyStr().constant("x"), name="bad")
    from dgen.dialects.builtin import ChainOp

    chain = ChainOp(lhs=bad, rhs=good, type=MyInt())
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad"):
        verify_constraints(_make_function(chain))


# -- verify_constraints: expression constraints (not yet implemented) --------


@_test.op("requires_positive")
@dataclass(eq=False, kw_only=True)
class RequiresPositiveOp(Op):
    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (ExpressionConstraint(expr="input > 0"),)


@_test.op("requires_equal_types")
@dataclass(eq=False, kw_only=True)
class RequiresEqualTypesOp(Op):
    lhs: Value
    rhs: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("lhs", Type), ("rhs", Type))
    __constraints__ = (ExpressionConstraint(expr="$lhs == $rhs"),)


@_test.op("requires_match")
@dataclass(eq=False, kw_only=True)
class RequiresMatchOp(Op):
    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (HasTypeConstraint(lhs="input", type=TypeRef("MyInt")),)


@pytest.mark.xfail(
    strict=True, reason="expression constraint verification not yet implemented"
)
def test_verify_expression_constraint_violated() -> None:
    op = RequiresPositiveOp(input=MyInt().constant(-1))
    with pytest.raises(ConstraintError):
        verify_constraints(_make_function(op))


@pytest.mark.xfail(
    strict=True, reason="expression constraint verification not yet implemented"
)
def test_verify_type_equality_expression_violated() -> None:
    op = RequiresEqualTypesOp(lhs=MyInt().constant(1), rhs=MyStr().constant("x"))
    with pytest.raises(ConstraintError):
        verify_constraints(_make_function(op))


@pytest.mark.xfail(
    strict=True, reason="match constraint verification not yet implemented"
)
def test_verify_match_constraint_violated() -> None:
    op = RequiresMatchOp(input=MyStr().constant("x"))
    with pytest.raises(ConstraintError):
        verify_constraints(_make_function(op))
