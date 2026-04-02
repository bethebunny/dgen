"""Tests for trait infrastructure: Trait base class, Value.has_trait(), verify_constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value, layout
from dgen.dialects.function import Function, FunctionOp
from dgen.gen.ast import (
    ExpressionConstraint,
    HasTraitConstraint,
    HasTypeConstraint,
    TypeRef,
)
from dgen.gen.parser import parse
from dgen.module import Module
from dgen.type import Fields
from dgen.verify import ConstraintError, verify_constraints


# -- Fixtures: a tiny dialect with traits, types, and ops --------------------

_test = Dialect("_trait_test")


@_test.trait("Numeric")
class Numeric(Trait):
    pass


@_test.trait("Ordered")
class Ordered(Trait):
    pass


@_test.type("MyInt")
@dataclass(frozen=True, eq=False)
class MyInt(Numeric, Ordered, Type):
    __layout__ = layout.Int()


@_test.type("MyStr")
@dataclass(frozen=True, eq=False)
class MyStr(Type):
    __layout__ = layout.String()


@_test.op("add_nums")
@dataclass(eq=False, kw_only=True)
class AddNumsOp(Numeric, Op):
    lhs: Value
    rhs: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("lhs", Type), ("rhs", Type))


@_test.op("requires_numeric")
@dataclass(eq=False, kw_only=True)
class RequiresNumericOp(Op):
    """An op with a trait constraint: operand 'input' must have Numeric trait."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (HasTraitConstraint(lhs="input", trait="Numeric"),)


@_test.op("requires_ordered")
@dataclass(eq=False, kw_only=True)
class RequiresOrderedOp(Op):
    """An op with a trait constraint: operand 'input' must have Ordered trait."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (HasTraitConstraint(lhs="input", trait="Ordered"),)


@_test.op("requires_both")
@dataclass(eq=False, kw_only=True)
class RequiresBothOp(Op):
    """An op requiring both Numeric and Ordered traits on its operand."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (
        HasTraitConstraint(lhs="input", trait="Numeric"),
        HasTraitConstraint(lhs="input", trait="Ordered"),
    )


@_test.op("requires_param_trait")
@dataclass(eq=False, kw_only=True)
class RequiresParamTraitOp(Op):
    """An op with a trait constraint on a compile-time parameter."""

    kind: Value[TypeType]
    type: Type = MyInt()
    __params__: ClassVar[Fields] = (("kind", TypeType),)
    __constraints__ = (HasTraitConstraint(lhs="kind", trait="Numeric"),)


@_test.op("no_constraints")
@dataclass(eq=False, kw_only=True)
class NoConstraintsOp(Op):
    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)


def _make_module(*body_ops: Op) -> Module:
    """Build a minimal Module wrapping ops inside a function body."""
    result = body_ops[-1] if body_ops else MyInt().constant(0)
    func = FunctionOp(
        name="test_fn",
        body=Block(result=result, args=[]),
        result=MyInt(),
        type=Function(result=MyInt()),
    )
    return Module(ops=[func])


# -- Trait base class --------------------------------------------------------


def test_trait_inherits_from_base() -> None:
    assert issubclass(Numeric, Trait)
    assert issubclass(Ordered, Trait)


def test_trait_registered_in_dialect_types() -> None:
    """Traits are stored in dialect.types alongside regular types."""
    assert "Numeric" in _test.types
    assert _test.types["Numeric"] is Numeric
    assert "Ordered" in _test.types


# -- Value.has_trait() on types -----------------------------------------------


def test_has_trait_type_positive() -> None:
    assert MyInt().has_trait(Numeric)
    assert MyInt().has_trait(Ordered)


def test_has_trait_type_negative() -> None:
    assert not MyStr().has_trait(Numeric)
    assert not MyStr().has_trait(Ordered)


# -- Value.has_trait() on ops ------------------------------------------------


def test_has_trait_op_own_traits() -> None:
    """has_trait on an op checks the op's own traits."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    # AddNumsOp has Numeric in its class hierarchy, not Ordered
    assert op.has_trait(Numeric)
    assert not op.has_trait(Ordered)


def test_has_trait_op_result_type() -> None:
    """op.type.has_trait checks the result type's traits."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    # The result type MyInt implements both Numeric and Ordered
    assert op.type.has_trait(Numeric)
    assert op.type.has_trait(Ordered)


# -- .dgen built traits inherit from Trait -----------------------------------


def test_dgen_built_trait_inherits() -> None:
    """Traits built from .dgen files inherit from Trait."""
    from dgen.dialects import algebra

    assert issubclass(algebra.AddMagma, Trait)
    assert issubclass(algebra.MulMagma, Trait)
    assert issubclass(algebra.TotalOrder, Trait)


def test_dgen_built_trait_in_dialect_types() -> None:
    """Traits built from .dgen are in dialect.types."""
    from dgen.dialects import algebra

    assert "AddMagma" in algebra.algebra.types
    assert issubclass(algebra.algebra.types["AddMagma"], Trait)


# -- Parser: has trait / has type constraints --------------------------------


def test_parse_has_trait_constraint() -> None:
    src = "op foo(x) -> Type:\n    requires x has trait Numeric\n"
    op = parse(src).ops[0]
    assert op.constraints == [HasTraitConstraint(lhs="x", trait="Numeric")]


def test_parse_type_body_requires() -> None:
    src = "type Foo<t: Type>:\n    requires t has trait Numeric\n"
    td = parse(src).types[0]
    assert td.constraints == [HasTraitConstraint(lhs="t", trait="Numeric")]


# -- verify_constraints: satisfied constraints -------------------------------


def test_verify_operand_trait_satisfied() -> None:
    """Constraint passes when operand type implements the required trait."""
    c = MyInt().constant(42)
    op = RequiresNumericOp(input=c)
    module = _make_module(op)
    verify_constraints(module)  # should not raise


def test_verify_both_traits_satisfied() -> None:
    """Both trait constraints pass when operand type implements both."""
    c = MyInt().constant(42)
    op = RequiresBothOp(input=c)
    module = _make_module(op)
    verify_constraints(module)  # should not raise


def test_verify_no_constraints_passes() -> None:
    """Ops without constraints pass verification."""
    c = MyStr().constant("hello")
    op = NoConstraintsOp(input=c)
    module = _make_module(op)
    verify_constraints(module)  # should not raise


def test_verify_param_trait_satisfied() -> None:
    """Constraint passes when a compile-time parameter's type has the trait."""
    op = RequiresParamTraitOp(kind=MyInt())
    module = _make_module(op)
    verify_constraints(module)  # should not raise


# -- verify_constraints: violated constraints --------------------------------


def test_verify_operand_trait_violated() -> None:
    """Constraint fails when operand type does not implement the required trait."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c)
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(module)


def test_verify_ordered_trait_violated() -> None:
    """MyStr does not implement Ordered — constraint fails."""
    c = MyStr().constant("hello")
    op = RequiresOrderedOp(input=c)
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="does not implement trait Ordered"):
        verify_constraints(module)


def test_verify_one_of_two_traits_violated() -> None:
    """When one of two constraints fails, verification raises."""
    # MyStr lacks both Numeric and Ordered — first failure is Numeric
    c = MyStr().constant("hello")
    op = RequiresBothOp(input=c)
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(module)


def test_verify_param_trait_violated() -> None:
    """Constraint fails when parameter type does not have the trait."""
    op = RequiresParamTraitOp(kind=MyStr())
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(module)


def test_verify_error_names_op() -> None:
    """Error message includes the op class name and op name."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="bad_op")
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad_op"):
        verify_constraints(module)


def test_verify_error_names_operand_and_type() -> None:
    """Error message includes the operand name and actual type."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="v0")
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="operand 'input' has type MyStr"):
        verify_constraints(module)


# -- verify_constraints: unknown subject / trait -----------------------------


def test_verify_unknown_subject_raises() -> None:
    """Constraint referencing a non-existent operand name raises."""

    @dataclass(eq=False, kw_only=True)
    class BadSubjectOp(Op):
        input: Value
        type: Type = MyInt()
        __operands__: ClassVar[Fields] = (("input", Type),)
        __constraints__ = (HasTraitConstraint(lhs="nonexistent", trait="Numeric"),)

    _test.op("bad_subject")(BadSubjectOp)

    c = MyInt().constant(42)
    op = BadSubjectOp(input=c)
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="unknown subject 'nonexistent'"):
        verify_constraints(module)


def test_verify_unknown_trait_raises() -> None:
    """Constraint referencing a non-existent trait name raises."""

    @dataclass(eq=False, kw_only=True)
    class BadTraitOp(Op):
        input: Value
        type: Type = MyInt()
        __operands__: ClassVar[Fields] = (("input", Type),)
        __constraints__ = (HasTraitConstraint(lhs="input", trait="NoSuchTrait"),)

    _test.op("bad_trait")(BadTraitOp)

    c = MyInt().constant(42)
    op = BadTraitOp(input=c)
    module = _make_module(op)
    with pytest.raises(ConstraintError, match="unknown trait 'NoSuchTrait'"):
        verify_constraints(module)


# -- verify_constraints: mixed ops in a module -------------------------------


def test_verify_mixed_ops_first_bad() -> None:
    """Module with one good op and one bad op: verification catches the bad one."""
    good = RequiresNumericOp(input=MyInt().constant(1), name="good")
    bad = RequiresNumericOp(input=MyStr().constant("x"), name="bad")
    # Both ops in the same function body — result is the last one
    from dgen.dialects.builtin import ChainOp

    chain = ChainOp(lhs=bad, rhs=good, type=MyInt())
    module = _make_module(chain)
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad"):
        verify_constraints(module)


# -- verify_constraints: expression constraints (not yet implemented) --------


@_test.op("requires_positive")
@dataclass(eq=False, kw_only=True)
class RequiresPositiveOp(Op):
    """Op with an expression constraint: requires value > 0."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (ExpressionConstraint(expr="input > 0"),)


@_test.op("requires_equal_types")
@dataclass(eq=False, kw_only=True)
class RequiresEqualTypesOp(Op):
    """Op with an expression constraint: requires lhs type == rhs type."""

    lhs: Value
    rhs: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("lhs", Type), ("rhs", Type))
    __constraints__ = (ExpressionConstraint(expr="$lhs == $rhs"),)


@_test.op("requires_match")
@dataclass(eq=False, kw_only=True)
class RequiresMatchOp(Op):
    """Op with a match constraint: requires input has type MyInt."""

    input: Value
    type: Type = MyInt()
    __operands__: ClassVar[Fields] = (("input", Type),)
    __constraints__ = (HasTypeConstraint(lhs="input", type=TypeRef("MyInt")),)


@pytest.mark.xfail(
    strict=True, reason="expression constraint verification not yet implemented"
)
def test_verify_expression_constraint_violated() -> None:
    """Expression constraint should fail when condition is false."""
    op = RequiresPositiveOp(input=MyInt().constant(-1))
    module = _make_module(op)
    with pytest.raises(ConstraintError):
        verify_constraints(module)


@pytest.mark.xfail(
    strict=True, reason="expression constraint verification not yet implemented"
)
def test_verify_type_equality_expression_violated() -> None:
    """Type equality expression should fail when operand types differ."""
    op = RequiresEqualTypesOp(lhs=MyInt().constant(1), rhs=MyStr().constant("x"))
    module = _make_module(op)
    with pytest.raises(ConstraintError):
        verify_constraints(module)


@pytest.mark.xfail(
    strict=True, reason="match constraint verification not yet implemented"
)
def test_verify_match_constraint_violated() -> None:
    """Match constraint should fail when operand type doesn't match."""
    op = RequiresMatchOp(input=MyStr().constant("x"))
    module = _make_module(op)
    with pytest.raises(ConstraintError):
        verify_constraints(module)
