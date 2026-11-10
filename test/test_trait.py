"""Tests for trait infrastructure: Trait base class, Value.has_trait(), verify_constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value, layout
from dgen.block import BlockArgument
from dgen.compiler import Compiler, IdentityPass, verify_passes
from dgen.dialects.function import Function, FunctionOp
from dgen.gen.ast import (
    ExpressionConstraint,
    HasTraitConstraint,
    HasTypeConstraint,
    TypeRef,
)
from dgen.gen.parser import parse
from dgen.module import pack
from dgen.passes.pass_ import Pass, lowering_for
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


def _make_function(*body_ops: Op) -> FunctionOp:
    """Build a minimal FunctionOp wrapping ops inside a function body."""
    result = body_ops[-1] if body_ops else MyInt().constant(0)
    return FunctionOp(
        name="test_fn",
        body=Block(result=result, args=[]),
        result_type=MyInt(),
        type=Function(arguments=pack(), result_type=MyInt()),
    )


# -- Trait base class --------------------------------------------------------


def test_trait_inherits_from_base() -> None:
    assert issubclass(Numeric, Trait)
    assert issubclass(Ordered, Trait)


def test_trait_registered_in_dialect_types() -> None:
    """Traits are stored in dialect.types alongside regular types."""
    assert "Numeric" in _test.types
    assert _test.types["Numeric"] is Numeric
    assert "Ordered" in _test.types


# -- has_trait ---------------------------------------------------------------


def test_has_trait_on_types() -> None:
    """has_trait on a type checks isinstance(self, trait)."""
    assert MyInt().has_trait(Numeric)
    assert MyInt().has_trait(Ordered)
    assert not MyStr().has_trait(Numeric)


def test_has_trait_on_ops() -> None:
    """has_trait on an op checks the op's own class traits."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    # AddNumsOp inherits from Numeric, not Ordered
    assert op.has_trait(Numeric)
    assert not op.has_trait(Ordered)


def test_has_trait_result_type() -> None:
    """value.type.has_trait checks the result type's traits."""
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    # The result type MyInt implements both Numeric and Ordered
    assert op.type.has_trait(Numeric)
    assert op.type.has_trait(Ordered)


def test_has_trait_on_constants() -> None:
    """value.type.has_trait checks the constant's type traits."""
    c = MyInt().constant(42)
    assert c.type.has_trait(Numeric)
    assert c.type.has_trait(Ordered)
    assert not MyStr().constant("x").type.has_trait(Numeric)


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
    fn = _make_function(op)
    verify_constraints(fn)  # should not raise


def test_verify_both_traits_satisfied() -> None:
    """Both trait constraints pass when operand type implements both."""
    c = MyInt().constant(42)
    op = RequiresBothOp(input=c)
    fn = _make_function(op)
    verify_constraints(fn)  # should not raise


def test_verify_no_constraints_passes() -> None:
    """Ops without constraints pass verification."""
    c = MyStr().constant("hello")
    op = NoConstraintsOp(input=c)
    fn = _make_function(op)
    verify_constraints(fn)  # should not raise


def test_verify_param_trait_satisfied() -> None:
    """Constraint passes when a compile-time parameter's type has the trait."""
    op = RequiresParamTraitOp(kind=MyInt())
    fn = _make_function(op)
    verify_constraints(fn)  # should not raise


# -- verify_constraints: violated constraints --------------------------------


def test_verify_operand_trait_violated() -> None:
    """Constraint fails when operand type does not implement the required trait."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c)
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(fn)


def test_verify_ordered_trait_violated() -> None:
    """MyStr does not implement Ordered — constraint fails."""
    c = MyStr().constant("hello")
    op = RequiresOrderedOp(input=c)
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="does not implement trait Ordered"):
        verify_constraints(fn)


def test_verify_one_of_two_traits_violated() -> None:
    """When one of two constraints fails, verification raises."""
    # MyStr lacks both Numeric and Ordered — first failure is Numeric
    c = MyStr().constant("hello")
    op = RequiresBothOp(input=c)
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(fn)


def test_verify_param_trait_violated() -> None:
    """Constraint fails when parameter type does not have the trait."""
    op = RequiresParamTraitOp(kind=MyStr())
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="does not implement trait Numeric"):
        verify_constraints(fn)


def test_verify_error_names_op() -> None:
    """Error message includes the op class name and op name."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="bad_op")
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad_op"):
        verify_constraints(fn)


def test_verify_error_names_operand_and_type() -> None:
    """Error message includes the operand name and actual type."""
    c = MyStr().constant("hello")
    op = RequiresNumericOp(input=c, name="v0")
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="subject 'input'.*does not implement"):
        verify_constraints(fn)


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
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="unknown subject 'nonexistent'"):
        verify_constraints(fn)


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
    fn = _make_function(op)
    with pytest.raises(ConstraintError, match="unknown trait 'NoSuchTrait'"):
        verify_constraints(fn)


# -- verify_constraints: mixed ops in a fn -------------------------------


def test_verify_mixed_ops_first_bad() -> None:
    """Module with one good op and one bad op: verification catches the bad one."""
    good = RequiresNumericOp(input=MyInt().constant(1), name="good")
    bad = RequiresNumericOp(input=MyStr().constant("x"), name="bad")
    # Both ops in the same function body — result is the last one
    from dgen.dialects.builtin import ChainOp

    chain = ChainOp(lhs=bad, rhs=good, type=MyInt())
    fn = _make_function(chain)
    with pytest.raises(ConstraintError, match="RequiresNumericOp %bad"):
        verify_constraints(fn)


# -- verify_constraints: expression and has-type constraints -----------------


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


def test_verify_expression_constraint_violated() -> None:
    """Expression constraint should fail when condition is false."""
    op = RequiresPositiveOp(input=MyInt().constant(-1))
    fn = _make_function(op)
    with pytest.raises(ConstraintError):
        verify_constraints(fn)


def test_verify_type_equality_expression_violated() -> None:
    """Type equality expression should fail when operand types differ."""
    op = RequiresEqualTypesOp(lhs=MyInt().constant(1), rhs=MyStr().constant("x"))
    fn = _make_function(op)
    with pytest.raises(ConstraintError):
        verify_constraints(fn)


def test_verify_match_constraint_violated() -> None:
    """Match constraint should fail when operand type doesn't match."""
    op = RequiresMatchOp(input=MyStr().constant("x"))
    fn = _make_function(op)
    with pytest.raises(ConstraintError):
        verify_constraints(fn)


def test_verify_expression_constraint_satisfied() -> None:
    verify_constraints(_make_function(RequiresPositiveOp(input=MyInt().constant(1))))


def test_verify_expression_error_names_values() -> None:
    op = RequiresPositiveOp(input=MyInt().constant(-1), name="neg")
    with pytest.raises(
        ConstraintError,
        match=r"RequiresPositiveOp %neg: requires input > 0 is false \(input = -1\)",
    ):
        verify_constraints(_make_function(op))


def test_verify_type_equality_expression_satisfied() -> None:
    op = RequiresEqualTypesOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    verify_constraints(_make_function(op))


def test_verify_type_equality_error_names_types() -> None:
    op = RequiresEqualTypesOp(lhs=MyInt().constant(1), rhs=MyStr().constant("x"))
    with pytest.raises(
        ConstraintError,
        match=r"\$lhs == \$rhs is false \(\$lhs = _trait_test.MyInt, \$rhs = _trait_test.MyStr\)",
    ):
        verify_constraints(_make_function(op))


def test_verify_match_constraint_satisfied() -> None:
    verify_constraints(_make_function(RequiresMatchOp(input=MyInt().constant(1))))


def test_verify_match_error_names_type() -> None:
    op = RequiresMatchOp(input=MyStr().constant("x"))
    with pytest.raises(
        ConstraintError,
        match=r"requires input has type MyInt is false \(input has type _trait_test.MyStr\)",
    ):
        verify_constraints(_make_function(op))


def test_has_type_with_arguments_is_a_syntax_error() -> None:
    with pytest.raises(SyntaxError, match="without arguments"):
        HasTypeConstraint(lhs="x", type=TypeRef("Tensor", [TypeRef("Index")]))


@pytest.mark.parametrize(
    ("expr", "holds"),
    [
        ("input * 2 + 1 == 7", True),
        ("(input + 1) * 2 == 8", True),
        ("input // 2 == 1 and input % 2 == 1", True),
        ("-input < 0", True),
        ("input - 4 >= 0", False),
        ("input <= 3 and input != 3", False),
        ("input < 0 or input == 3", True),
        ("not input == 3", False),
    ],
)
def test_verify_expression_operators(expr: str, holds: bool) -> None:
    class CheckOp(RequiresPositiveOp):
        __constraints__ = (ExpressionConstraint(expr),)

    _test.op("check")(CheckOp)
    fn = _make_function(CheckOp(input=MyInt().constant(3)))
    if holds:
        verify_constraints(fn)
    else:
        with pytest.raises(ConstraintError, match=r"is false \(input = 3\)"):
            verify_constraints(fn)


@pytest.mark.parametrize(
    ("expr", "error"),
    [
        ("input + 1", "is 4, not a boolean"),
        ("input // 0 == 1", "divides by zero"),
        ("$input < 1", "is {.*}, not a number"),
    ],
)
def test_verify_expression_invalid(expr: str, error: str) -> None:
    class CheckOp(RequiresPositiveOp):
        __constraints__ = (ExpressionConstraint(expr),)

    _test.op("check")(CheckOp)
    with pytest.raises(ConstraintError, match=error):
        verify_constraints(_make_function(CheckOp(input=MyInt().constant(3))))


def _positive_of_argument() -> FunctionOp:
    arg = BlockArgument(name="x", type=MyInt())
    op = RequiresPositiveOp(input=arg, name="pos")
    return FunctionOp(
        name="test_fn",
        body=Block(result=op, args=[arg]),
        result_type=MyInt(),
        type=Function(arguments=pack([MyInt()]), result_type=MyInt()),
    )


def test_verify_undecided_expression_waits() -> None:
    """A clause on a run-time operand passes until it must be decided."""
    verify_constraints(_positive_of_argument())


def test_verify_undecided_expression_fails_when_decided() -> None:
    with pytest.raises(
        ConstraintError,
        match="RequiresPositiveOp %pos: requires input > 0 is undecided: "
        "input is %x, which is not known until run time",
    ):
        verify_constraints(_positive_of_argument(), decided=True)


class _LowerPositive(Pass):
    @lowering_for(RequiresPositiveOp)
    def lower(self, op: RequiresPositiveOp) -> Value | None:
        return MyInt().constant(1)


def test_pass_rejects_lowering_undecided_op() -> None:
    """Replacing an op is its last check, flag or not."""
    token = verify_passes.set(False)
    try:
        with pytest.raises(ConstraintError, match="is undecided"):
            _LowerPositive().run(_positive_of_argument(), Compiler([], IdentityPass()))
    finally:
        verify_passes.reset(token)


def test_pass_rejects_false_op_before_its_handler() -> None:
    token = verify_passes.set(False)
    fn = _make_function(RequiresPositiveOp(input=MyInt().constant(-1)))
    try:
        with pytest.raises(ConstraintError, match="is false"):
            _LowerPositive().run(fn, Compiler([], IdentityPass()))
    finally:
        verify_passes.reset(token)
