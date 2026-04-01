"""Tests for trait infrastructure: Trait base class, has_trait(), verify_constraints."""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Dialect, Op, Trait, Type, Value, has_trait, layout
from dgen.gen.ast import TraitConstraint
from dgen.gen.parser import parse
from dgen.trait import Trait as TraitBase


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
    __operands__ = (("lhs", Type), ("rhs", Type))


# -- Trait base class --------------------------------------------------------


def test_trait_inherits_from_base():
    assert issubclass(Numeric, TraitBase)
    assert issubclass(Ordered, TraitBase)


def test_trait_registered_in_dialect_traits():
    assert "Numeric" in _test.traits
    assert _test.traits["Numeric"] is Numeric
    assert "Ordered" in _test.traits


def test_trait_not_in_dialect_types():
    assert "Numeric" not in _test.types
    assert "Ordered" not in _test.types


# -- has_trait() on types ----------------------------------------------------


def test_has_trait_type_positive():
    assert has_trait(MyInt(), Numeric)
    assert has_trait(MyInt(), Ordered)


def test_has_trait_type_negative():
    assert not has_trait(MyStr(), Numeric)
    assert not has_trait(MyStr(), Ordered)


# -- has_trait() on ops ------------------------------------------------------


def test_has_trait_op_positive():
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    assert has_trait(op, Numeric)


def test_has_trait_op_negative():
    op = AddNumsOp(lhs=MyInt().constant(1), rhs=MyInt().constant(2))
    assert not has_trait(op, Ordered)


# -- .dgen built traits inherit from Trait -----------------------------------


def test_dgen_built_trait_inherits():
    """Traits built from .dgen files inherit from Trait."""
    from dgen.dialects import algebra

    assert issubclass(algebra.AddMagma, TraitBase)
    assert issubclass(algebra.MulMagma, TraitBase)
    assert issubclass(algebra.TotalOrder, TraitBase)


def test_dgen_built_trait_in_dialect_traits():
    """Traits built from .dgen are in dialect.traits, not dialect.types."""
    from dgen.dialects import algebra

    assert "AddMagma" in algebra.algebra.traits
    assert "AddMagma" not in algebra.algebra.types


# -- Parser: has trait / has type constraints --------------------------------


def test_parse_has_trait_constraint():
    src = "op foo(x) -> Type:\n    requires x has trait Numeric\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert isinstance(c, TraitConstraint)
    assert c.lhs == "x"
    assert c.trait == "Numeric"


def test_parse_type_body_requires():
    src = "type Foo<t: Type>:\n    requires t has trait Numeric\n"
    td = parse(src).types[0]
    assert len(td.constraints) == 1
    c = td.constraints[0]
    assert isinstance(c, TraitConstraint)
    assert c.lhs == "t"
    assert c.trait == "Numeric"
