"""Tests for parametric trait support in .dgen and ``Value.has_trait``.

Exercises the language-level mechanism only — independent of any specific
effect dialect:

- Parser: ``trait Foo<x: Type>`` and ``has trait Foo<Bar<E>>`` round-trip
  through the AST.
- Builder: a type declaring ``has trait Foo<...>`` carries the resolved
  trait instance in a runtime ``__declared_traits__`` slot, with the
  type's own parameters substituted in.
- ``Value.has_trait``: accepts both a trait class (unparameterized,
  isinstance-based) and a trait instance (structural match).
- Verification: ``requires x has trait Foo<Bar<E>>`` resolves with
  per-op-instance substitution of the op's parameters.

Concrete fixture: a tiny ``_paramtrait_test`` dialect with a parametric
trait ``Slot<element: Type>`` and a type ``Box<element>`` declaring
``has trait Slot<element>``. Real dialect uses (e.g. ``error.RaiseHandler``
declaring ``Handler<Raise<...>>``) live with their dialect's tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value, layout
from dgen.builtins import pack
from dgen.dialects import builtin
from dgen.dialects.function import Function, FunctionOp
from dgen.ir.verification import ConstraintError, verify_constraints
from dgen.spec.ast import (
    HasTraitConstraint,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)
from dgen.spec.parser import parse
from dgen.type import Fields


# -- Fixture: a tiny dialect with a parametric trait -------------------------

_test = Dialect("_paramtrait_test")


@_test.trait("Slot")
@dataclass(frozen=True, eq=False)
class Slot(Trait):
    element: Value[TypeType]
    __params__: ClassVar[Fields] = (("element", TypeType),)


@_test.type("Marker")
@dataclass(frozen=True, eq=False)
class Marker(Type):
    """Bare type used as a parameter value below."""

    __layout__ = layout.Void()


@_test.type("Box")
@dataclass(frozen=True, eq=False)
class Box(Type):
    element: Value[TypeType]
    __layout__ = layout.Void()
    __params__: ClassVar[Fields] = (("element", TypeType),)

    @property
    def __declared_traits__(self) -> tuple[Slot, ...]:
        # Per-instance substitution: each Box<E> claims Slot<E>.
        return (Slot(element=self.element),)


# -- Parser ------------------------------------------------------------------


def test_parse_parametric_trait_no_body():
    """``trait Slot<element: Type>`` parses with one param."""
    result = parse("trait Slot<element: Type>\n")
    assert result.traits == [
        TraitDecl(
            name="Slot",
            params=[ParamDecl(name="element", type=TypeRef("Type"))],
        )
    ]


def test_parse_parametric_trait_with_body():
    """Parametric traits with static fields still parse correctly."""
    src = 'trait Indexed<element: Type>:\n    static label: String = "indexed"\n'
    result = parse(src)
    td = result.traits[0]
    assert td.name == "Indexed"
    assert td.params == [ParamDecl(name="element", type=TypeRef("Type"))]
    assert len(td.statics) == 1


def test_parse_has_parametric_trait_on_type():
    """``has trait Slot<Box<E>>`` preserves the full TypeRef tree."""
    src = (
        "type Box<element: Type>:\n    layout Void\n    has trait Slot<Box<element>>\n"
    )
    result = parse(src)
    td = result.types[0]
    assert isinstance(td, TypeDecl)
    assert td.traits == [
        TypeRef(name="Slot", args=[TypeRef(name="Box", args=[TypeRef(name="element")])])
    ]


def test_parse_requires_has_parametric_trait():
    """``requires x has trait Slot<E>`` parses as HasTraitConstraint with TypeRef."""
    src = (
        "op put<element: Type>(box) -> Nil:\n    requires box has trait Slot<element>\n"
    )
    op = parse(src).ops[0]
    assert op.constraints == [
        HasTraitConstraint(
            lhs="box",
            trait=TypeRef("Slot", args=[TypeRef("element")]),
        )
    ]


# -- Builder / runtime -------------------------------------------------------


def test_declared_traits_substitutes_per_instance():
    """``has_trait`` against a parametric instance reads the type's per-instance
    binding — Box<Nil> is a Slot<Nil>, not a Slot<Byte>."""
    nil_box = Box(element=builtin.Nil())
    byte_box = Box(element=builtin.Byte())
    nil_slot = Slot(element=builtin.Nil())
    byte_slot = Slot(element=builtin.Byte())
    assert nil_box.has_trait(nil_slot)
    assert not nil_box.has_trait(byte_slot)
    assert byte_box.has_trait(byte_slot)
    assert not byte_box.has_trait(nil_slot)


def test_has_trait_class_matches_any_binding():
    """``has_trait(Slot)`` (the bare class) returns True regardless of params."""
    assert Box(element=builtin.Nil()).has_trait(Slot)
    assert Box(element=builtin.Byte()).has_trait(Slot)


def test_unparameterized_trait_check_unchanged():
    """Unparameterized traits still match via isinstance on the class."""

    @_test.trait("Tag")
    class Tag(Trait):
        pass

    @_test.type("Tagged")
    @dataclass(frozen=True, eq=False)
    class Tagged(Tag, Type):
        __layout__ = layout.Void()

    assert Tagged().has_trait(Tag)
    assert not Marker().has_trait(Tag)


def test_builtin_handler_is_parametric_trait():
    """The built-in ``Handler<effect_type>`` round-trips through the builder."""
    assert builtin.Handler is not None
    h = builtin.Handler(effect_type=builtin.Nil())
    assert isinstance(h.effect_type, builtin.Nil)


# -- Verification ------------------------------------------------------------


@_test.op("put")
@dataclass(eq=False, kw_only=True)
class PutOp(Op):
    """Constraint requires box's type to be Slot<element>."""

    element: Value[TypeType]
    box: Value
    type: Type = Marker()
    __operands__: ClassVar[Fields] = (("box", Type),)
    __params__: ClassVar[Fields] = (("element", TypeType),)
    __constraints__ = (
        HasTraitConstraint(
            lhs="box",
            trait=TypeRef("Slot", args=[TypeRef("element")]),
        ),
    )


def _wrap(*body_ops: Op) -> FunctionOp:
    result = body_ops[-1] if body_ops else Marker().constant(None)
    return FunctionOp(
        name="f",
        body=Block(result=result, args=[]),
        result_type=Marker(),
        type=Function(arguments=pack(), result_type=Marker()),
    )


def test_verify_parametric_trait_satisfied():
    """A Box<Nil> satisfies ``has trait Slot<Nil>`` for the put op."""
    nil_box = Box(element=builtin.Nil()).constant(None)
    op = PutOp(element=builtin.Nil(), box=nil_box)
    verify_constraints(_wrap(op))  # no raise


def test_verify_parametric_trait_violated():
    """A Box<Byte> does NOT satisfy ``has trait Slot<Nil>`` for the put op."""
    byte_box = Box(element=builtin.Byte()).constant(None)
    op = PutOp(element=builtin.Nil(), box=byte_box)
    with pytest.raises(ConstraintError, match="does not implement trait"):
        verify_constraints(_wrap(op))
