"""Tests for parametric trait support in .dgen and ``Value.has_trait``.

Exercises the language-level mechanism only — independent of any specific
effect dialect:

- Parser: ``trait Foo<x: Type>`` and ``has trait Foo<Bar<E>>`` round-trip
  through the AST.
- Builder: a type declaring ``has trait Foo<...>`` gets a
  ``_declared_traits`` method that constructs the trait instance with the
  declaring type's params substituted in. No ``_DeferredTrait`` wrapper —
  the method body is a plain lambda.
- ``Value.has_trait``: accepts a trait **instance** (not a class) and
  compares structurally via canonical ``to_json()`` equality. Users that
  previously relied on ``isinstance(v, SomeTrait)`` must migrate to
  ``v.has_trait(SomeTrait())``.
- Verification: ``requires x has trait Foo<Bar<E>>`` resolves the
  constraint's TypeRef with the op's own parameter values substituted in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import pytest

import dgen.imports
from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value, layout
from dgen.builtins import pack
from dgen.dialects import builtin
from dgen.dialects.function import Function, FunctionOp
from dgen.ir.constraints import TraitConstraint
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
    __layout__ = layout.Void()
    __params__: ClassVar[Fields] = (("element", TypeType),)


@_test.type("Marker")
@dataclass(frozen=True, eq=False)
class Marker(Type):
    """Plain type used as a parameter value below."""

    __layout__ = layout.Void()


@_test.type("Box")
@dataclass(frozen=True, eq=False)
class Box(Type):
    element: Value[TypeType]
    __layout__ = layout.Void()
    __params__: ClassVar[Fields] = (("element", TypeType),)

    @property
    def traits(self) -> Iterator[Type]:
        # Per-instance substitution — the property body reads ``self.element``.
        yield Slot(element=self.element)
        yield from super().traits


# -- Parser ------------------------------------------------------------------


def test_parse_parametric_trait_no_body():
    result = parse("trait Slot<element: Type>\n")
    assert result.traits == [
        TraitDecl(
            name="Slot",
            params=[ParamDecl(name="element", type=TypeRef("Type"))],
        )
    ]


def test_parse_parametric_trait_with_body():
    src = 'trait Indexed<element: Type>:\n    static label: String = "indexed"\n'
    result = parse(src)
    td = result.traits[0]
    assert td.name == "Indexed"
    assert td.params == [ParamDecl(name="element", type=TypeRef("Type"))]
    assert len(td.statics) == 1


def test_parse_has_parametric_trait_on_type():
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
    """``has_trait`` picks the binding from the value's own params —
    Box<Nil> is a Slot<Nil>, not a Slot<Byte>."""
    nil_box = Box(element=builtin.Nil())
    byte_box = Box(element=builtin.Byte())
    nil_slot = Slot(element=builtin.Nil())
    byte_slot = Slot(element=builtin.Byte())
    assert nil_box.has_trait(nil_slot)
    assert not nil_box.has_trait(byte_slot)
    assert byte_box.has_trait(byte_slot)
    assert not byte_box.has_trait(nil_slot)


def test_has_trait_uses_structural_tojson_equality():
    """has_trait matches two distinct Python instances that serialize
    identically — it's comparing structure, not identity."""
    assert Box(element=builtin.Nil()).has_trait(Slot(element=builtin.Nil()))
    # Different instance of the same-shape trait → still matches.
    repeat = Slot(element=builtin.Nil())
    assert Box(element=builtin.Nil()).has_trait(repeat)


def test_builtin_handler_is_parametric_trait():
    """``Handler<effect_type>`` is declared in builtin.dgen and instantiable
    with a concrete Type argument."""
    assert builtin.Handler is not None
    h = builtin.Handler(effect_type=builtin.Nil())
    assert isinstance(h.effect_type, builtin.Nil)


def test_builtin_effect_exists():
    """``Effect`` is an unparameterized trait declared in builtin.dgen."""
    assert builtin.Effect is not None
    assert issubclass(builtin.Effect, Trait)


# -- Verification ------------------------------------------------------------


@_test.op("put")
@dataclass(eq=False, kw_only=True)
class PutOp(Op):
    """Parametric trait constraint: ``box`` must have ``Slot<element>``,
    where ``element`` is the op's own compile-time parameter."""

    element: Value[TypeType]
    box: Value
    type: Type = Marker()
    __operands__: ClassVar[Fields] = (("box", Type),)
    __params__: ClassVar[Fields] = (("element", TypeType),)
    __constraints__ = (
        TraitConstraint(
            subject="box",
            build_target=lambda op: Slot(element=op.element),
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
    nil_box = Box(element=builtin.Nil()).constant(None)
    op = PutOp(element=builtin.Nil(), box=nil_box)
    verify_constraints(_wrap(op))  # no raise


def test_verify_parametric_trait_violated():
    byte_box = Box(element=builtin.Byte()).constant(None)
    op = PutOp(element=builtin.Nil(), box=byte_box)
    with pytest.raises(ConstraintError, match="does not implement trait .*Slot"):
        verify_constraints(_wrap(op))


# -- MRO inheritance ---------------------------------------------------------


def test_subclass_inherits_parent_traits_via_super():
    """A subclass's ``traits`` chains through ``super().traits`` and so picks
    up the parent's declared traits — without re-declaring them."""

    @_test.type("Box64")
    @dataclass(frozen=True, eq=False)
    class Box64(Box):
        # No traits declared here; should inherit Box's Slot<element>.
        pass

    b = Box64(element=builtin.Nil())
    assert b.has_trait(Slot(element=builtin.Nil()))


def test_subclass_can_add_traits_on_top_of_parent():
    """A subclass that yields its own traits *and* ``super().traits`` keeps
    both — no shadowing, no merge code in the user."""

    @_test.trait("Linear")
    @dataclass(frozen=True, eq=False)
    class Linear(Trait):
        __layout__ = layout.Void()

    @_test.type("LinearBox")
    @dataclass(frozen=True, eq=False)
    class LinearBox(Box):
        @property
        def traits(self) -> Iterator[Type]:
            yield Linear()
            yield from super().traits

    b = LinearBox(element=builtin.Nil())
    assert b.has_trait(Linear())
    assert b.has_trait(Slot(element=builtin.Nil()))


# -- Deep parametric nesting -------------------------------------------------


def test_deeply_nested_parametric_constraint():
    """Three-level nesting in a constraint: ``A<B<C<param>>>``.

    Exercises the recursive ``_trait_fn`` in the builder via .dgen, which
    only had single-level coverage before.
    """

    dialect = dgen.imports.load(
        "_deep_paramtraits_test",
        source=(
            "trait Wrapper<inner: Type>\n"
            "type Triple<a: Type>:\n"
            "    layout Void\n"
            "    has trait Wrapper<Wrapper<Wrapper<a>>>\n"
        ),
    )
    triple = dialect.types["Triple"](a=builtin.Nil())
    wrapper = dialect.types["Wrapper"]
    target = wrapper(inner=wrapper(inner=wrapper(inner=builtin.Nil())))
    assert triple.has_trait(target)
    # Substitutes correctly: a different leaf doesn't match.
    not_target = wrapper(inner=wrapper(inner=wrapper(inner=builtin.Byte())))
    assert not triple.has_trait(not_target)


# -- Op-level traits ---------------------------------------------------------


def test_op_can_declare_its_own_trait_via_dgen():
    """``op my_op: has trait Numeric`` end-to-end through the builder."""

    dialect = dgen.imports.load(
        "_optrait_test",
        source=(
            "from builtin import Nil\n"
            "trait Marker\n"
            "op tagged() -> Nil:\n"
            "    has trait Marker\n"
        ),
    )
    op_cls = dialect.ops["tagged"]
    op = op_cls()
    assert op.has_trait(dialect.types["Marker"]())


# -- Negative path: malformed constraints ----------------------------------


def test_unknown_trait_in_dgen_constraint_fails_at_build_time():
    """``has trait NoSuchTrait`` references an unregistered name. The
    builder's ``_trait_fn`` resolves names eagerly, so the error happens
    when the .dgen file is loaded, not at verify time."""

    with pytest.raises((KeyError, AttributeError, TypeError)):
        dgen.imports.load(
            "_bad_trait_ref_test",
            source=("type Foo:\n    layout Void\n    has trait DoesNotExist\n"),
        )
