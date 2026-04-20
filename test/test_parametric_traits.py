"""Tests for parametric trait support in .dgen and has_trait.

Covers:
- Parser: ``trait Foo<x: Type>`` and ``has trait Handler<Raise<E>>`` round-trip
  through the AST.
- Builder: types declaring ``has trait Handler<...>`` carry the resolved
  trait instance in ``__declared_traits__``.
- ``Value.has_trait``: supports both trait classes (unparameterized) and
  trait instances (parameterized).
- Verification: ``requires x has trait Handler<...>`` is enforced with
  parametric substitution of the op's own parameters.
"""

from __future__ import annotations

from dgen.block import Block, BlockParameter
from dgen.builtins import pack
from dgen.dialects import builtin, error
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.ir.verification import verify_constraints
from dgen.spec.ast import (
    HasTraitConstraint,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)
from dgen.spec.parser import parse


# -- Parser ------------------------------------------------------------------


def test_parse_parametric_trait_no_body():
    """``trait Handler<effect_type: Effect>`` parses with one param."""
    result = parse("trait Handler<effect_type: Effect>\n")
    assert result.traits == [
        TraitDecl(
            name="Handler",
            params=[ParamDecl(name="effect_type", type=TypeRef("Effect"))],
        )
    ]


def test_parse_parametric_trait_with_body():
    """Parametric traits with static fields still parse correctly."""
    src = (
        "trait Supervise<failure_type: Type>:\n"
        '    static label: String = "supervisor"\n'
    )
    result = parse(src)
    assert result.traits[0].name == "Supervise"
    assert result.traits[0].params == [
        ParamDecl(name="failure_type", type=TypeRef("Type"))
    ]
    assert len(result.traits[0].statics) == 1


def test_parse_has_parametric_trait_on_type():
    """``has trait Handler<Raise<E>>`` preserves the full TypeRef tree."""
    src = (
        "type RaiseHandler<error_type: Type>:\n"
        "    layout Void\n"
        "    has trait Handler<Raise<error_type>>\n"
    )
    result = parse(src)
    td = result.types[0]
    assert isinstance(td, TypeDecl)
    assert td.traits == [
        TypeRef(
            name="Handler",
            args=[TypeRef(name="Raise", args=[TypeRef(name="error_type")])],
        )
    ]


def test_parse_requires_has_parametric_trait():
    """``requires x has trait Handler<Raise<E>>`` parses as HasTraitConstraint."""
    src = (
        "op raise<handler: Type>(error) -> Nil:\n"
        "    requires handler has trait Handler<Raise<E>>\n"
    )
    op = parse(src).ops[0]
    assert op.constraints == [
        HasTraitConstraint(
            lhs="handler",
            trait=TypeRef("Handler", args=[TypeRef("Raise", args=[TypeRef("E")])]),
        )
    ]


# -- Builder -----------------------------------------------------------------


def test_builtin_effect_and_handler_exist():
    """Parametric traits round-trip through the runtime builder."""
    assert builtin.Effect is not None
    assert builtin.Handler is not None
    # Handler is instantiable with a compile-time effect_type value.
    h = builtin.Handler(effect_type=builtin.Nil())
    assert isinstance(h.effect_type, builtin.Nil)


def test_raise_has_effect_trait_via_inheritance():
    """``type Raise<E>: has trait Effect`` inherits Effect (unparameterized)."""
    r = error.Raise(error_type=builtin.Nil())
    assert r.has_trait(builtin.Effect)
    # Raise is literally a subclass of Effect.
    assert isinstance(r, builtin.Effect)


def test_raise_handler_has_parametric_handler_trait():
    """``RaiseHandler<E>`` declares ``Handler<Raise<E>>`` with E substituted."""
    h = error.RaiseHandler(error_type=builtin.Nil())
    target = builtin.Handler(effect_type=error.Raise(error_type=builtin.Nil()))
    assert h.has_trait(target)


def test_raise_handler_mismatched_error_type():
    """Same trait class, different effect parameter → not equivalent."""
    h = error.RaiseHandler(error_type=builtin.Nil())
    wrong = builtin.Handler(effect_type=error.Raise(error_type=builtin.Byte()))
    assert not h.has_trait(wrong)


def test_raise_handler_has_handler_class():
    """``has_trait(Handler)`` with the bare class returns True for any bound."""
    h = error.RaiseHandler(error_type=builtin.Nil())
    assert h.has_trait(builtin.Handler)


def test_unparameterized_trait_check_unchanged():
    """Unparameterized traits still match via isinstance on the class."""
    r = error.Raise(error_type=builtin.Nil())
    assert r.has_trait(builtin.Effect)
    # A type that doesn't implement Effect returns False.
    assert not builtin.Nil().has_trait(builtin.Effect)


def test_declared_traits_substitutes_per_instance():
    """``__declared_traits__`` reads the instance's params each time."""
    h1 = error.RaiseHandler(error_type=builtin.Nil())
    h2 = error.RaiseHandler(error_type=builtin.Byte())
    # Each instance builds its own Handler<Raise<T>> with T substituted.
    r1 = h1.__declared_traits__[0]
    r2 = h2.__declared_traits__[0]
    assert isinstance(r1, builtin.Handler) and isinstance(r2, builtin.Handler)
    assert isinstance(r1.effect_type, error.Raise)
    assert isinstance(r1.effect_type.error_type, builtin.Nil)
    assert isinstance(r2.effect_type.error_type, builtin.Byte)


# -- Verification ------------------------------------------------------------


def test_verify_parametric_trait_satisfied():
    """raise(handler, err) verifies when handler's effect type matches err.

    (RaiseOp has no explicit ``requires`` today; the error_type match is
    enforced structurally via the shared op parameter. This test documents
    that verification nonetheless completes cleanly on a well-formed raise.)
    """
    handler = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=handler,
        error=Index().constant(0),
    )
    fn = FunctionOp(
        name="f",
        body=Block(result=raise_op, parameters=[handler]),
        result_type=Index(),
        type=Function(arguments=pack(), result_type=Index()),
    )
    verify_constraints(fn)  # should not raise
