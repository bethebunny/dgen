"""IR-level constraints attached to op classes for verification.

These are the runtime forms of constraints declared in ``.dgen`` source.
The spec layer parses ``requires X has trait Foo<bar>`` into an AST
:class:`HasTraitConstraint`, and the builder compiles that into one of
the classes here. By the time IR verification runs, only the IR-level
forms remain — verification doesn't reach back into the AST.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import dgen


@dataclass(frozen=True)
class TraitConstraint:
    """An op-level ``requires <subject> has trait <trait>`` constraint.

    ``subject`` names an operand or compile-time parameter on the op.
    ``build_target`` produces the trait instance to test against — it's a
    closure compiled by the builder that captures the constraint's type
    references and substitutes the op's own parameter values at
    verification time. This lets ``Handler<Raise<error_type>>`` resolve
    differently for each op instance without the verifier touching the
    spec AST.
    """

    subject: str
    build_target: Callable[["dgen.Op"], "dgen.Type"]


def has_trait(subject: str, trait: "dgen.Type") -> TraitConstraint:
    """Convenience factory for hand-written ``__constraints__`` declarations.

    Hand-written op classes — common in tests and in dialects that aren't
    going through ``.dgen`` source — declare static trait constraints by
    naming a fixed trait instance. This wraps that pattern:

        __constraints__ = (has_trait("input", Numeric()),)

    Parametric constraints (where the trait depends on the op's params)
    must build the closure directly: ``TraitConstraint(subject=...,
    build_target=lambda op: SomeTrait(field=op.param))``.
    """
    return TraitConstraint(subject=subject, build_target=lambda _op: trait)
