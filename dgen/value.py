from __future__ import annotations

from dataclasses import dataclass

import dgen

from .type import Type


@dataclass(eq=False, kw_only=True)
class Value:
    """Base class for SSA values. An op or block argument."""

    name: str | None = None
    type: Type

    @property
    def operands(self) -> list[Value]:
        return []

    @property
    def blocks(self) -> dict[str, dgen.Block]:
        return {}


class Comptime(Value):
    """Type hint: a Value that must be resolvable at compile time.

    Used as a field annotation on Op dataclasses to mark operands whose
    concrete value is needed for type computation (e.g. tile count).
    At runtime the field still holds a regular Value — Comptime is
    purely a type-level marker inspected by the staging evaluator.
    """

    pass
