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
