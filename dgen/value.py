from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import dgen

from .type import Memory, Type

T = TypeVar("T", bound=Type)


@dataclass(eq=False, kw_only=True)
class Value(Generic[T]):
    """Base class for SSA values. An op or block argument."""

    name: str | None = None
    type: T
    ready: ClassVar[bool] = False

    @property
    def operands(self) -> list[Value]:
        return []

    @property
    def blocks(self) -> dict[str, dgen.Block]:
        return {}

    @property
    def __constant__(self) -> Memory[T]:
        raise NotImplementedError


@dataclass(eq=False, kw_only=True)
class Constant(Value[T]):
    ready: ClassVar[bool] = True
    value: Memory[T]

    @property
    def __constant__(self) -> Memory[T]:
        return self.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return NotImplemented
        return self.type == other.type and self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self), self.type, self.value))
