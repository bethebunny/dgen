from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import dgen

from .type import Memory, Type

if TYPE_CHECKING:
    from .layout import Layout

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

    @property
    def __layout__(self) -> Layout:
        """Transparent layout access for type-kinded params.

        For TypeType constants, returns the concrete type's __layout__.
        This means self.element_type.__layout__ works whether element_type
        is a bare Type or a Constant[TypeType].
        """
        from dgen.dialects.builtin import TypeType

        if isinstance(self.type, TypeType):
            return self.type.concrete.__layout__
        return self.type.__layout__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return NotImplemented
        if self.type != other.type:
            return False
        # For TypeType constants, compare concrete types directly
        # (Memory buffer comparison fails because string pointers differ)
        from dgen.dialects.builtin import TypeType

        if isinstance(self.type, TypeType):
            return self.type.concrete == other.type.concrete
        return self.value == other.value

    def __hash__(self) -> int:
        from dgen.dialects.builtin import TypeType

        if isinstance(self.type, TypeType):
            return hash((type(self), self.type, self.type.concrete))
        return hash((type(self), self.type, self.value))
