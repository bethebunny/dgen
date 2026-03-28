"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .op import Op
    from .type import Type

_O = TypeVar("_O", bound="Op")
_T = TypeVar("_T", bound="Type")


class Dialect:
    _registry: dict[str, Dialect] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: dict[str, builtins.type[Op]] = {}
        self.types: dict[str, builtins.type[Type]] = {}
        Dialect._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Dialect:
        return cls._registry[name]

    @classmethod
    def resolve_type_tag(cls, tag: str) -> tuple[Dialect, builtins.type[Type]]:
        """Parse 'dialect.name' tag and return (Dialect, type class).

        Raises ValueError with a clear message on malformed or unknown tags.
        """
        parts = tag.split(".", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Malformed type tag {tag!r}: expected 'dialect.name'")
        dialect_name, type_name = parts
        try:
            dialect = cls._registry[dialect_name]
        except KeyError:
            raise ValueError(
                f"Unknown dialect {dialect_name!r} in type tag {tag!r}"
            ) from None
        try:
            type_cls = dialect.types[type_name]
        except KeyError:
            raise ValueError(
                f"Unknown type {type_name!r} in dialect {dialect_name!r} (tag {tag!r})"
            ) from None
        return dialect, type_cls

    def op(self, asm_name: str) -> Callable[[builtins.type[_O]], builtins.type[_O]]:
        def decorator(cls: builtins.type[_O]) -> builtins.type[_O]:
            cls.asm_name = asm_name
            cls.dialect = self
            self.ops[asm_name] = cls
            return cls

        return decorator

    def type(self, name: str) -> Callable[[builtins.type[_T]], builtins.type[_T]]:
        def decorator(cls: builtins.type[_T]) -> builtins.type[_T]:
            cls.asm_name = name
            cls.dialect = self
            self.types[name] = cls
            return cls

        return decorator

    def trait(self, name: str) -> Callable[[builtins.type[_T]], builtins.type[_T]]:
        def decorator(cls: builtins.type[_T]) -> builtins.type[_T]:
            cls.asm_name = name
            cls.dialect = self
            self.types[name] = cls
            return cls

        return decorator
