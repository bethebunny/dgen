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
        self.children: dict[str, Dialect] = {}
        Dialect._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Dialect:
        return cls._registry[name]

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

    def trait(self, name: str) -> Callable[[builtins.type], builtins.type]:
        """Register a trait. Traits are stored in .types (they are types)."""
        return self.type(name)
