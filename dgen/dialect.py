"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .op import Op

_T = TypeVar("_T")


class Dialect:
    _registry: dict[str, Dialect] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: dict[str, type[Op]] = {}
        self.types: dict[str, Callable] = {}
        Dialect._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Dialect:
        return cls._registry[name]

    def op(self, asm_name: str) -> Callable[[_T], _T]:
        def decorator(cls: _T) -> _T:
            setattr(cls, "_asm_name", asm_name)
            setattr(cls, "dialect", self)
            self.ops[asm_name] = cls  # type: ignore[arg-type]
            return cls

        return decorator

    def type(self, name: str) -> Callable[[_T], _T]:
        def decorator(cls: _T) -> _T:
            setattr(cls, "_asm_name", name)
            setattr(cls, "dialect", self)
            self.types[name] = cls  # type: ignore[arg-type]
            return cls

        return decorator
