"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

from collections.abc import Callable


class Dialect:
    _registry: dict[str, Dialect] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: dict[str, type] = {}
        self.types: dict[str, Callable] = {}
        Dialect._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Dialect:
        return cls._registry[name]

    def op(self, asm_name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            cls._asm_name = asm_name
            cls.dialect = self
            self.ops[asm_name] = cls
            return cls

        return decorator

    def type(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            cls._asm_name = name
            cls.dialect = self
            self.types[name] = cls
            return cls

        return decorator
