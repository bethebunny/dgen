"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import get_type_hints


class Dialect:
    _registry: dict[str, Dialect] = {}

    def __init__(self, name: str):
        self.name = name
        self.op_table: dict[str, type] = {}
        self.type_table: dict[str, Callable] = {}
        Dialect._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Dialect:
        return cls._registry[name]

    def op(self, asm_name: str):
        def decorator(cls):
            from toy_python.asm.formatting import op_asm

            cls = dataclasses.dataclass(cls)
            cls._asm_name = asm_name
            cls._dialect_name = self.name

            @property
            def _asm(self) -> Iterable[str]:
                return op_asm(self)

            if "asm" not in cls.__dict__:
                cls.asm = _asm
            self.op_table[asm_name] = cls
            return cls

        return decorator

    def type(self, name: str):
        def decorator(fn):
            self.type_table[name] = fn
            return fn

        return decorator
