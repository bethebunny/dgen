"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Union, get_args, get_origin, get_type_hints


class Dialect:
    _registry: dict[str, Dialect] = {}

    def __init__(self, name: str):
        self.name = name
        self.op_table: dict[str, type] = {}
        self.keyword_table: dict[str, type] = {}
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

            cls.asm = _asm

            # Categorize into tables
            hints = get_type_hints(cls, include_extras=True)
            has_result = "result" in hints
            if not has_result:
                self.keyword_table[asm_name] = cls
            else:
                result_hint = hints["result"]
                if _is_optional(result_hint) is not None:
                    self.op_table[asm_name] = cls
                    self.keyword_table[asm_name] = cls
                else:
                    self.op_table[asm_name] = cls
            return cls

        return decorator

    def type(self, name: str):
        def decorator(fn):
            self.type_table[name] = fn
            return fn

        return decorator


def _is_optional(hint):
    origin = get_origin(hint)
    if origin is Union:
        args = get_args(hint)
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return None
