"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .op import Op
    from .type import Type

_O = TypeVar("_O", bound="Op")
_T = TypeVar("_T", bound="Type")


class Dialect:
    def __init__(self, name: str, module: ModuleType | None = None) -> None:
        self.name = name
        self.ops: dict[str, builtins.type[Op]] = {}
        self.types: dict[str, builtins.type[Type]] = {}
        # Backing Python module (set by the builder); used to look up
        # cross-dialect re-exports such as ``from builtin import Index``,
        # where ``Index`` lives in ``builtin``'s module namespace but not
        # in its own ``types`` dict.
        self.module = module
        # Register in the global dialect table so subsequent imports
        # (Python, .dgen, ASM) find the same instance.  Imported lazily to
        # break the dgen.imports ↔ dgen.dialect cycle.
        from dgen.imports import DIALECTS

        DIALECTS[name] = self

    def __getattr__(self, attr: str) -> object:
        # Allow `import ndbuffer` in a .dgen file to bind the Dialect, then
        # `ndbuffer.Shape` to resolve a contained type/op (or a re-exported
        # name from ``ndbuffer``'s module namespace).
        if attr in self.types:
            return self.types[attr]
        if attr in self.ops:
            return self.ops[attr]
        if self.module is not None and attr in self.module.__dict__:
            return self.module.__dict__[attr]
        raise AttributeError(f"dialect {self.name!r} has no member {attr!r}")

    def qualified_name(self, local_name: str) -> str:
        """Return dialect-qualified name, e.g. ``toy.Tensor``.  Builtin names are unqualified."""
        if self.name == "builtin":
            return local_name
        return f"{self.name}.{local_name}"

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
