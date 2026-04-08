"""Dialect class with decorator-based op/type registration."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .op import Op
    from .type import Type

_O = TypeVar("_O", bound="Op")
_T = TypeVar("_T", bound="Type")


class Dialect:
    _registry: dict[str, Dialect] = {}
    paths: list[Path] = []
    """Search paths for ``.dgen`` files, analogous to ``sys.path``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: dict[str, builtins.type[Op]] = {}
        self.types: dict[str, builtins.type[Type]] = {}
        Dialect._registry[name] = self

    def qualified_name(self, local_name: str) -> str:
        """Return dialect-qualified name, e.g. ``toy.Tensor``.  Builtin names are unqualified."""
        if self.name == "builtin":
            return local_name
        return f"{self.name}.{local_name}"

    @classmethod
    def get(cls, name: str) -> Dialect:
        """Look up a dialect by name.

        1. Return the dialect if already registered (like ``sys.modules``).
        2. Search ``Dialect.paths`` for ``{name}.dgen`` and import it.
        3. Raise ``KeyError`` if not found.
        """
        if name in cls._registry:
            return cls._registry[name]
        for directory in cls.paths:
            candidate = directory / f"{name}.dgen"
            if candidate.exists():
                import importlib

                # The DgenFinder hook handles .dgen → module compilation.
                # We need to figure out the Python module name for this path.
                from dgen.spec.importer import _path_to_module

                py_mod = _path_to_module(candidate)
                if py_mod is not None:
                    importlib.import_module(py_mod)
                    if name in cls._registry:
                        return cls._registry[name]
        raise KeyError(name)

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
