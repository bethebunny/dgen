"""Jupyter notebook magic for dgen dialect definitions.

Load the extension in a notebook cell::

    %load_ext dgen.notebook

Then define a dialect with the ``%%dgen-dialect`` cell magic::

    %%dgen-dialect myname
    from builtin import Index, Nil
    op add(lhs: Index, rhs: Index) -> Index

The dialect is compiled from the cell body, registered in ``sys.modules``
under *myname*, and injected into the current IPython namespace so that
subsequent cells can reference it by name.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Protocol

from IPython import get_ipython

from dgen.gen.build import build
from dgen.gen.importer import find_dgen, _path_to_module
from dgen.gen.parser import parse


class _MagicHost(Protocol):
    """Structural type for the IPython shell object passed to ``load_ipython_extension``."""

    user_ns: dict[str, object]

    def register_magic_function(
        self,
        func: Callable[..., None],
        magic_kind: str = "line",
        magic_name: str | None = None,
    ) -> None: ...


def _resolve_notebook_import(module_name: str) -> str:
    """Resolve a dgen import to a Python module path in a notebook context.

    Searches ``sys.path`` for ``.dgen`` files first, then falls back to
    scanning already-loaded modules in ``sys.modules``.
    """
    dgen_file = find_dgen(module_name)
    if dgen_file is not None:
        py_path = _path_to_module(dgen_file)
        if py_path is not None:
            return py_path
    for mod_name in sys.modules:
        if mod_name == module_name or mod_name.endswith(f".{module_name}"):
            return mod_name
    raise ImportError(f"Cannot resolve dgen import: {module_name}")


def _dgen_dialect_magic(line: str, cell: str) -> None:
    """Cell magic: compile a ``.dgen`` cell and inject the dialect into the namespace."""
    dialect_name = line.strip()
    if not dialect_name:
        raise ValueError("%%dgen-dialect requires a dialect name as its argument")
    ast = parse(cell)
    module = ModuleType(dialect_name)
    # Register before build so dataclasses can resolve cls.__module__ via sys.modules.
    sys.modules[dialect_name] = module
    build(
        ast,
        dialect_name=dialect_name,
        resolve_import=_resolve_notebook_import,
        module=module,
    )
    ip = get_ipython()
    if ip is not None:
        ip.user_ns[dialect_name] = module


def load_ipython_extension(ip: _MagicHost) -> None:
    """Register ``%%dgen-dialect`` when loaded via ``%load_ext dgen.notebook``."""
    ip.register_magic_function(
        _dgen_dialect_magic, magic_kind="cell", magic_name="dgen-dialect"
    )
