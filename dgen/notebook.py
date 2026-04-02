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

from dgen.gen.ast import DgenFile
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


def _make_import_map(ast: DgenFile) -> dict[str, str]:
    """Build an import map for dialect imports in a notebook cell.

    Searches ``sys.path`` for ``.dgen`` files first, then falls back to
    scanning already-loaded modules in ``sys.modules``.
    """
    loaded_modules = list(sys.modules)
    result: dict[str, str] = {}
    for decl in ast.imports:
        dgen_file = find_dgen(decl.module)
        if dgen_file is not None:
            py_path = _path_to_module(dgen_file)
            if py_path is not None:
                result[decl.module] = py_path
                continue
        for mod_name in loaded_modules:
            if mod_name == decl.module or mod_name.endswith(f".{decl.module}"):
                result[decl.module] = mod_name
                break
    return result


def _dgen_dialect_magic(line: str, cell: str) -> None:
    """Cell magic: compile a ``.dgen`` cell and inject the dialect into the namespace."""
    dialect_name = line.strip()
    if not dialect_name:
        raise ValueError("%%dgen-dialect requires a dialect name as its argument")
    ast = parse(cell)
    import_map = _make_import_map(ast)
    module = ModuleType(dialect_name)
    # Register before build so dataclasses can resolve cls.__module__ via sys.modules.
    sys.modules[dialect_name] = module
    build(ast, dialect_name=dialect_name, import_map=import_map, module=module)
    ip = get_ipython()
    if ip is not None:
        ip.user_ns[dialect_name] = module


def load_ipython_extension(ip: _MagicHost) -> None:
    """Register ``%%dgen-dialect`` when loaded via ``%load_ext dgen.notebook``."""
    ip.register_magic_function(
        _dgen_dialect_magic, magic_kind="cell", magic_name="dgen-dialect"
    )
