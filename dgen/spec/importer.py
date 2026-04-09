"""Import hook for .dgen dialect files.

Install with ``install()`` (called automatically when ``dgen`` is imported) to
make ``.dgen`` files importable as regular Python modules.  The hook searches
for ``<name>.dgen`` alongside the normal Python path entries and, when found,
builds the dialect module in-memory via :func:`dgen.spec.builder.build`.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

from dgen.spec.ast import DgenFile
from dgen.spec.builder import build
from dgen.spec.parser import parse


def _path_to_module(dgen_path: Path) -> str | None:
    """Convert an absolute .dgen file path to a dotted Python module name.

    Iterates ``sys.path`` entries to find one that is a parent of *dgen_path*,
    then constructs the module name from the relative path.  Returns ``None``
    if no ``sys.path`` entry covers the file.
    """
    resolved = dgen_path.resolve()
    for entry in sys.path:
        try:
            rel = resolved.relative_to(Path(entry).resolve())
        except ValueError:
            continue
        parts = rel.with_suffix("").parts
        return ".".join(parts)
    return None


def _find_dgen(module_name: str, extra_dirs: list[Path] | None = None) -> Path | None:
    """Find a ``.dgen`` file on ``sys.path`` for a (possibly dotted) module name.

    Converts dots to directory separators: ``foo.bar`` → ``foo/bar.dgen``.
    *extra_dirs* are searched before ``sys.path`` (e.g. the directory containing
    the importing ``.dgen`` file, so sibling imports work).
    Returns the first match found, or ``None``.
    """
    relative = Path(*module_name.split(".")).with_suffix(".dgen")
    search: list[Path] = list(extra_dirs or [])
    search.extend(Path(entry) if entry else Path.cwd() for entry in sys.path)
    for base in search:
        candidate = base / relative
        if candidate.exists():
            return candidate
    return None


def _resolve_import(module_name: str, dgen_dir: Path) -> str:
    """Resolve a single import to a Python module path.

    Searches the *dgen_dir* (sibling files) and ``sys.path`` for a ``.dgen``
    file.  If found, derives the dotted Python module name.  Otherwise, checks
    whether the module is already loadable via Python's import system (which
    includes the DgenFinder hook for ``.dgen`` files in packages).
    """
    dgen_file = _find_dgen(module_name, extra_dirs=[dgen_dir])
    if dgen_file is not None:
        py_path = _path_to_module(dgen_file)
        if py_path is not None:
            return py_path
    # The module may be importable via Python packages (e.g. "ndbuffer" is
    # dgen.dialects.ndbuffer).  Check sys.modules first, then try find_spec.
    for mod_name in sys.modules:
        if mod_name == module_name or mod_name.endswith(f".{module_name}"):
            return mod_name
    raise ImportError(f"Cannot resolve dgen import: {module_name}")


class DgenLoader(importlib.abc.Loader):
    """Loader that compiles a .dgen file into a live Python module."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Populated by exec_module; available for introspection afterward.
        self.ast: DgenFile | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self.path.read_text()
        self.ast = parse(source)
        dgen_dir = self.path.parent
        build(
            self.ast,
            dialect_name=self.path.stem,
            resolve_import=lambda name: _resolve_import(name, dgen_dir),
            module=module,
        )


class DgenFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that locates ``.dgen`` files on sys.path."""

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if path is not None:
            # Sub-package import: path is the parent's __path__, use last component
            name = fullname.rsplit(".", 1)[-1]
            search_dirs = list(path)
        else:
            # Top-level import: convert dots to directory separators
            name = str(Path(*fullname.split(".")))
            search_dirs = sys.path
        for search_dir in search_dirs:
            dgen_file = (
                Path(search_dir) if search_dir else Path.cwd()
            ) / f"{name}.dgen"
            if dgen_file.exists():
                return importlib.machinery.ModuleSpec(
                    fullname,
                    DgenLoader(dgen_file),
                    origin=str(dgen_file),
                )
        return None


_finder: DgenFinder | None = None


def install() -> None:
    """Install the .dgen import hook into ``sys.meta_path`` (idempotent)."""
    global _finder
    if _finder is None:
        _finder = DgenFinder()
        sys.meta_path.insert(0, _finder)
