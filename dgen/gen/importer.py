"""Import hook for .dgen dialect files.

Install with ``install()`` (called automatically when ``dgen`` is imported) to
make ``.dgen`` files importable as regular Python modules.  The hook searches
for ``<name>.dgen`` alongside the normal Python path entries and, when found,
builds the dialect module in-memory via :func:`dgen.gen.build.build`.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

from dgen.gen.ast import DgenFile
from dgen.gen.build import build
from dgen.gen.parser import parse

# Hardcoded fallback: if relative discovery fails, try this mapping.
_DEFAULT_MAP: dict[str, str] = {
    "builtin": "dgen.dialects.builtin",
}


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


def _resolve_imports(dgen_path: Path, ast: DgenFile) -> dict[str, str]:
    """Build an import map for the generator from a parsed .dgen file.

    For each import declaration in *ast*, look for a sibling ``<module>.dgen``
    file in the same directory.  If found, derive the Python module path from
    ``sys.path``.  Fall back to :data:`_DEFAULT_MAP` for unresolved names.
    """
    result: dict[str, str] = {}
    dgen_dir = dgen_path.parent
    for decl in ast.imports:
        candidate = dgen_dir / f"{decl.module}.dgen"
        if candidate.exists():
            py_path = _path_to_module(candidate)
            if py_path is not None:
                result[decl.module] = py_path
                continue
        fallback = _DEFAULT_MAP.get(decl.module)
        if fallback is not None:
            result[decl.module] = fallback
    return result


class DgenLoader(importlib.abc.Loader):
    """Loader that compiles a .dgen file into a live Python module."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Populated by exec_module; available for introspection afterward.
        self.ast: DgenFile | None = None
        self.import_map: dict[str, str] = {}

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self.path.read_text()
        self.ast = parse(source)
        self.import_map = _resolve_imports(self.path, self.ast)
        build(
            self.ast,
            dialect_name=self.path.stem,
            import_map=self.import_map,
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
        name = fullname.rsplit(".", 1)[-1]
        search_dirs: list[str] = list(path) if path is not None else sys.path
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
