"""Dialect import system.

Two pieces of module-level state, analogous to the Python import system:

* :data:`PATH` — directories searched for ``.dgen`` files (like ``sys.path``).
* :data:`DIALECTS` — qualname → loaded :class:`~dgen.Dialect` (like ``sys.modules``).

:func:`load` is the single entry point for resolving a dialect.  Imports inside
``.dgen`` files and ASM text both go through it, as does the Python import
hook installed by :func:`install_hook` (which fires whenever Python's normal
resolution lands on a ``.dgen`` file).

Resolution rules:

* ``load(qualname)`` checks :data:`DIALECTS`, then walks :data:`PATH`.
* ``load(qualname, _from=dir)`` is the relative form: checks :data:`DIALECTS`
  then ``<dir>/<qualname>.dgen``.  Never consults :data:`PATH`.
* ``load(qualname, source=text)`` parses ``text`` directly — no file lookup.
  Useful for tests and notebook development.

Whichever path runs, the resulting :class:`~dgen.Dialect` is stored in
:data:`DIALECTS` keyed by ``qualname``.  Like ``sys.modules``, an existing
entry is authoritative — repeat loads return the cached Dialect rather than
rebuild.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from dgen.spec.builder import build
from dgen.spec.parser import parse

if TYPE_CHECKING:
    from dgen.dialect import Dialect

PATH: list[Path] = [Path(__file__).parent / "dialects"]
"""Directories searched for ``.dgen`` files (like ``sys.path`` for dgen)."""

DIALECTS: dict[str, "Dialect"] = {}
"""Loaded dialects keyed by qualname (like ``sys.modules`` for dgen)."""


def load(
    qualname: str,
    *,
    source: str | None = None,
    _from: Path | None = None,
) -> "Dialect":
    """Resolve a (possibly dotted) qualname to a :class:`~dgen.Dialect`.

    See module docstring for resolution rules.
    """
    if source is not None:
        return _build(qualname, source, dgen_dir=Path.cwd())
    if qualname in DIALECTS:
        return DIALECTS[qualname]
    rel = Path(*qualname.split(".")).with_suffix(".dgen")
    bases = [_from] if _from is not None else PATH
    for base in bases:
        candidate = base / rel
        if candidate.exists():
            return _load_file(qualname, candidate)
    where = f"in {_from}" if _from is not None else "on dgen.PATH"
    raise ImportError(f"Cannot find dgen dialect '{qualname}' {where}")


def _load_file(qualname: str, path: Path) -> "Dialect":
    """Load a ``.dgen`` file.  Routes through Python's import system when the
    file sits under a ``sys.path`` entry — that way the ``.dgen`` import hook
    fires, ``sys.modules`` is populated under the natural dotted name, and
    re-imports converge on the same module.  Falls back to a build into a
    plain dict when the file isn't reachable via ``sys.path``.
    """
    py_mod = path_to_python_module(path)
    if py_mod is not None:
        importlib.import_module(py_mod)
        if qualname in DIALECTS:
            return DIALECTS[qualname]
    return _build(qualname, path.read_text(), dgen_dir=path.parent)


def path_to_python_module(path: Path) -> str | None:
    """Return the dotted Python module name covering ``path`` via ``sys.path``,
    or ``None`` if no entry covers it.
    """
    resolved = path.resolve()
    for entry in sys.path:
        try:
            rel = resolved.relative_to(Path(entry).resolve())
        except ValueError:
            continue
        return ".".join(rel.with_suffix("").parts)
    return None


def _build(qualname: str, source: str, dgen_dir: Path) -> "Dialect":
    """Parse ``source`` and build a Dialect under ``qualname``.

    Cross-dialect imports inside ``source`` resolve relative to ``dgen_dir``
    for ``from .`` and via :func:`load` otherwise.

    Registers a synthetic module under ``qualname`` in ``sys.modules`` and
    builds into its namespace so that ``@dataclass`` can resolve string
    annotations on generated classes (it looks up ``cls.__module__`` in
    ``sys.modules``). Without this, any source containing an ``op`` —
    whose generated dataclass has annotations like ``"Block"`` /
    ``"Type"`` / ``"Value"`` — fails to instantiate at build time.
    """
    mod = ModuleType(qualname)
    sys.modules[qualname] = mod
    return build(parse(source), qualname=qualname, dgen_dir=dgen_dir, ns=mod.__dict__)


# ---------------------------------------------------------------------------
# Python import hook
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _DgenLoader(importlib.abc.Loader):
    """Loader that compiles a ``.dgen`` file into a live Python module."""

    path: Path
    qualname: str

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        # If the dialect is already loaded (e.g. via dgen.imports.load), don't
        # rebuild it — DIALECTS is authoritative, like sys.modules.  Inject
        # the existing types and ops by name so callers of
        # ``from <this-module> import X`` keep working.
        existing = DIALECTS.get(self.qualname)
        if existing is not None:
            ns = module.__dict__
            ns[self.qualname.split(".")[-1]] = existing
            ns.update(existing.types)
            ns.update(existing.ops)
            return

        build(
            parse(self.path.read_text()),
            qualname=self.qualname,
            dgen_dir=self.path.parent,
            ns=module.__dict__,
        )


@dataclasses.dataclass
class _DgenFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that locates ``.dgen`` files via Python's normal
    import resolution.  When a ``<name>.dgen`` file sits where Python would
    have looked for ``<name>.py``, intercept and load it.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if path is not None:
            # Sub-package import: search the parent's __path__.
            name = fullname.rsplit(".", 1)[-1]
            search_dirs: list[str] = list(path)
        else:
            # Top-level: convert dots to directory separators.
            name = str(Path(*fullname.split(".")))
            search_dirs = sys.path
        for search_dir in search_dirs:
            dgen_file = (
                Path(search_dir) if search_dir else Path.cwd()
            ) / f"{name}.dgen"
            if dgen_file.exists():
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _DgenLoader(path=dgen_file, qualname=dgen_file.stem),
                    origin=str(dgen_file),
                )
        return None


_finder: _DgenFinder | None = None


def install_hook() -> None:
    """Install the ``.dgen`` import hook into ``sys.meta_path`` (idempotent)."""
    global _finder
    if _finder is None:
        _finder = _DgenFinder()
        sys.meta_path.insert(0, _finder)
