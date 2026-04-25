"""Tests for the dialect import system: dgen.PATH, dgen.DIALECTS, dgen.imports.load."""

from __future__ import annotations


import pytest

import dgen


def test_load_caches_in_dialects():
    """load() returns the same Dialect instance on repeated calls."""
    a = dgen.imports.load("builtin")
    b = dgen.imports.load("builtin")
    assert a is b
    assert dgen.DIALECTS["builtin"] is a


def test_load_missing_dialect_raises_import_error():
    with pytest.raises(ImportError, match="Cannot find dgen dialect 'no_such_dialect'"):
        dgen.imports.load("no_such_dialect")


def test_load_inline_source():
    """load(name, source=...) parses an inline string and registers the dialect."""
    source = "type Widget:\n    layout Void\n"
    d = dgen.imports.load("inline_widget_dialect", source=source)
    assert "Widget" in d.types
    assert dgen.DIALECTS["inline_widget_dialect"] is d


def test_load_inline_source_with_cross_dialect_import():
    """An inline source can `from ... import ...` other dialects via PATH/DIALECTS."""
    source = "from index import Index\n\ntype HasIndex:\n    n: Index\n"
    d = dgen.imports.load("inline_with_index", source=source)
    assert "HasIndex" in d.types


def test_load_relative_only_searches_sibling(tmp_path):
    """``_from`` ignores dgen.PATH and searches only the given directory."""
    (tmp_path / "sibling.dgen").write_text("type SiblingType:\n    layout Void\n")

    # Cleanup any stale entry from previous runs.
    dgen.DIALECTS.pop("sibling", None)

    d = dgen.imports.load("sibling", _from=tmp_path)
    assert "SiblingType" in d.types

    # PATH lookup for the same name elsewhere should fail (file isn't on PATH).
    dgen.DIALECTS.pop("sibling", None)
    with pytest.raises(ImportError):
        dgen.imports.load("sibling")


def test_relative_import_in_dgen_file(tmp_path):
    """`from . import x` in a .dgen file resolves a sibling dialect file."""
    (tmp_path / "leaf.dgen").write_text("type Leaf:\n    layout Void\n")
    (tmp_path / "root.dgen").write_text(
        "from . import leaf\n\ntype RootHolder:\n    inner: leaf.Leaf\n"
    )

    dgen.DIALECTS.pop("leaf", None)
    dgen.DIALECTS.pop("root", None)

    d = dgen.imports.load("root", _from=tmp_path)
    assert "RootHolder" in d.types
    # The relative import should have populated DIALECTS for the sibling too.
    assert "leaf" in dgen.DIALECTS


def test_dotted_qualname_round_trips(tmp_path):
    """A dotted qualname maps to a nested ``<a>/<b>.dgen`` path."""
    nested = tmp_path / "x86"
    nested.mkdir()
    (nested / "intrinsics.dgen").write_text("type Reg:\n    layout Void\n")

    dgen.PATH.append(tmp_path)
    try:
        dgen.DIALECTS.pop("x86.intrinsics", None)
        d = dgen.imports.load("x86.intrinsics")
        assert d.name == "x86.intrinsics"
        assert dgen.DIALECTS["x86.intrinsics"] is d
        assert "Reg" in d.types
    finally:
        dgen.PATH.remove(tmp_path)


def test_path_search_picks_up_new_directories(tmp_path):
    """A dialect placed in a directory added to dgen.PATH is discoverable."""
    (tmp_path / "new_dialect.dgen").write_text("type Token:\n    layout Void\n")

    dgen.PATH.append(tmp_path)
    try:
        dgen.DIALECTS.pop("new_dialect", None)
        d = dgen.imports.load("new_dialect")
        assert "Token" in d.types
    finally:
        dgen.PATH.remove(tmp_path)


def test_relative_does_not_consult_path(tmp_path):
    """Relative form must not fall back to dgen.PATH."""
    # Place the file on PATH but not in the relative dir.
    (tmp_path / "on_path.dgen").write_text("type X:\n    layout Void\n")
    other = tmp_path / "other"
    other.mkdir()

    dgen.PATH.append(tmp_path)
    try:
        dgen.DIALECTS.pop("on_path", None)
        with pytest.raises(ImportError):
            dgen.imports.load("on_path", _from=other)
    finally:
        dgen.PATH.remove(tmp_path)


def test_imports_do_not_re_export(tmp_path):
    """`from foo import X` requires X to be defined in foo, not just imported.

    Names that ``foo`` itself imported are not visible to ``from foo import``
    elsewhere — every dialect must import names from where they are declared.
    """
    (tmp_path / "leaf.dgen").write_text("type Leaf:\n    layout Void\n")
    (tmp_path / "middle.dgen").write_text("from . import leaf\n")
    (tmp_path / "user.dgen").write_text(
        "from . import middle\n\ntype Holder:\n    inner: middle.Leaf\n"
    )

    for name in ("leaf", "middle", "user"):
        dgen.DIALECTS.pop(name, None)

    with pytest.raises(AttributeError, match="has no member 'Leaf'"):
        dgen.imports.load("user", _from=tmp_path)
