"""Tests for the IRSnapshotExtension syrupy integration."""

import pytest

from dgen.asm.parser import parse_module
from dgen.syrupy_ext import IRSnapshotExtension
from toy.dialects import toy  # noqa: F401 — registers toy dialect
from toy.test.helpers import strip_prefix


IR = strip_prefix("""
    | import toy
    |
    | %main : Nil = function<Nil>() ():
    |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    |     %1 : Nil = toy.print(%0)
    |     %_ : Nil = return(%1)
""")

IR_RENAMED = strip_prefix("""
    | import toy
    |
    | %main : Nil = function<Nil>() ():
    |     %tensor : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    |     %result : Nil = toy.print(%tensor)
    |     %_ : Nil = return(%result)
""")

IR_DIFFERENT = strip_prefix("""
    | import toy
    |
    | %main : Nil = function<Nil>() ():
    |     %0 : toy.Tensor<[2, 3], F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    |     %1 : Nil = toy.print(%0)
    |     %_ : Nil = return(%1)
""")


def test_serialize_produces_ir_text(ir_snapshot):
    """serialize() formats the Module as IR text."""
    ext = IRSnapshotExtension()
    module = parse_module(IR)
    result = ext.serialize(module)
    assert "toy.print" in result
    assert "function" in result


def test_matches_graph_equivalent(ir_snapshot):
    """matches() returns True for graph-equivalent modules (SSA names differ)."""
    ext = IRSnapshotExtension()
    a = "\n".join(parse_module(IR).asm)
    b = "\n".join(parse_module(IR_RENAMED).asm)
    assert ext.matches(serialized_data=a, snapshot_data=b)


def test_matches_false_for_different_ir(ir_snapshot):
    """matches() returns False when modules have different values."""
    ext = IRSnapshotExtension()
    a = "\n".join(parse_module(IR).asm)
    b = "\n".join(parse_module(IR_DIFFERENT).asm)
    assert not ext.matches(serialized_data=a, snapshot_data=b)


def test_diff_lines_empty_when_equivalent(ir_snapshot):
    """diff_lines() yields nothing when modules are graph-equivalent."""
    ext = IRSnapshotExtension()
    a = "\n".join(parse_module(IR).asm)
    b = "\n".join(parse_module(IR_RENAMED).asm)
    lines = list(ext.diff_lines(a, b))
    assert lines == []


def test_diff_lines_shows_diff_when_different(ir_snapshot):
    """diff_lines() yields diff output when modules differ semantically."""
    ext = IRSnapshotExtension()
    a = "\n".join(parse_module(IR).asm)
    b = "\n".join(parse_module(IR_DIFFERENT).asm)
    lines = list(ext.diff_lines(a, b))
    assert lines  # non-empty on semantic change


def test_serialize_rejects_non_module(ir_snapshot):
    """serialize() raises AssertionError for non-Module input."""
    ext = IRSnapshotExtension()
    with pytest.raises(AssertionError, match="IRSnapshotExtension expects a Module"):
        ext.serialize("not a module")


def test_ir_snapshot_fixture_end_to_end(ir_snapshot):
    """Snapshot round-trip: generate then verify against same module."""
    module = parse_module(IR)
    assert module == ir_snapshot
