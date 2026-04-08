"""Tests for the IRSnapshotExtension syrupy integration."""

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.testing.syrupy import IRSnapshotExtension
from toy.dialects import toy  # noqa: F401 — registers toy dialect
from dgen.testing import strip_prefix


IR = strip_prefix("""
    | import function
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %main : function.Function<[], Nil> = function.function<Nil>() body():
    |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    |     %1 : Nil = toy.print(%0)
""")

IR_RENAMED = strip_prefix("""
    | import function
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %main : function.Function<[], Nil> = function.function<Nil>() body():
    |     %tensor : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    |     %result : Nil = toy.print(%tensor)
""")

IR_DIFFERENT = strip_prefix("""
    | import function
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %main : function.Function<[], Nil> = function.function<Nil>() body():
    |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    |     %1 : Nil = toy.print(%0)
""")


def test_serialize_produces_ir_text(ir_snapshot):
    """serialize() formats the Module as IR text."""
    ext = IRSnapshotExtension()
    value = parse(IR)
    result = ext.serialize(value)
    assert "toy.print" in result
    assert "function" in result


def test_matches_graph_equivalent(ir_snapshot):
    """matches() returns True for graph-equivalent modules (SSA names differ)."""
    ext = IRSnapshotExtension()
    a = asm.format(parse(IR))
    b = asm.format(parse(IR_RENAMED))
    assert ext.matches(serialized_data=a, snapshot_data=b)


def test_matches_false_for_different_ir(ir_snapshot):
    """matches() returns False when modules have different values."""
    ext = IRSnapshotExtension()
    a = asm.format(parse(IR))
    b = asm.format(parse(IR_DIFFERENT))
    assert not ext.matches(serialized_data=a, snapshot_data=b)


def test_diff_lines_empty_when_equivalent(ir_snapshot):
    """diff_lines() yields nothing when modules are graph-equivalent."""
    ext = IRSnapshotExtension()
    a = asm.format(parse(IR))
    b = asm.format(parse(IR_RENAMED))
    lines = list(ext.diff_lines(a, b))
    assert lines == []


def test_diff_lines_shows_diff_when_different(ir_snapshot):
    """diff_lines() yields diff output when modules differ semantically."""
    ext = IRSnapshotExtension()
    a = asm.format(parse(IR))
    b = asm.format(parse(IR_DIFFERENT))
    lines = list(ext.diff_lines(a, b))
    assert lines  # non-empty on semantic change


def test_serialize_rejects_non_module(ir_snapshot):
    """serialize() raises AssertionError for non-Module input."""
    ext = IRSnapshotExtension()
    with pytest.raises(AssertionError, match="IRSnapshotExtension expects a Value"):
        ext.serialize("not a value")


def test_ir_snapshot_fixture_end_to_end(ir_snapshot):
    """Snapshot round-trip: generate then verify against same value."""
    value = parse(IR)
    assert value == ir_snapshot
