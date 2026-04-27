"""Tests for the IRSnapshotExtension syrupy integration."""

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import control_flow
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import LoweringSnapshot, strip_prefix
from dgen.testing.syrupy import IRSnapshotExtension
from toy.dialects import toy  # noqa: F401 — registers toy dialect


IR = strip_prefix("""
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    | %1 : Nil = toy.print(%0)
""")

IR_RENAMED = strip_prefix("""
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %tensor : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    | %result : Nil = toy.print(%tensor)
""")

IR_DIFFERENT = strip_prefix("""
    | import ndbuffer
    | import number
    | import toy
    | import index
    |
    | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    | %1 : Nil = toy.print(%0)
""")


def test_serialize_produces_ir_text(ir_snapshot):
    """serialize() formats the Module as IR text."""
    ext = IRSnapshotExtension()
    value = parse(IR)
    result = ext.serialize(value)
    assert "toy.print" in result
    assert "toy.Tensor" in result


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


# -- LoweringSnapshot serialization --


def test_lowering_snapshot_serialize_emits_header():
    """``LoweringSnapshot`` serializes with a leading ``#``-prefixed header."""
    ext = IRSnapshotExtension()
    value = parse(IR)
    snapshot = LoweringSnapshot(
        result=value,
        pass_names=("FakePass",),
        input_asm="import toy\n%x : index.Index = 1",
    )
    serialized = ext.serialize(snapshot)
    assert isinstance(serialized, str)
    head, _, body = serialized.partition("\n\n")
    # Header is comment-only and names the pass.
    assert all(line == "" or line.startswith("#") for line in head.splitlines())
    assert "FakePass" in head
    assert "import toy" in head
    # Body is parseable IR (no comment prefix).
    assert "toy.print" in body
    assert "#" not in body.splitlines()[0]


def test_lowering_snapshot_serialize_multiple_passes():
    """Multi-pass header lists each pass on its own bullet line."""
    ext = IRSnapshotExtension()
    snapshot = LoweringSnapshot(
        result=parse(IR),
        pass_names=("First", "Second", "Third"),
        input_asm="import toy",
    )
    serialized = ext.serialize(snapshot)
    header = serialized.split("\n\n", 1)[0]
    assert "Lowered through 3 passes:" in header
    for name in ("First", "Second", "Third"):
        assert f"#   - {name}" in header


def test_lowering_snapshot_header_is_round_trippable():
    """The ``#`` comment header is ignored by the parser, so the snapshot file
    round-trips through ``parse``."""
    ext = IRSnapshotExtension()
    snapshot = LoweringSnapshot(
        result=parse(IR),
        pass_names=("FakePass",),
        input_asm="import toy\n%x : index.Index = 1",
    )
    serialized = ext.serialize(snapshot)
    reparsed = parse(serialized)
    assert ext.matches(
        serialized_data=serialized,
        snapshot_data="\n".join(asm.asm_with_imports(reparsed)),
    )


def test_lowering_snapshot_fixture_runs_passes(lowering_snapshot):
    """The ``lowering_snapshot`` fixture parses input, runs passes, and returns
    the lowered ``Value`` so the test can do additional assertions."""
    result = lowering_snapshot(
        [ControlFlowToGoto()],
        """
        | import algebra
        | import control_flow
        | import index
        | import number
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %brk : Never = control_flow.break()
        """,
    )
    # Returned value is the lowered IR; can drive further assertions.
    assert not isinstance(result, control_flow.WhileOp)
