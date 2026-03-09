"""Tests for builtin.Tuple type."""

from dgen import layout
from dgen.dialects.builtin import F64, Index, String, Tuple


def test_tuple_construction():
    """Tuple<[Index, String]> constructs with a list of types."""
    t = Tuple(types=[Index(), String()])
    assert len(t.types) == 2


def test_tuple_layout():
    """Tuple<[Index, String]> has Record layout with fields "0", "1"."""
    t = Tuple(types=[Index(), String()])
    expected = layout.Record([("0", layout.Int()), ("1", layout.String())])
    assert t.__layout__ == expected


def test_empty_tuple_layout():
    """Tuple<[]> has empty Record layout (zero bytes)."""
    t = Tuple(types=[])
    expected = layout.Record([])
    assert t.__layout__ == expected
    assert t.__layout__.byte_size == 0


def test_tuple_three_types():
    """Tuple<[Index, F64, Index]> layout has three fields."""
    t = Tuple(types=[Index(), F64(), Index()])
    expected = layout.Record(
        [
            ("0", layout.Int()),
            ("1", layout.Float64()),
            ("2", layout.Int()),
        ]
    )
    assert t.__layout__ == expected
