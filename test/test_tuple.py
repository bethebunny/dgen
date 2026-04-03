"""Tests for builtin.Tuple type."""

import pytest

from dgen import asm, layout
from dgen.type import format_value as type_asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent
from dgen.dialects.builtin import Index, String, Tuple
from dgen.dialects.number import Float64
from dgen.module import pack
from dgen.type import _type_from_dict, type_constant
from dgen.testing import strip_prefix


def test_tuple_construction():
    """Tuple<[index.Index, String]> constructs with a list of types."""
    t = Tuple(types=pack([Index(), String()]))
    assert len(list(t.types)) == 2


def test_tuple_layout():
    """Tuple<[index.Index, String]> has Record layout with fields "0", "1"."""
    t = Tuple(types=pack([Index(), String()]))
    expected = layout.Record([("0", layout.Int()), ("1", layout.String())])
    assert t.__layout__ == expected


def test_empty_tuple_layout():
    """Tuple<[]> has empty Record layout (zero bytes)."""
    t = Tuple(types=pack([]))
    expected = layout.Record([])
    assert t.__layout__ == expected
    assert t.__layout__.byte_size == 0


def test_tuple_type_asm_format():
    """Tuple<[index.Index, String]> formats as Tuple<[index.Index, String]>."""
    t = Tuple(types=pack([Index(), String()]))
    assert type_asm(t) == "Tuple<[index.Index, String]>"


def test_empty_tuple_asm_format():
    """Tuple<[]> formats as Tuple<[]>."""
    t = Tuple(types=pack([]))
    assert type_asm(t) == "Tuple<[]>"


def test_tuple_constant_roundtrip():
    """Tuple type in IR round-trips through ASM."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %x : Tuple<[index.Index, String]> = [42, "hello"]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_tuple_type_constant_serialization():
    """Tuple.__constant__ serializes types list as a proper list of dicts."""
    t = Tuple(types=pack([Index(), String()]))
    data = t.__constant__.to_json()
    assert data == {
        "tag": "builtin.Tuple",
        "types": [{"tag": "index.Index"}, {"tag": "builtin.String"}],
    }


def test_tuple_type_from_dict_roundtrip():
    """Tuple type round-trips through __constant__ → _type_from_dict."""
    t = Tuple(types=pack([Index(), String()]))
    data = t.__constant__.to_json()
    reconstructed = _type_from_dict(data)
    assert isinstance(reconstructed, Tuple)
    types = list(reconstructed.types)
    assert len(types) == 2
    assert isinstance(type_constant(types[0]), Index)
    assert isinstance(type_constant(types[1]), String)


def test_tuple_three_types():
    """Tuple<[index.Index, Float64, Index]> layout has three fields."""
    t = Tuple(types=pack([Index(), Float64(), Index()]))
    expected = layout.Record(
        [
            ("0", layout.Int()),
            ("1", layout.Float64()),
            ("2", layout.Int()),
        ]
    )
    assert t.__layout__ == expected


@pytest.mark.xfail(
    reason="Parser stores Type objects as Tuple constant values; should store dicts"
)
def test_tuple_type_values():
    """Tuple of type values: %types : Tuple<[Type, Type]> = [Index, String]."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %types : Tuple<[Type, Type]> = [Index, String]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
