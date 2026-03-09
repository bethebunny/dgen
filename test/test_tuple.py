"""Tests for builtin.Tuple type."""

import json

from dgen import asm, layout
from dgen.asm.formatting import type_asm
from dgen.asm.parser import parse_module
from dgen.dialects.builtin import F64, Index, String, Tuple
from dgen.type import _type_from_dict, type_constant
from toy.test.helpers import strip_prefix


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


def test_tuple_type_asm_format():
    """Tuple<[Index, String]> formats as Tuple<[Index, String]>."""
    t = Tuple(types=[Index(), String()])
    assert type_asm(t) == "Tuple<[Index, String]>"


def test_empty_tuple_asm_format():
    """Tuple<[]> formats as Tuple<[]>."""
    t = Tuple(types=[])
    assert type_asm(t) == "Tuple<[]>"


def test_tuple_constant_roundtrip():
    """Tuple type in IR round-trips through ASM."""
    ir = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %x : Tuple<[Index, String]> = [42, "hello"]
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_tuple_type_constant_serialization():
    """Tuple.__constant__ serializes list params as JSON string."""
    t = Tuple(types=[Index(), String()])
    data = t.__constant__.to_json()
    assert isinstance(data, dict)
    assert data["tag"] == "builtin.Tuple"
    # List params are stored as JSON strings in the binary layout.
    assert isinstance(data["types"], str)
    decoded = json.loads(data["types"])
    assert len(decoded) == 2
    assert decoded[0]["tag"] == "builtin.Index"
    assert decoded[1]["tag"] == "builtin.String"


def test_tuple_type_from_dict_roundtrip():
    """Tuple type round-trips through __constant__ → _type_from_dict."""
    t = Tuple(types=[Index(), String()])
    data = t.__constant__.to_json()
    assert isinstance(data, dict)
    reconstructed = _type_from_dict(data)
    assert isinstance(reconstructed, Tuple)
    assert len(reconstructed.types) == 2
    assert type_constant(reconstructed.types[0]) == Index()
    assert type_constant(reconstructed.types[1]) == String()


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


def test_tuple_type_values():
    """Tuple of type values: %types : Tuple<[TypeType<Index>, TypeType<String>]> = [Index, String]."""
    ir = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %types : Tuple<[TypeType<Index>, TypeType<String>]> = [Index, String]
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
