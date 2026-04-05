"""Roundtrip tests for number dialect types."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import number  # noqa: F401 — registers the dialect for parsing
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_boolean():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        |
        | %f : function.Function<[number.Boolean], Nil> = function.function<Nil>() body(%x: number.Boolean):
        |     %0 : index.Index = 0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_signed_integer():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        |
        | %f : function.Function<[number.SignedInteger<64>], Nil> = function.function<Nil>() body(%x: number.SignedInteger<64>):
        |     %0 : index.Index = 0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_unsigned_integer():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        |
        | %f : function.Function<[number.UnsignedInteger<32>], Nil> = function.function<Nil>() body(%x: number.UnsignedInteger<32>):
        |     %0 : index.Index = 0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_float64():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        |
        | %f : function.Function<[number.Float64], Nil> = function.function<Nil>() body(%x: number.Float64):
        |     %0 : index.Index = 0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
