"""Tests for integer layouts sized by their bits."""

import pytest

from dgen import layout
from dgen.asm.parser import parse
from dgen.codegen import llvm_type
from dgen.dialects import llvm, number
from dgen.dialects.index import Index
from dgen.testing import llvm_compile, strip_prefix
from dgen.type import Memory


def signed(bits: int) -> number.SignedInteger:
    return number.SignedInteger(bits=Index().constant(bits))


def unsigned(bits: int) -> number.UnsignedInteger:
    return number.UnsignedInteger(bits=Index().constant(bits))


@pytest.mark.parametrize(("bits", "size"), [(1, 1), (8, 1), (16, 2), (32, 4), (64, 8)])
def test_int_size_follows_bits(bits: int, size: int):
    assert layout.Int(bits).byte_size == size
    assert layout.Int(bits).alignment == size


def test_signed_integer_layout():
    assert signed(32).__layout__ == layout.Int(32)
    assert str(signed(32).__layout__) == "int32"


def test_unsigned_integer_layout():
    assert unsigned(8).__layout__ == layout.UInt(8)
    assert Memory.from_json(unsigned(8), 255).to_json() == 255


def test_signed_integer_roundtrip():
    assert Memory.from_json(signed(16), -2).to_json() == -2
    assert Memory.from_json(signed(16), -2).buffer == b"\xfe\xff"


def test_llvm_int_layout():
    assert llvm.Int(bits=Index().constant(1)).__layout__.byte_size == 1


def test_llvm_type_uses_bits():
    assert llvm_type(signed(32)) == "i32"
    assert llvm_type(unsigned(16)) == "i16"


def test_int_descriptors_parse():
    assert layout.parse("int32") == layout.Int(32)
    assert layout.parse("uint8") == layout.UInt(8)
    assert layout.parse("record<a: int8, b: int64>").byte_size == 16


def test_i32_function_call():
    ir = strip_prefix("""
        | import algebra
        | import function
        | import number
        |
        | %add : function.Function<[number.SignedInteger<32>, number.SignedInteger<32>], number.SignedInteger<32>> = function.function<number.SignedInteger<32>>() body(%a: number.SignedInteger<32>, %b: number.SignedInteger<32>):
        |     %r : number.SignedInteger<32> = algebra.add(%a, %b)
    """)
    exe = llvm_compile(parse(ir))
    assert "define i32 @add(i32 %a, i32 %b)" in exe.ir
    assert exe.run(2**31 - 1, 1).to_json() == -(2**31)
