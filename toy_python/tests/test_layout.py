"""Tests for memory layout types."""

from toy_python.layout import BYTE, INT, FLOAT64, Array, Pointer, FatPointer


def test_primitive_sizes():
    assert BYTE.byte_size() == 1
    assert INT.byte_size() == 8
    assert FLOAT64.byte_size() == 8


def test_array():
    assert Array(BYTE, 8).byte_size() == 8
    assert Array(FLOAT64, 6).byte_size() == 48


def test_pointer():
    assert Pointer(FLOAT64).byte_size() == 8
    assert Pointer(INT).byte_size() == 8


def test_fat_pointer():
    assert FatPointer(BYTE).byte_size() == 16
    assert FatPointer(INT).byte_size() == 16


def test_string_layout():
    from toy_python.dialects.builtin import String

    assert String.__layout__.byte_size() == 16


def test_f64type_layout():
    from toy_python.dialects.affine import F64Type

    assert F64Type().__layout__.byte_size() == 8


def test_index_type_layout():
    from toy_python.dialects.affine import IndexType

    assert IndexType().__layout__.byte_size() == 8
