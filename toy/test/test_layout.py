"""Tests for memory layout types."""

from dgen.dialects import builtin
from dgen.layout import BYTE, FLOAT64, INT, Array, FatPointer, Pointer


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
    assert builtin.String.__layout__.byte_size() == 16


def test_f64type_layout():
    assert builtin.F64Type().__layout__.byte_size() == 8


def test_index_type_layout():
    assert builtin.IndexType().__layout__.byte_size() == 8


def test_tensor_type_layout():
    from toy.dialects.toy import TensorType

    t = TensorType(shape=[2, 3])
    layout = t.__layout__
    assert layout.byte_size() == 48  # 6 * 8 bytes
    assert layout.count == 6
