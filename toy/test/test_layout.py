"""Tests for memory layout types."""

from dgen.dialects import builtin
from dgen.layout import BYTE, FLOAT64, INT, Array, FatPointer, Memory, Pointer


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


# ---------------------------------------------------------------------------
# Memory.from_asm / Memory.from_value
# ---------------------------------------------------------------------------


def test_from_value_scalar_int():
    mem = Memory.from_value(builtin.IndexType(), 42)
    assert mem.unpack() == (42,)


def test_from_value_scalar_float():
    mem = Memory.from_value(builtin.F64Type(), 3.14)
    (val,) = mem.unpack()
    assert abs(val - 3.14) < 1e-12


def test_from_value_tensor():
    from toy.dialects.toy import TensorType

    t = TensorType(shape=[2, 3])
    mem = Memory.from_value(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert mem.unpack() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_from_asm_int():
    mem = Memory.from_asm(builtin.IndexType(), "42")
    assert mem.unpack() == (42,)


def test_from_asm_float():
    mem = Memory.from_asm(builtin.F64Type(), "3.14")
    (val,) = mem.unpack()
    assert abs(val - 3.14) < 1e-12


def test_from_asm_list():
    from toy.dialects.toy import TensorType

    t = TensorType(shape=[2, 3])
    mem = Memory.from_asm(t, "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]")
    assert mem.unpack() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
