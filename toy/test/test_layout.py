"""Tests for memory layout types."""

from dgen.dialects import builtin
from dgen.layout import BYTE, FLOAT64, INT, Array, Bytes, FatPointer, Pointer
from dgen.module import string_value


def test_primitive_sizes():
    assert BYTE.byte_size == 1
    assert INT.byte_size == 8
    assert FLOAT64.byte_size == 8


def test_array():
    assert Array(BYTE, 8).byte_size == 8
    assert Array(FLOAT64, 6).byte_size == 48


def test_pointer():
    assert Pointer(FLOAT64).byte_size == 8
    assert Pointer(INT).byte_size == 8


def test_fat_pointer():
    assert FatPointer(BYTE).byte_size == 16
    assert FatPointer(INT).byte_size == 16


def test_bytes_layout():
    b = Bytes(5)
    assert b.byte_size == 5
    assert b.parse(b"hello") == b"hello"


def test_string_layout():
    s = builtin.String.for_value("hello")
    assert isinstance(s.__layout__, FatPointer)
    assert s.__layout__.byte_size == 16


def test_string_fatpointer_layout():
    """String type uses FatPointer(BYTE) layout — 16 bytes, not inline."""
    s = builtin.String()  # No params needed
    layout = s.__layout__
    assert isinstance(layout, FatPointer)
    assert layout.byte_size == 16
    assert layout.pointee is BYTE


def test_string_constant_fatpointer():
    """String constant via FatPointer: data in origin, pointer in Memory."""
    import ctypes

    c = builtin.String().constant("hello")
    mem = c.__constant__
    # Memory is 16 bytes (ptr + i64 length)
    assert mem.layout.byte_size == 16
    # Unpack gives (pointer, length)
    ptr, length = mem.unpack()
    assert length == 5
    # Pointer dereferences to "hello"
    data = bytes((ctypes.c_char * length).from_address(ptr))
    assert data == b"hello"
    # string_value still works
    assert string_value(c) == "hello"


def test_string_type_asm_no_params():
    """String type formats as 'String' with no angle brackets."""
    from dgen.asm.formatting import type_asm

    s = builtin.String()
    assert type_asm(s) == "String"


def test_list_fatpointer_layout():
    """List type uses FatPointer(element.__layout__) — 16 bytes, not inline."""
    list_type = builtin.List(element_type=builtin.IndexType())
    layout = list_type.__layout__
    assert isinstance(layout, FatPointer)
    assert layout.byte_size == 16
    assert layout.pointee is INT


def test_list_constant_fatpointer():
    """List constant via FatPointer: data in origin, pointer in Memory."""
    import ctypes

    list_type = builtin.List(element_type=builtin.IndexType())
    c = list_type.constant([10, 20, 30])
    mem = c.__constant__
    # Memory is 16 bytes (ptr + i64 length)
    assert mem.layout.byte_size == 16
    # Unpack gives (pointer, length)
    ptr, length = mem.unpack()
    assert length == 3
    # Pointer dereferences to [10, 20, 30]
    arr = (ctypes.c_int64 * length).from_address(ptr)
    assert list(arr) == [10, 20, 30]
    # to_python round-trip
    assert mem.to_python() == [10, 20, 30]


def test_list_type_asm_one_param():
    """List type formats as 'List<index>' — just element_type, no count."""
    from dgen.asm.formatting import type_asm

    list_type = builtin.List(element_type=builtin.IndexType())
    assert type_asm(list_type) == "List<index>"


def test_int_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.IndexType(), 42)
    assert mem.to_json() == 42


def test_float_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.F64Type(), 3.14)
    assert mem.to_json() == 3.14


def test_byte_to_json():
    buf = bytearray(1)
    BYTE.struct.pack_into(buf, 0, 65)
    assert BYTE.to_json(buf, 0) == 65


def test_f64type_layout():
    assert builtin.F64Type().__layout__.byte_size == 8


def test_index_type_layout():
    assert builtin.IndexType().__layout__.byte_size == 8


def test_tensor_type_layout():
    from toy.dialects import shape_constant
    from toy.dialects.toy import TensorType

    t = TensorType(shape=shape_constant([2, 3]))
    layout = t.__layout__
    assert layout.byte_size == 48  # 6 * 8 bytes
    assert isinstance(layout, Array)
    assert layout.count == 6


def test_array_to_json():
    from dgen.type import Memory
    from toy.dialects import shape_constant
    from toy.dialects.toy import TensorType

    ty = TensorType(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]


def test_fatpointer_to_json():
    from dgen.type import Memory

    ty = builtin.List(element_type=builtin.IndexType())
    mem = Memory.from_value(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.String(), "hello")
    assert mem.to_json() == "hello"
