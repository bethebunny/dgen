"""Tests for memory layout types."""

from dgen import layout
from dgen.dialects import builtin
from dgen.layout import Array, Byte, FatPointer, Float64, Pointer
from dgen.module import string_value


def test_primitive_sizes():
    assert Byte().byte_size == 1
    assert layout.Int().byte_size == 8
    assert Float64().byte_size == 8


def test_array():
    assert Array(Byte(), 8).byte_size == 8
    assert Array(Float64(), 6).byte_size == 48


def test_pointer():
    assert Pointer(Float64()).byte_size == 8
    assert Pointer(layout.Int()).byte_size == 8


def test_fat_pointer():
    assert FatPointer(Byte()).byte_size == 16
    assert FatPointer(layout.Int()).byte_size == 16


def test_string_layout():
    s = builtin.String.for_value("hello")
    assert isinstance(s.__layout__, FatPointer)
    assert s.__layout__.byte_size == 16


def test_string_layout_is_string_layout():
    """String type uses layout.String — 16 bytes, FatPointer subclass."""
    s = builtin.String()
    ly = s.__layout__
    assert isinstance(ly, layout.String)
    assert ly.byte_size == 16


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
    list_type = builtin.List(element_type=builtin.Index())
    ly = list_type.__layout__
    assert isinstance(ly, FatPointer)
    assert ly.byte_size == 16
    assert isinstance(ly.pointee, layout.Int)


def test_list_constant_fatpointer():
    """List constant via FatPointer: data in origin, pointer in Memory."""
    import ctypes

    list_type = builtin.List(element_type=builtin.Index())
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
    # to_json round-trip
    assert mem.to_json() == [10, 20, 30]


def test_list_type_asm_one_param():
    """List type formats as 'List<index>' — just element_type, no count."""
    from dgen.asm.formatting import type_asm

    list_type = builtin.List(element_type=builtin.Index())
    assert type_asm(list_type) == "List<Index>"


def test_int_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_float_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.F64(), 3.14)
    assert mem.to_json() == 3.14


def test_byte_to_json():
    b = Byte()
    buf = bytearray(1)
    b.struct.pack_into(buf, 0, 65)
    assert b.to_json(buf, 0) == 65


def test_f64type_layout():
    assert builtin.F64().__layout__.byte_size == 8


def test_index_type_layout():
    assert builtin.Index().__layout__.byte_size == 8


def test_tensor_type_layout():
    from toy.dialects import shape_constant
    from toy.dialects.toy import Tensor

    t = Tensor(shape=shape_constant([2, 3]))
    ly = t.__layout__
    assert ly.byte_size == 48  # 6 * 8 bytes
    assert isinstance(ly, Array)
    assert ly.count == 6


def test_array_to_json():
    from dgen.type import Memory
    from toy.dialects import shape_constant
    from toy.dialects.toy import Tensor

    ty = Tensor(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]


def test_fatpointer_to_json():
    from dgen.type import Memory

    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_to_json():
    from dgen.type import Memory

    mem = Memory.from_value(builtin.String(), "hello")
    assert mem.to_json() == "hello"


def test_int_from_json_roundtrip():
    from dgen.type import Memory

    mem = Memory.from_json(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_list_from_json_roundtrip():
    from dgen.type import Memory

    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_json(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_from_json_roundtrip():
    from dgen.type import Memory

    mem = Memory.from_json(builtin.String(), "hello")
    assert mem.to_json() == "hello"


def test_type_tag_layout():
    """TypeTag wraps a String layout — 16 bytes."""
    tag = builtin.TypeTag()
    assert tag.__layout__.byte_size == 16
    assert isinstance(tag.__layout__, layout.String)


def test_nested_list_from_json_roundtrip():
    from dgen.type import Memory

    inner = builtin.List(element_type=builtin.Index())
    outer = builtin.List(element_type=inner)
    mem = Memory.from_json(outer, [[1, 2], [3, 4, 5]])
    assert mem.to_json() == [[1, 2], [3, 4, 5]]
