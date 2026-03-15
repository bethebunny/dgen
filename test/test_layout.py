"""Tests for memory layout types."""

import ctypes

from dgen import layout
from dgen.asm.formatting import type_asm
from dgen.dialects import builtin
from dgen.layout import Array, Byte, Span, Float64, Pointer
from dgen.module import string_value
from dgen.type import Constant, Memory, Type, TypeType, Value


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
    assert Span(Byte()).byte_size == 16
    assert Span(layout.Int()).byte_size == 16


def test_string_layout():
    s = builtin.String()
    assert isinstance(s.__layout__, Span)
    assert s.__layout__.byte_size == 16


def test_string_layout_is_string_layout():
    """String type uses layout.String — 16 bytes, Span subclass."""
    s = builtin.String()
    ly = s.__layout__
    assert isinstance(ly, layout.String)
    assert ly.byte_size == 16


def test_string_constant_fatpointer():
    """String constant via Span: data in origin, pointer in Memory."""
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
    s = builtin.String()
    assert type_asm(s) == "String"


def test_list_fatpointer_layout():
    """List type uses Span(element.__layout__) — 16 bytes, not inline."""
    list_type = builtin.List(element_type=builtin.Index())
    ly = list_type.__layout__
    assert isinstance(ly, Span)
    assert ly.byte_size == 16
    assert isinstance(ly.pointee, layout.Int)


def test_list_constant_fatpointer():
    """List constant via Span: data in origin, pointer in Memory."""
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
    list_type = builtin.List(element_type=builtin.Index())
    assert type_asm(list_type) == "List<Index>"


def test_int_to_json():
    mem = Memory.from_value(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_float_to_json():
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


def test_fatpointer_to_json():
    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_to_json():
    mem = Memory.from_value(builtin.String(), "hello")
    assert mem.to_json() == "hello"


def test_int_from_json_roundtrip():
    mem = Memory.from_json(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_list_from_json_roundtrip():
    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_json(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_from_json_roundtrip():
    mem = Memory.from_json(builtin.String(), "hello")
    assert mem.to_json() == "hello"


def test_type_tag_layout():
    """TypeTag wraps a String layout — 16 bytes."""
    tag = builtin.TypeTag()
    assert tag.__layout__.byte_size == 16
    assert isinstance(tag.__layout__, layout.String)


def test_nested_list_from_json_roundtrip():
    inner = builtin.List(element_type=builtin.Index())
    outer = builtin.List(element_type=inner)
    mem = Memory.from_json(outer, [[1, 2], [3, 4, 5]])
    assert mem.to_json() == [[1, 2], [3, 4, 5]]


def test_type_layout_non_parametric():
    """Non-parametric type layout is a fixed-size TypeValue pointer (8 bytes)."""
    from dgen.layout import TypeValue

    ty = builtin.Index()
    tl = ty.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_layout_parametric_value_param():
    """Parametric type layout is a fixed-size TypeValue pointer (8 bytes)."""
    from dgen.layout import TypeValue

    list_type = builtin.List(element_type=builtin.Index())
    tl = list_type.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_layout_parametric_type_param_nested():
    """Nested parametric type layout is still a fixed-size TypeValue pointer."""
    from dgen.layout import TypeValue

    inner = builtin.List(element_type=builtin.F64())
    outer = builtin.List(element_type=inner)
    tl = outer.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_value_memory_non_parametric():
    """Pack and unpack Index() as a type value through Memory."""
    metatype = TypeType()
    mem = Memory.from_json(metatype, {"tag": "builtin.Index"})
    assert mem.to_json() == {"tag": "builtin.Index"}


def test_type_value_memory_parametric():
    """Pack and unpack List<Index> as a type value through Memory."""
    metatype = TypeType()
    data = {"tag": "builtin.List", "element_type": {"tag": "builtin.Index"}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_pointer():
    """Pack and unpack Pointer<F64> as a type value through Memory."""
    metatype = TypeType()
    data = {"tag": "builtin.Pointer", "pointee": {"tag": "builtin.F64"}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_nil():
    """Pack and unpack Nil as a type value through Memory."""
    metatype = TypeType()
    mem = Memory.from_json(metatype, {"tag": "builtin.Nil"})
    assert mem.to_json() == {"tag": "builtin.Nil"}


def test_type_value_memory_nested():
    """Pack and unpack List<List<F64>> as a type value through Memory."""
    metatype = TypeType()
    data = {
        "tag": "builtin.List",
        "element_type": {
            "tag": "builtin.List",
            "element_type": {"tag": "builtin.F64"},
        },
    }
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_type_layout_non_parametric():
    """TypeType() layout matches Index().type.__layout__."""
    tt = TypeType()
    assert tt.__layout__ == builtin.Index().type.__layout__


def test_type_type_layout_parametric():
    """TypeType() layout matches the list's type.__layout__."""
    inner = builtin.List(element_type=builtin.Index())
    tt = TypeType()
    assert tt.__layout__ == inner.type.__layout__


def test_type_layout_size_fixed():
    """Type layout size is fixed (8-byte pointer) regardless of params."""
    simple = builtin.List(element_type=builtin.Index())
    nested = builtin.List(element_type=builtin.List(element_type=builtin.F64()))
    assert simple.type.__layout__.byte_size == 8
    assert nested.type.__layout__.byte_size == 8
    assert simple.type.__layout__.byte_size == nested.type.__layout__.byte_size


# ===----------------------------------------------------------------------=== #
# Type-is-Value tests
# ===----------------------------------------------------------------------=== #


def test_type_is_value():
    """Every Type instance is a Value."""
    ty = builtin.F64()
    assert isinstance(ty, Value)
    assert isinstance(ty.type, TypeType)


def test_type_constant_non_parametric():
    """Non-parametric type's __constant__ serializes to just a tag."""
    ty = builtin.F64()
    assert ty.__constant__.to_json() == {"tag": "builtin.F64"}


def test_type_constant_parametric():
    """Parametric type's __constant__ includes param values."""
    ty = builtin.List(element_type=builtin.Index())
    data = ty.__constant__.to_json()
    expected = {"tag": "builtin.List", "element_type": {"tag": "builtin.Index"}}
    assert data == expected


def test_type_params_are_bare_types():
    """Type-kinded params are bare Type instances, not Constant[TypeType]."""
    ty = builtin.List(element_type=builtin.Index())
    assert isinstance(ty.element_type, builtin.Index)
    assert isinstance(ty.element_type, Type)
    assert not isinstance(ty.element_type, Constant)
