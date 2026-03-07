"""Tests for memory layout types."""

import ctypes

from dgen import layout
from dgen.asm.formatting import type_asm
from dgen.dialects import builtin
from dgen.layout import Array, Byte, FatPointer, Float64, Pointer, Record
from dgen.layout import String as StringLayout
from dgen.module import string_value
from dgen.type import Constant, Memory, Type, TypeType, Value
from toy.dialects import shape_constant
from toy.dialects.toy import Tensor


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
    s = builtin.String()
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
    """List type uses FatPointer(element.__layout__) — 16 bytes, not inline."""
    list_type = builtin.List(element_type=builtin.Index())
    ly = list_type.__layout__
    assert isinstance(ly, FatPointer)
    assert ly.byte_size == 16
    assert isinstance(ly.pointee, layout.Int)


def test_list_constant_fatpointer():
    """List constant via FatPointer: data in origin, pointer in Memory."""
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


def test_tensor_type_layout():
    t = Tensor(shape=shape_constant([2, 3]))
    ly = t.__layout__
    assert ly.byte_size == 48  # 6 * 8 bytes
    assert isinstance(ly, Array)
    assert ly.count == 6


def test_array_to_json():
    ty = Tensor(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]


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
    """Non-parametric type has layout Record([("tag", String)])."""
    ty = builtin.Index()
    tl = ty.type.__layout__
    assert isinstance(tl, Record)
    assert len(tl.fields) == 1
    assert tl.fields[0][0] == "tag"
    assert isinstance(tl.fields[0][1], StringLayout)


def test_type_layout_parametric_value_param():
    """Parametric type with value param includes param's type layout."""
    list_type = builtin.List(element_type=builtin.Index())
    tl = list_type.type.__layout__
    assert isinstance(tl, Record)
    # tag + element_type
    assert len(tl.fields) == 2
    assert tl.fields[0][0] == "tag"
    assert tl.fields[1][0] == "element_type"
    # element_type is Type-kinded, so it's Index's type.__layout__ (a Record)
    inner = tl.fields[1][1]
    assert isinstance(inner, Record)


def test_type_layout_parametric_type_param_nested():
    """List<List<F64>> inlines nested type layouts."""
    inner = builtin.List(element_type=builtin.F64())
    outer = builtin.List(element_type=inner)
    tl = outer.type.__layout__
    assert isinstance(tl, Record)
    assert len(tl.fields) == 2
    # element_type field is inner's type.__layout__
    inner_tl = tl.fields[1][1]
    assert isinstance(inner_tl, Record)
    assert len(inner_tl.fields) == 2
    # inner's element_type is F64's type.__layout__ (just a tag)
    f64_tl = inner_tl.fields[1][1]
    assert isinstance(f64_tl, Record)
    assert len(f64_tl.fields) == 1


def test_type_value_memory_non_parametric():
    """Pack and unpack Index() as a type value through Memory."""
    ty = builtin.Index()
    metatype = TypeType(concrete=ty)
    mem = Memory.from_json(metatype, {"tag": "builtin.Index"})
    assert mem.to_json() == {"tag": "builtin.Index"}


def test_type_value_memory_parametric():
    """Pack and unpack List<Index> as a type value through Memory."""
    metatype = TypeType(concrete=builtin.List(element_type=builtin.Index()))
    data = {"tag": "builtin.List", "element_type": {"tag": "builtin.Index"}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_pointer():
    """Pack and unpack Pointer<F64> as a type value through Memory."""
    metatype = TypeType(concrete=builtin.Pointer(pointee=builtin.F64()))
    data = {"tag": "builtin.Pointer", "pointee": {"tag": "builtin.F64"}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_nil():
    """Pack and unpack Nil as a type value through Memory."""
    metatype = TypeType(concrete=builtin.Nil())
    mem = Memory.from_json(metatype, {"tag": "builtin.Nil"})
    assert mem.to_json() == {"tag": "builtin.Nil"}


def test_type_value_memory_nested():
    """Pack and unpack List<List<F64>> as a type value through Memory."""
    inner = builtin.List(element_type=builtin.F64())
    metatype = TypeType(concrete=builtin.List(element_type=inner))
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
    """TypeType(concrete=Index()) layout matches Index().type.__layout__."""
    tt = TypeType(concrete=builtin.Index())
    assert tt.__layout__ == builtin.Index().type.__layout__


def test_type_type_layout_parametric():
    """TypeType(concrete=List(Index())) layout matches the list's type.__layout__."""
    inner = builtin.List(element_type=builtin.Index())
    tt = TypeType(concrete=inner)
    assert tt.__layout__ == inner.type.__layout__


def test_type_layout_size_varies_by_params():
    """Type layout size depends on concrete params (inline design)."""
    # List<Index> and List<List<F64>> have different type.__layout__ sizes
    simple = builtin.List(element_type=builtin.Index())
    nested = builtin.List(element_type=builtin.List(element_type=builtin.F64()))
    assert simple.type.__layout__.byte_size < nested.type.__layout__.byte_size


# ===----------------------------------------------------------------------=== #
# Type-is-Value tests
# ===----------------------------------------------------------------------=== #


def test_type_is_value():
    """Every Type instance is a Value."""
    ty = builtin.F64()
    assert isinstance(ty, Value)
    assert ty.ready
    assert isinstance(ty.type, TypeType)
    assert ty.type.concrete is ty


def test_type_constant_non_parametric():
    """Non-parametric type's __constant__ serializes to just a tag."""
    ty = builtin.F64()
    assert ty.__constant__.to_json() == {"tag": "builtin.F64"}


def test_type_constant_parametric():
    """Parametric type's __constant__ includes param values."""
    ty = builtin.List(element_type=builtin.Index())
    data = ty.__constant__.to_json()
    assert data["tag"] == "builtin.List"
    assert data["element_type"] == {"tag": "builtin.Index"}


def test_type_params_are_bare_types():
    """Type-kinded params are bare Type instances, not Constant[TypeType]."""
    ty = builtin.List(element_type=builtin.Index())
    assert isinstance(ty.element_type, builtin.Index)
    assert isinstance(ty.element_type, Type)
    assert not isinstance(ty.element_type, Constant)
