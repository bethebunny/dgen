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


def test_type_layout_non_parametric():
    """Non-parametric type has layout Record([("tag", String)])."""
    from dgen.layout import Record, String as StringLayout

    ty = builtin.Index()
    tl = ty.type_layout
    assert isinstance(tl, Record)
    assert len(tl.fields) == 1
    assert tl.fields[0][0] == "tag"
    assert isinstance(tl.fields[0][1], StringLayout)


def test_type_layout_parametric_value_param():
    """Parametric type with value param includes param's type layout."""
    from dgen.layout import Record

    list_type = builtin.List(element_type=builtin.Index())
    tl = list_type.type_layout
    assert isinstance(tl, Record)
    # tag + element_type
    assert len(tl.fields) == 2
    assert tl.fields[0][0] == "tag"
    assert tl.fields[1][0] == "element_type"
    # element_type is Type-kinded, so it's Index's type_layout (a Record)
    inner = tl.fields[1][1]
    assert isinstance(inner, Record)


def test_type_layout_parametric_type_param_nested():
    """List<List<F64>> inlines nested type layouts."""
    from dgen.layout import Record

    inner = builtin.List(element_type=builtin.F64())
    outer = builtin.List(element_type=inner)
    tl = outer.type_layout
    assert isinstance(tl, Record)
    assert len(tl.fields) == 2
    # element_type field is inner's type_layout
    inner_tl = tl.fields[1][1]
    assert isinstance(inner_tl, Record)
    assert len(inner_tl.fields) == 2
    # inner's element_type is F64's type_layout (just a tag)
    f64_tl = inner_tl.fields[1][1]
    assert isinstance(f64_tl, Record)
    assert len(f64_tl.fields) == 1


def test_type_to_json_non_parametric():
    """Non-parametric type serializes to just a tag."""
    ty = builtin.Index()
    result = ty.type_to_json()
    assert result == {"tag": "builtin.Index"}


def test_type_to_json_parametric():
    """Parametric type with type-kinded param serializes recursively."""
    ty = builtin.List(element_type=builtin.Index())
    result = ty.type_to_json()
    assert result == {
        "tag": "builtin.List",
        "element_type": {"tag": "builtin.Index"},
    }


def test_type_from_json_non_parametric():
    """Reconstruct Index() from its JSON representation."""
    from dgen.type import Type

    result = Type.type_from_json({"tag": "builtin.Index"})
    assert result == builtin.Index()


def test_type_from_json_parametric():
    """Reconstruct List<Index> from its JSON representation."""
    from dgen.type import Type

    result = Type.type_from_json(
        {
            "tag": "builtin.List",
            "element_type": {"tag": "builtin.Index"},
        }
    )
    assert result == builtin.List(element_type=builtin.Index())


def test_type_value_full_roundtrip():
    """type_to_json -> type_from_json round-trip."""
    from dgen.type import Type

    ty = builtin.List(element_type=builtin.F64())
    json_val = ty.type_to_json()
    reconstructed = Type.type_from_json(json_val)
    assert reconstructed == ty


def test_type_pack_roundtrip():
    """Pack a type value into Memory and read it back via to_json."""
    from dgen.type import Memory

    ty = builtin.List(element_type=builtin.Index())
    tl = ty.type_layout
    mem = Memory.__new__(Memory)
    mem.type = ty
    mem.buffer = bytearray(tl.byte_size)
    mem.origins = []
    tl.from_json(mem.buffer, 0, ty.type_to_json(), mem.origins)
    result = tl.to_json(mem.buffer, 0)
    assert result == {
        "tag": "builtin.List",
        "element_type": {"tag": "builtin.Index"},
    }


def test_type_value_memory_roundtrip_nested():
    """Round-trip List<List<F64>> through Memory buffer."""
    from dgen.type import Memory, Type

    inner = builtin.List(element_type=builtin.F64())
    ty = builtin.List(element_type=inner)
    tl = ty.type_layout
    mem = Memory.__new__(Memory)
    mem.type = ty
    mem.buffer = bytearray(tl.byte_size)
    mem.origins = []
    tl.from_json(mem.buffer, 0, ty.type_to_json(), mem.origins)
    result = tl.to_json(mem.buffer, 0)
    reconstructed = Type.type_from_json(result)
    assert reconstructed == ty


def test_type_to_json_pointer():
    """Pointer<F64> serializes with nested type param."""
    ty = builtin.Pointer(pointee=builtin.F64())
    result = ty.type_to_json()
    assert result == {
        "tag": "builtin.Pointer",
        "pointee": {"tag": "builtin.F64"},
    }


def test_type_from_json_pointer_roundtrip():
    """Round-trip Pointer<Index> through JSON."""
    from dgen.type import Type

    ty = builtin.Pointer(pointee=builtin.Index())
    reconstructed = Type.type_from_json(ty.type_to_json())
    assert reconstructed == ty


def test_type_to_json_nil():
    """Nil type (zero-size layout) serializes correctly."""
    ty = builtin.Nil()
    result = ty.type_to_json()
    assert result == {"tag": "builtin.Nil"}


def test_type_type_layout_non_parametric():
    """TypeType(concrete=Index()) layout matches Index().type_layout."""
    tt = builtin.TypeType(concrete=builtin.Index())
    assert tt.__layout__ == builtin.Index().type_layout


def test_type_type_layout_parametric():
    """TypeType(concrete=List(Index())) layout matches the list's type_layout."""
    inner = builtin.List(element_type=builtin.Index())
    tt = builtin.TypeType(concrete=inner)
    assert tt.__layout__ == inner.type_layout


def test_type_layout_size_varies_by_params():
    """Type layout size depends on concrete params (inline design)."""
    # List<Index> and List<List<F64>> have different type_layout sizes
    simple = builtin.List(element_type=builtin.Index())
    nested = builtin.List(element_type=builtin.List(element_type=builtin.F64()))
    assert simple.type_layout.byte_size < nested.type_layout.byte_size


def test_type_value_through_memory():
    """Store Index() as a type value in Memory via TypeType."""
    from dgen.type import Memory, Type

    ty = builtin.Index()
    metatype = builtin.TypeType(concrete=ty)
    mem = Memory.from_json(metatype, ty.type_to_json())
    result = mem.to_json()
    assert result == {"tag": "builtin.Index"}
    reconstructed = Type.type_from_json(result)
    assert reconstructed == ty


def test_type_value_through_memory_parametric():
    """Store List<F64> as a type value in Memory via TypeType."""
    from dgen.type import Memory, Type

    ty = builtin.List(element_type=builtin.F64())
    metatype = builtin.TypeType(concrete=ty)
    mem = Memory.from_json(metatype, ty.type_to_json())
    result = mem.to_json()
    assert result == {
        "tag": "builtin.List",
        "element_type": {"tag": "builtin.F64"},
    }
    reconstructed = Type.type_from_json(result)
    assert reconstructed == ty


def test_type_value_through_memory_nested():
    """Store List<List<Index>> as a type value in Memory via TypeType."""
    from dgen.type import Memory, Type

    inner = builtin.List(element_type=builtin.Index())
    ty = builtin.List(element_type=inner)
    metatype = builtin.TypeType(concrete=ty)
    mem = Memory.from_json(metatype, ty.type_to_json())
    reconstructed = Type.type_from_json(mem.to_json())
    assert reconstructed == ty
