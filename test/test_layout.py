"""Tests for memory layout types."""

import ctypes
from dataclasses import dataclass
from typing import ClassVar


from dgen import Dialect, asm, layout
from dgen.type import format_value as type_asm
from dgen.asm.parser import parse
from dgen.dialects import algebra, builtin, number
from dgen.layout import Array, Byte, Float64, Pointer, Span, TypeValue
from dgen.testing import strip_prefix
from dgen.memory import Memory
from dgen.type import Constant, Fields, Type, TypeType, Value, constant


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
    assert constant(c) == "hello"


def test_string_type_asm_no_params():
    """String type formats as 'String' with no angle brackets."""
    s = builtin.String()
    assert type_asm(s) == "String"


def test_span_fatpointer_layout():
    """Span type uses Span(pointee.__layout__) — 16 bytes, not inline."""
    list_type = builtin.Span(pointee=builtin.Index())
    ly = list_type.__layout__
    assert isinstance(ly, Span)
    assert ly.byte_size == 16
    assert isinstance(ly.pointee, layout.Int)


def test_span_constant_fatpointer():
    """Span constant: data in origin, pointer in Memory."""
    list_type = builtin.Span(pointee=builtin.Index())
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


def test_span_type_asm_one_param():
    """Span type formats as 'Span<index>' — just pointee, no count."""
    list_type = builtin.Span(pointee=builtin.Index())
    assert type_asm(list_type) == "Span<index.Index>"


def test_int_to_json():
    mem = Memory.from_value(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_float_to_json():
    mem = Memory.from_value(number.Float64(), 3.14)
    assert mem.to_json() == 3.14


def test_byte_to_json():
    b = Byte()
    buf = bytearray(1)
    b.struct.pack_into(buf, 0, 65)
    assert b.to_json(buf, 0) == 65


def test_f64type_layout():
    assert number.Float64().__layout__.byte_size == 8


def test_index_type_layout():
    assert builtin.Index().__layout__.byte_size == 8


def test_fatpointer_to_json():
    ty = builtin.Span(pointee=builtin.Index())
    mem = Memory.from_value(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_string_to_json():
    mem = Memory.from_value(builtin.String(), "hello")
    assert mem.to_json() == "hello"


def test_int_from_json_roundtrip():
    mem = Memory.from_json(builtin.Index(), 42)
    assert mem.to_json() == 42


def test_span_from_json_roundtrip():
    ty = builtin.Span(pointee=builtin.Index())
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


def test_nested_span_from_json_roundtrip():
    inner = builtin.Span(pointee=builtin.Index())
    outer = builtin.Span(pointee=inner)
    mem = Memory.from_json(outer, [[1, 2], [3, 4, 5]])
    assert mem.to_json() == [[1, 2], [3, 4, 5]]


def test_type_layout_non_parametric():
    """Non-parametric type layout is a fixed-size TypeValue pointer (8 bytes)."""
    ty = builtin.Index()
    tl = ty.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_layout_parametric_value_param():
    """Parametric type layout is a fixed-size TypeValue pointer (8 bytes)."""
    list_type = builtin.Span(pointee=builtin.Index())
    tl = list_type.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_layout_parametric_type_param_nested():
    """Nested parametric type layout is still a fixed-size TypeValue pointer."""
    inner = builtin.Span(pointee=number.Float64())
    outer = builtin.Span(pointee=inner)
    tl = outer.type.__layout__
    assert isinstance(tl, TypeValue)
    assert tl.byte_size == 8


def test_type_value_memory_non_parametric():
    """Pack and unpack Index() as a type value through Memory."""
    metatype = TypeType()
    data = {"tag": "index.Index", "params": {}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_parametric():
    """Pack and unpack Span<index.Index> as a type value through Memory."""
    metatype = TypeType()
    data = builtin.Span(pointee=builtin.Index()).to_json()
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_pointer():
    """Pack and unpack Pointer<number.Float64> as a type value through Memory."""
    metatype = TypeType()
    data = builtin.Pointer(pointee=number.Float64()).to_json()
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_nil():
    """Pack and unpack Nil as a type value through Memory."""
    metatype = TypeType()
    data = {"tag": "builtin.Nil", "params": {}}
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_memory_nested():
    """Pack and unpack Span<Span<number.Float64>> as a type value through Memory."""
    metatype = TypeType()
    data = builtin.Span(pointee=builtin.Span(pointee=number.Float64())).to_json()
    mem = Memory.from_json(metatype, data)
    assert mem.to_json() == data


def test_type_value_self_describing_format():
    """Type.to_json produces self-describing format with param types."""
    arr = builtin.Array(element_type=builtin.Index(), n=builtin.Index().constant(4))
    assert arr.to_json() == {
        "tag": "builtin.Array",
        "params": {
            "element_type": {
                "type": {"tag": "builtin.Type", "params": {}},
                "value": {"tag": "index.Index", "params": {}},
            },
            "n": {
                "type": {"tag": "index.Index", "params": {}},
                "value": 4,
            },
        },
    }


def test_type_value_from_json_roundtrip():
    """Type.from_json reconstructs a Type from its self-describing dict."""
    from dgen.type import Type

    original = builtin.Array(element_type=number.Float64(), n=builtin.Index().constant(8))
    data = original.to_json()
    reconstructed = Type.from_json(data)
    assert type(reconstructed).__name__ == "Array"
    assert reconstructed.to_json() == data


def test_type_value_dependent_param_roundtrip():
    """Types with dependent params (Shape) round-trip through TypeValue."""
    from dgen.type import Type
    from dgen.dialects.ndbuffer import NDBuffer, Shape

    rank = builtin.Index().constant(3)
    shape = Shape(rank=rank)
    ty = NDBuffer(shape=shape, dtype=number.Float64())
    data = ty.to_json()
    # Shape param's type is TypeType (shapes are types, types have metatype)
    shape_param = data["params"]["shape"]
    assert shape_param["type"]["tag"] == "builtin.Type"
    # The value is the Shape descriptor with its own params
    assert shape_param["value"]["tag"] == "ndbuffer.Shape"

    # Full round-trip through Memory
    metatype = TypeType()
    mem = Memory.from_json(metatype, data)
    reconstructed_data = mem.to_json()
    assert reconstructed_data == data
    reconstructed = Type.from_json(reconstructed_data)
    assert type(reconstructed).__name__ == "NDBuffer"


def test_type_value_non_parametric_format():
    """Non-parametric types have an empty 'params' dict."""
    assert builtin.Index().to_json() == {"tag": "index.Index", "params": {}}


def test_type_type_layout_non_parametric():
    """TypeType() layout matches Index().type.__layout__."""
    tt = TypeType()
    assert tt.__layout__ == builtin.Index().type.__layout__


def test_type_type_layout_parametric():
    """TypeType() layout matches the list's type.__layout__."""
    inner = builtin.Span(pointee=builtin.Index())
    tt = TypeType()
    assert tt.__layout__ == inner.type.__layout__


def test_type_layout_size_fixed():
    """Type layout size is fixed (8-byte pointer) regardless of params."""
    simple = builtin.Span(pointee=builtin.Index())
    nested = builtin.Span(pointee=builtin.Span(pointee=number.Float64()))
    assert simple.type.__layout__.byte_size == 8
    assert nested.type.__layout__.byte_size == 8
    assert simple.type.__layout__.byte_size == nested.type.__layout__.byte_size


# ===----------------------------------------------------------------------=== #
# Type-is-Value tests
# ===----------------------------------------------------------------------=== #


def test_type_is_value():
    """Every Type instance is a Value."""
    ty = number.Float64()
    assert isinstance(ty, Value)
    assert isinstance(ty.type, TypeType)


def test_type_constant_non_parametric():
    """Non-parametric type's __constant__ serializes to just a tag."""
    ty = number.Float64()
    assert ty.__constant__.to_json() == {"tag": "number.Float64", "params": {}}


def test_type_constant_parametric():
    """Parametric type's __constant__ includes self-describing param values."""
    ty = builtin.Span(pointee=builtin.Index())
    data = ty.__constant__.to_json()
    expected = ty.to_json()
    assert data == expected
    # Verify the self-describing structure
    assert data["tag"] == "builtin.Span"
    assert "params" in data
    pointee = data["params"]["pointee"]
    assert pointee["type"] == {"tag": "builtin.Type", "params": {}}
    assert pointee["value"] == {"tag": "index.Index", "params": {}}


def test_type_params_are_bare_types():
    """Type-kinded params are bare Type instances, not Constant[TypeType]."""
    ty = builtin.Span(pointee=builtin.Index())
    assert isinstance(ty.pointee, builtin.Index)
    assert isinstance(ty.pointee, Type)
    assert not isinstance(ty.pointee, Constant)


# ===----------------------------------------------------------------------=== #
# Pointer<Array> tests
# ===----------------------------------------------------------------------=== #


def test_pointer_array_size():
    """Pointer(Array(...)) is always pointer-sized (8 bytes) regardless of count."""
    assert Pointer(Array(Float64(), 4)).byte_size == 8
    assert Pointer(Array(Byte(), 100)).byte_size == 8
    assert Pointer(Array(layout.Int(), 1)).byte_size == 8


def test_pointer_array_type_layout():
    """Pointer<Array<number.Float64, 4>> type produces a Pointer(Array) layout."""
    pa = builtin.Pointer(
        pointee=builtin.Array(
            element_type=number.Float64(), n=builtin.Index().constant(4)
        )
    )
    ly = pa.__layout__
    assert isinstance(ly, Pointer)
    assert ly.byte_size == 8
    assert isinstance(ly.pointee, Array)
    assert ly.pointee.count == 4


def test_pointer_array_roundtrip():
    """Pointer<Array> constant round-trips through Memory."""
    pa = builtin.Pointer(
        pointee=builtin.Array(
            element_type=builtin.Index(), n=builtin.Index().constant(3)
        )
    )
    mem = Memory.from_json(pa, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]


def test_pointer_array_type_asm():
    """Pointer<Array<number.Float64, 4>> formats correctly."""
    pa = builtin.Pointer(
        pointee=builtin.Array(
            element_type=number.Float64(), n=builtin.Index().constant(4)
        )
    )
    assert type_asm(pa) == "Pointer<Array<number.Float64, index.Index(4)>>"


def test_parse_type_with_pointer_array_param():
    """Parsing a type whose param is Pointer<Array<...>> with an explicit typed literal.

    With the Type<params>(literal) syntax, Pointer<Array<number.Float64, index.Index(3)>>([10, 20, 30])
    is parsed without any inference — the type is fully specified.
    """
    test_dialect = Dialect("_test_pa")

    @test_dialect.type("Wrapper")
    @dataclass(frozen=True, eq=False)
    class Wrapper(Type):
        data: Value[builtin.Pointer]
        __params__: ClassVar[Fields] = (("data", builtin.Pointer),)
        __layout__ = layout.Void()

    ir = strip_prefix("""
        | import function
        | import index
        | import number
        | import _test_pa
        |
        | %f : function.Function<[], _test_pa.Wrapper<Pointer<Array<number.Float64, index.Index(3)>>([10, 20, 30])>> = function.function<_test_pa.Wrapper<Pointer<Array<number.Float64, index.Index(3)>>([10, 20, 30])>>() body():
        |     %_ : Nil = ()
    """)
    parse(ir)


# ===----------------------------------------------------------------------=== #
# Existential (Some / Any) layout
# ===----------------------------------------------------------------------=== #


def test_some_layout_record_shape():
    """Some<bound> derives a 16-byte Record(existential: TypeValue, value: Pointer)."""
    ly = builtin.Some(bound=builtin.TypeTag()).__layout__
    assert isinstance(ly, layout.Record)
    assert ly.byte_size == 16
    fields = dict(ly.fields)
    assert isinstance(fields["existential"], TypeValue)
    assert isinstance(fields["value"], layout.Pointer)


def test_any_aliases_some_type():
    """Any has the same 16-byte Record layout as Some<Type>."""
    any_ly = builtin.Any().__layout__
    some_ly = builtin.Some(bound=builtin.TypeTag()).__layout__
    assert isinstance(any_ly, layout.Record)
    assert any_ly.byte_size == some_ly.byte_size == 16


def test_some_type_asm_format():
    """Some<bound> and Any format with no dialect prefix (builtin)."""
    assert type_asm(builtin.Some(bound=builtin.TypeTag())) == "Some<TypeTag>"
    assert type_asm(builtin.Any()) == "Any"


def test_some_with_trait_bound():
    """A trait can serve as the bound for Some<Trait>."""
    ty = builtin.Some(bound=algebra.AddMagma())
    assert isinstance(ty.__layout__, layout.Record)
    assert ty.__layout__.byte_size == 16


def test_existential_any_example_roundtrip():
    """The existential_any example uses sugared type ASM and round-trips."""
    text = strip_prefix("""
        | import function
        | import index
        | import number
        |
        | %main : function.Function<[], Any> = function.function<Any>() body():
        |     %r : Any = {"existential": number.SignedInteger<index.Index(32)>, "value": ()}
    """)
    value = parse(text)
    formatted = asm.format(value)
    # Sugared type-ASM, not raw {tag, params} dict.
    assert "number.SignedInteger<index.Index(32)>" in formatted
    assert '"tag":' not in formatted
    # And it round-trips through the parser cleanly.
    parse(formatted)
