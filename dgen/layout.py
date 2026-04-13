"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte(), Int(), Float64() are leaf types;
Array, Pointer, Span are parameterized via constructors.

Each layout has a `struct` attribute (a `struct.Struct`) that describes
its binary encoding. The `Memory` class pairs a layout with a buffer
for pack/unpack operations.
"""

from __future__ import annotations

import ctypes
from functools import cached_property
from itertools import accumulate
from struct import Struct
from typing import TYPE_CHECKING

from .dialect import Dialect

if TYPE_CHECKING:
    from .type import Type


def align_up(size: int, alignment: int) -> int:
    """Round ``size`` up to the next multiple of ``alignment``."""
    return (size + alignment - 1) // alignment


def _bytearray_address(buf: bytearray) -> int:
    """Get the raw ctypes address of a bytearray."""
    ct = (ctypes.c_char * len(buf)).from_buffer(buf)
    return ctypes.addressof(ct)


# Aggregates ≤ 16 bytes fit in two 64-bit registers under x86_64 SysV
# and can be passed/returned by value. Above that, they have to be
# passed in memory.
_REGISTER_SIZE_LIMIT = 16


class Layout:
    """Base for memory layout types."""

    struct: Struct

    @property
    def byte_size(self) -> int:
        return self.struct.size

    @property
    def register_passable(self) -> bool:
        return self.byte_size <= _REGISTER_SIZE_LIMIT

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return type(self) is type(other) and self.struct.format == other.struct.format

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.struct.format))

    def to_json(self, buf: bytes | bytearray, offset: int) -> object:
        """Read this layout's bytes as pure JSON (dicts/lists/scalars only).

        This is the wire-format reader: anything emitted here must round-trip
        through ``json.dumps`` unchanged. Type values are returned as
        ``{"tag": ..., "params": ...}`` descriptor dicts.
        """
        raise NotImplementedError

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        raise NotImplementedError

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> object:
        """Read this layout's bytes as rich native Python objects.

        ("native" as in "Python-native", not as in dgen ``Value``.) Same as
        ``to_json`` for most layouts, but type values are returned as
        first-class ``Type`` instances and container layouts (Record, Span,
        Array) recurse via ``to_native_value`` so embedded types are
        preserved through nesting. This is the form the IR formatter and
        the use-def walker want — raw JSON would force them to sniff for
        type-shaped dicts to recover types.
        """
        return self.to_json(buf, offset)


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    struct = Struct("0s")

    def to_json(self, buf: bytes | bytearray, offset: int) -> None:
        return None

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        pass


class Byte(Layout):
    struct = Struct("B")
    register_passable = True

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Int(Layout):
    """64-bit integer (i64)."""

    struct = Struct("q")

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Float64(Layout):
    """64-bit float (f64)."""

    struct = Struct("d")

    def to_json(self, buf: bytes | bytearray, offset: int) -> float:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, (int, float))
        self.struct.pack_into(buf, offset, float(value))


class Array(Layout):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    def __init__(self, element: Layout, count: int) -> None:
        self.element = element
        self.count = count

    @cached_property
    def struct(self) -> Struct:
        return Struct(f"{self.count}{self.element.struct.format}")

    def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
        return [
            self.element.to_json(buf, offset + i * self.element.struct.size)
            for i in range(self.count)
        ]

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> list[object]:
        return [
            self.element.to_native_value(buf, offset + i * self.element.struct.size)
            for i in range(self.count)
        ]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, list)
        es = self.element.struct.size
        for i, v in enumerate(value):
            self.element.from_json(buf, offset + i * es, v, origins)


class Pointer(Layout):
    """8-byte pointer to T."""

    struct = Struct("P")

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee

    def to_json(self, buf: bytes | bytearray, offset: int) -> object:
        (ptr,) = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        data = bytes((ctypes.c_char * pointee.struct.size).from_address(ptr))
        return pointee.to_json(data, 0)

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> object:
        (ptr,) = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        data = bytes((ctypes.c_char * pointee.struct.size).from_address(ptr))
        return pointee.to_native_value(data, 0)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        pointee = self.pointee
        origin = bytearray(pointee.struct.size)
        origins.append(origin)
        pointee.from_json(origin, 0, value, origins)
        ptr = _bytearray_address(origin)
        self.struct.pack_into(buf, offset, ptr)


class Span(Layout):
    """Pointer + i64 length (16 bytes)."""

    struct = Struct("PQ")

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee

    def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
        ptr, length = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        ps = pointee.struct.size
        data = bytes((ctypes.c_char * (length * ps)).from_address(ptr))
        return [pointee.to_json(data, i * ps) for i in range(length)]

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> list[object]:
        ptr, length = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        ps = pointee.struct.size
        data = bytes((ctypes.c_char * (length * ps)).from_address(ptr))
        return [pointee.to_native_value(data, i * ps) for i in range(length)]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, list)
        pointee = self.pointee
        ps = pointee.struct.size
        origin = bytearray(ps * len(value))
        origins.append(origin)
        for i, v in enumerate(value):
            pointee.from_json(origin, i * ps, v, origins)
        ptr = _bytearray_address(origin)
        self.struct.pack_into(buf, offset, ptr, len(value))


class Record(Layout):
    """Fixed struct of named fields, laid out sequentially."""

    def __init__(self, fields: list[tuple[str, Layout]]) -> None:
        self.fields = fields

    @cached_property
    def _offsets(self) -> list[int]:
        sizes = (layout.byte_size for _, layout in self.fields)
        return list(accumulate(sizes, initial=0))[:-1]

    @cached_property
    def struct(self) -> Struct:
        format = (
            "".join(layout.struct.format for _, layout in self.fields)
            if self.fields
            else "0s"
        )
        return Struct(format)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        if len(self.fields) != len(other.fields):
            return False
        return all(
            n1 == n2 and l1 == l2
            for (n1, l1), (n2, l2) in zip(self.fields, other.fields)
        )

    def __hash__(self) -> int:
        return hash(tuple((name, type(lay).__name__) for name, lay in self.fields))

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        return {
            name: layout.to_json(buf, offset + field_offset)
            for (name, layout), field_offset in zip(self.fields, self._offsets)
        }

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        return {
            name: layout.to_native_value(buf, offset + field_offset)
            for (name, layout), field_offset in zip(self.fields, self._offsets)
        }

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        if isinstance(value, list):
            for (_, lay), field_offset, v in zip(self.fields, self._offsets, value):
                lay.from_json(buf, offset + field_offset, v, origins)
        else:
            assert isinstance(value, dict)
            for (name, lay), field_offset in zip(self.fields, self._offsets):
                lay.from_json(buf, offset + field_offset, value[name], origins)


def _descriptor_layout(type_instance: Type) -> Record:
    """Build the Record layout for a concrete type's self-describing descriptor."""
    return _descriptor_layout_for_cls(type(type_instance))


def _some_inner_layout(type_instance: Type) -> Record:
    """Layout for the heap block behind a Some pointer: {TypeValue, value}."""
    return Record([("type", TypeValue()), ("value", type_instance.__layout__)])


def _descriptor_layout_for_cls(cls: type[Type]) -> Record:
    """Build the descriptor Record layout from a type class (no instance needed)."""
    return Record(
        [
            ("tag", String()),
            ("params", Record([(name, Some()) for name, _ in cls.__params__])),
        ]
    )


class TypeValue(Layout):
    """Pointer to a self-describing type value.

    Each parameter carries its own type descriptor, so the record is fully
    self-describing.  Fixed size (8-byte pointer) regardless of concrete type.
    """

    struct = Struct("P")
    register_passable = True

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        from .type import Type

        # Raw pointer integer — from a register-passable JIT return or a
        # ctypes callback. Pack the pointer directly.
        if isinstance(value, int):
            self.struct.pack_into(buf, offset, value)
            return
        # Accept either a self-describing dict or a parsed Type instance —
        # the parser produces the latter when type-ASM sugar appears in
        # value position (e.g. inside a literal dict).
        if isinstance(value, Type):
            value = value.to_json()
        assert isinstance(value, dict)
        layout = _descriptor_layout(Type.from_json(value))
        origin = bytearray(layout.byte_size)
        origins.append(origin)
        layout.from_json(origin, 0, value, origins)
        self.struct.pack_into(buf, offset, _bytearray_address(origin))

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> object:
        from .type import Type

        # Rich form: rehydrate the type descriptor into a real Type instance,
        # so callers can use ``format_asm`` / ``.dialect`` directly without
        # sniffing JSON shape.
        return Type.from_json(self.to_json(buf, offset))

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        (ptr,) = self.struct.unpack_from(buf, offset)
        # Read the tag to look up param names.
        tag_layout = String()
        tag_bytes = bytes((ctypes.c_char * tag_layout.byte_size).from_address(ptr))
        tag = tag_layout.to_json(tag_bytes, 0)
        assert isinstance(tag, str)
        dialect_name, type_name = tag.split(".")
        cls = Dialect.get(dialect_name).types[type_name]
        # Each param is a Some (pointer to {TypeValue, value_inline}).
        layout = _descriptor_layout_for_cls(cls)
        data = bytes((ctypes.c_char * layout.byte_size).from_address(ptr))
        result = layout.to_json(data, 0)
        assert isinstance(result, dict)
        return result


class Some(Layout):
    """Runtime-typed value: pointer to a heap block containing a witness
    type descriptor followed by the value bytes inline.

    Binary format: 8 bytes = single pointer. Behind the pointer:
    ``{TypeValue, value_bytes}``. The TypeValue is read first to
    determine the value's layout, then the value is read inline.

    The ``bound`` is a compile-time annotation (e.g. a trait); it
    doesn't affect the binary layout.
    """

    struct = Struct("P")
    register_passable = True

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        from .type import Type

        assert isinstance(value, dict)
        type_json = value["type"]
        val_json = value["value"]
        type_inst = (
            Type.from_json(type_json) if not isinstance(type_json, Type) else type_json
        )
        # Build the inner layout: {TypeValue, value_inline}.
        inner = _some_inner_layout(type_inst)
        origin = bytearray(inner.byte_size)
        origins.append(origin)
        inner.from_json(origin, 0, {"type": type_json, "value": val_json}, origins)
        self.struct.pack_into(buf, offset, _bytearray_address(origin))

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        from .type import Type

        (ptr,) = self.struct.unpack_from(buf, offset)
        # Read the TypeValue first to determine the value layout.
        tv = TypeValue()
        tv_bytes = bytes((ctypes.c_char * tv.byte_size).from_address(ptr))
        type_json = tv.to_json(tv_bytes, 0)
        type_inst = Type.from_json(type_json)
        # Read the full inner record.
        inner = _some_inner_layout(type_inst)
        data = bytes((ctypes.c_char * inner.byte_size).from_address(ptr))
        result = inner.to_json(data, 0)
        assert isinstance(result, dict)
        return result

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> object:
        from .type import Type

        (ptr,) = self.struct.unpack_from(buf, offset)
        tv = TypeValue()
        tv_bytes = bytes((ctypes.c_char * tv.byte_size).from_address(ptr))
        type_inst = tv.to_native_value(tv_bytes, 0)
        assert isinstance(type_inst, Type)
        inner = _some_inner_layout(type_inst)
        data = bytes((ctypes.c_char * inner.byte_size).from_address(ptr))
        inner_json = inner.to_json(data, 0)
        assert isinstance(inner_json, dict)
        return {"type": type_inst, "value": inner_json["value"]}


class String(Span):
    """Span(Byte()) with str ↔ list[int] conversion in to_json/from_json."""

    def __init__(self, pointee: Layout | None = None) -> None:
        super().__init__(pointee or Byte())

    def to_json(self, buf: bytes | bytearray, offset: int) -> str:
        byte_list = super().to_json(buf, offset)
        return bytes(bytearray(byte_list)).decode("utf-8")

    def to_native_value(self, buf: bytes | bytearray, offset: int) -> str:
        # A Python ``str`` is already rich and JSON-compat — don't fall through
        # to ``Span.to_native_value``, which would walk byte-by-byte and lose the
        # str decoding from ``to_json``.
        return self.to_json(buf, offset)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        if isinstance(value, str):
            value = list(value.encode("utf-8"))
        elif isinstance(value, bytes):
            value = list(value)
        super().from_json(buf, offset, value, origins)
