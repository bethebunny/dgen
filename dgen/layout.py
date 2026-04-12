"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte(), Int(), Float64() are leaf types;
Array, Pointer, Span are parameterized via constructors.

Each layout has a `struct` attribute (a `struct.Struct`) that describes
its binary encoding. The `Memory` class pairs a layout with a buffer
for pack/unpack operations.
"""

from __future__ import annotations

import ctypes
from struct import Struct
from typing import TYPE_CHECKING

from .dialect import Dialect

if TYPE_CHECKING:
    from .type import Type


def _bytearray_address(buf: bytearray) -> int:
    """Get the raw ctypes address of a bytearray."""
    ct = (ctypes.c_char * len(buf)).from_buffer(buf)
    return ctypes.addressof(ct)


# Aggregates ≤ 16 bytes fit in two 64-bit registers under x86_64 SysV
# and can be passed/returned by value. Above that, they have to be
# passed in memory.
_AGGREGATE_REGISTER_LIMIT = 16


class Layout:
    """Base for memory layout types."""

    struct: Struct
    register_passable: bool = False

    @property
    def byte_size(self) -> int:
        return self.struct.size

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
    register_passable = True

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
    register_passable = True

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
    register_passable = True

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
        self.struct = Struct(f"{count}{element.struct.format}")
        self.register_passable = self.byte_size <= _AGGREGATE_REGISTER_LIMIT

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
    register_passable = True

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
        self._offsets: list[int] = []
        offset = 0
        for _, lay in fields:
            self._offsets.append(offset)
            offset += lay.byte_size
        self._total_size = offset
        # Use a raw byte format for the total size
        self.struct = Struct(f"{offset}s") if offset > 0 else Struct("0s")
        self.register_passable = self.byte_size <= _AGGREGATE_REGISTER_LIMIT

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
    """Build the Record layout for a concrete type's self-describing descriptor.

    The layout mirrors the JSON structure produced by ``Type.to_json()``::

        Record([
            ("tag", String()),
            ("params", Record([
                (name, Record([("type", TypeValue()), ("value", <layout>)])),
                ...
            ]))
        ])
    """
    # fmt: off
    return Record([
        ("tag", String()),
        ("params", Record([
            (name, Record([("type", TypeValue()), ("value", param.type.__layout__)]))
            for name, param in type_instance.parameters
        ])),
    ])
    # fmt: on


class TypeValue(Layout):
    """Pointer to a self-describing type value.

    Each parameter carries its own type descriptor, so the record is fully
    self-describing.  Fixed size (8-byte pointer) regardless of concrete type.
    """

    struct = Struct("P")

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        from .type import Type

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
        from .type import Type

        (ptr,) = self.struct.unpack_from(buf, offset)
        # Read the tag to look up param names, then read each param's type
        # descriptor to determine its value layout.
        tag_layout = String()
        tag_bytes = bytes((ctypes.c_char * tag_layout.byte_size).from_address(ptr))
        tag = tag_layout.to_json(tag_bytes, 0)
        assert isinstance(tag, str)
        dialect_name, type_name = tag.split(".")
        cls = Dialect.get(dialect_name).types[type_name]
        # Build the full Record layout by reading type descriptors sequentially.
        read_offset = tag_layout.byte_size
        param_fields: list[tuple[str, Layout]] = []
        for param_name, _param_cls in cls.__params__:
            tv = TypeValue()
            tv_bytes = bytes(
                (ctypes.c_char * tv.byte_size).from_address(ptr + read_offset)
            )
            param_type = Type.from_json(tv.to_json(tv_bytes, 0))
            value_layout = param_type.__layout__
            param_fields.append(
                (param_name, Record([("type", tv), ("value", value_layout)]))
            )
            read_offset += tv.byte_size + value_layout.byte_size
        layout = Record([("tag", tag_layout), ("params", Record(param_fields))])
        data = bytes((ctypes.c_char * layout.byte_size).from_address(ptr))
        result = layout.to_json(data, 0)
        assert isinstance(result, dict)
        return result


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
