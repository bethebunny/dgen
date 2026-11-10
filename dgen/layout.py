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


def _bytearray_address(buf: bytearray) -> int:
    """Get the raw ctypes address of a bytearray."""
    ct = (ctypes.c_char * len(buf)).from_buffer(buf)
    return ctypes.addressof(ct)


def _aligned(offset: int, alignment: int) -> int:
    return -(-offset // alignment) * alignment


class Layout:
    """Base for memory layout types."""

    struct: Struct
    register_passable: bool = False
    alignment: int = 8

    @property
    def byte_size(self) -> int:
        return self.struct.size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def to_json(self, buf: bytes | bytearray, offset: int) -> object:
        raise NotImplementedError

    def write(self, image: Image, offset: int) -> None:
        """Make the addresses in the value at offset in image into offsets, copying what they reach."""

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        raise NotImplementedError


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    struct = Struct("<0s")
    register_passable = True
    alignment = 1

    def __str__(self) -> str:
        return "void"

    def to_json(self, buf: bytes | bytearray, offset: int) -> None:
        return None

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        pass


class Byte(Layout):
    struct = Struct("<B")
    register_passable = True
    alignment = 1

    def __str__(self) -> str:
        return "byte"

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Int(Layout):
    """64-bit integer (i64)."""

    struct = Struct("<q")
    register_passable = True

    def __str__(self) -> str:
        return "int64"

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Float64(Layout):
    """64-bit float (f64)."""

    struct = Struct("<d")
    register_passable = True

    def __str__(self) -> str:
        return "float64"

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
        self.alignment = element.alignment
        self.struct = Struct("<" + element.struct.format.lstrip("<") * count)

    def __str__(self) -> str:
        return f"array<{self.element}, {self.count}>"

    def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
        return [
            self.element.to_json(buf, offset + i * self.element.struct.size)
            for i in range(self.count)
        ]

    def write(self, image: Image, offset: int) -> None:
        for i in range(self.count):
            self.element.write(image, offset + i * self.element.byte_size)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        assert isinstance(value, list)
        es = self.element.struct.size
        for i, v in enumerate(value):
            self.element.from_json(buf, offset + i * es, v, origins)


class Pointer(Layout):
    """8-byte pointer to T."""

    struct = Struct("<Q")
    register_passable = True

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee

    def __str__(self) -> str:
        return f"pointer<{self.pointee}>"

    def to_json(self, buf: bytes | bytearray, offset: int) -> object:
        (ptr,) = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        data = bytes((ctypes.c_char * pointee.struct.size).from_address(ptr))
        return pointee.to_json(data, 0)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        pointee = self.pointee
        origin = bytearray(pointee.struct.size)
        origins.append(origin)
        pointee.from_json(origin, 0, value, origins)
        ptr = _bytearray_address(origin)
        self.struct.pack_into(buf, offset, ptr)

    def write(self, image: Image, offset: int) -> None:
        (address,) = self.struct.unpack_from(image.data, offset)
        if address == 0:
            return
        if self.pointee.byte_size == 0:
            raise ValueError(f"a {self} holds a host address, which has no wire form")
        image.link(offset, image.copy(address, self.pointee))


class Span(Layout):
    """Pointer + i64 length (16 bytes)."""

    struct = Struct("<QQ")

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee

    def __str__(self) -> str:
        return f"span<{self.pointee}>"

    def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
        ptr, length = self.struct.unpack_from(buf, offset)
        pointee = self.pointee
        ps = pointee.struct.size
        data = bytes((ctypes.c_char * (length * ps)).from_address(ptr))
        return [pointee.to_json(data, i * ps) for i in range(length)]

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

    def write(self, image: Image, offset: int) -> None:
        address, count = self.struct.unpack_from(image.data, offset)
        if count == 0:
            self.struct.pack_into(image.data, offset, 0, 0)
            return
        image.link(offset, image.copy(address, self.pointee, count=count))


class Record(Layout):
    """Named fields in order, each at the next offset its alignment allows, as in C."""

    def __init__(self, fields: list[tuple[str, Layout]]) -> None:
        self.fields = fields
        self.alignment = max((lay.alignment for _, lay in fields), default=1)
        self.offsets: list[int] = []
        end = 0
        for _, lay in fields:
            self.offsets.append(_aligned(end, lay.alignment))
            end = self.offsets[-1] + lay.byte_size
        self.struct = Struct(f"<{_aligned(end, self.alignment)}s")

    def __str__(self) -> str:
        return f"record<{', '.join(f'{name}: {lay}' for name, lay in self.fields)}>"

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        result: dict[str, object] = {}
        for (name, lay), field_offset in zip(self.fields, self.offsets):
            result[name] = lay.to_json(buf, offset + field_offset)
        return result

    def write(self, image: Image, offset: int) -> None:
        ends = [*self.offsets[1:], self.byte_size]
        for (_, lay), start, end in zip(self.fields, self.offsets, ends):
            # Padding is zeroed so equal values have equal wire bytes.
            image.data[offset + start + lay.byte_size : offset + end] = bytes(
                end - start - lay.byte_size
            )
            lay.write(image, offset + start)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        if isinstance(value, list):
            for (_, lay), field_offset, v in zip(self.fields, self.offsets, value):
                lay.from_json(buf, offset + field_offset, v, origins)
        else:
            assert isinstance(value, dict)
            for (name, lay), field_offset in zip(self.fields, self.offsets):
                lay.from_json(buf, offset + field_offset, value[name], origins)


class String(Span):
    """Span(Byte()) with str ↔ list[int] conversion in to_json/from_json."""

    def __init__(self, pointee: Layout | None = None) -> None:
        super().__init__(pointee or Byte())

    def __str__(self) -> str:
        return "string"

    def to_json(self, buf: bytes | bytearray, offset: int) -> str:
        byte_list = super().to_json(buf, offset)
        return bytes(bytearray(byte_list)).decode("utf-8")

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        if isinstance(value, str):
            value = list(value.encode("utf-8"))
        elif isinstance(value, bytes):
            value = list(value)
        super().from_json(buf, offset, value, origins)


class TypeValue(Layout):
    """An 8-byte address of a type record, which describes a type without the dialect registry.

    The record is TYPE_RECORD:
      - tag: the type's dialect and name, eg. "ndbuffer.Shape"
      - layout: the descriptor of the layout of the type's values, eg. "array<int64, 2>"
      - params_layout: the descriptor of the record of the parameters, eg. "record<rank: int64>"
      - params: the address of that record, or null when it is empty
      - param_types: the type value of each parameter, in order
    """

    struct = Struct("<Q")

    def __str__(self) -> str:
        return "type"

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list
    ) -> None:
        from .type import _type_from_dict

        assert isinstance(value, dict)
        record = _type_from_dict(value).__constant__
        origins.append(record)
        buf[offset : offset + self.byte_size] = record.buffer

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        (address,) = self.struct.unpack_from(buf, offset)
        tag, params_record, params = type_record_parts(address)
        return {"tag": tag} | params_record.to_json(params, 0)

    def write(self, image: Image, offset: int) -> None:
        (address,) = self.struct.unpack_from(image.data, offset)
        _, params_record, _ = type_record_parts(address)
        record = image.copy(address, TYPE_RECORD, walk=False)
        tag_at, layout_at, params_layout_at, params_at, types_at = TYPE_RECORD.offsets
        for at in tag_at, layout_at, params_layout_at:
            STRING.write(image, record + at)
        PARAM_TYPES.write(image, record + types_at)
        (params,) = Pointer.struct.unpack_from(image.data, record + params_at)
        if params != 0:
            image.link(record + params_at, image.copy(params, params_record))
        image.link(offset, record)


STRING = String()
PARAM_TYPES = Span(TypeValue())
TYPE_RECORD = Record(
    [
        ("tag", STRING),
        ("layout", STRING),
        ("params_layout", STRING),
        ("params", Pointer(Void())),
        ("param_types", PARAM_TYPES),
    ]
)


def type_record(
    tag: str,
    layout: Layout,
    params: list[tuple[str, Layout, bytes, bytes]],
    origins: list,
) -> int:
    """The address of a new type record, whose parts origins keeps alive.

    Each parameter is its name, its layout, its value's bytes and its type value's bytes.
    """
    params_record = Record([(name, lay) for name, lay, _, _ in params])
    data = bytearray(params_record.byte_size)
    for (_, _, value, _), at in zip(params, params_record.offsets):
        data[at : at + len(value)] = value
    types = bytearray(b"".join(type_value for *_, type_value in params))
    record = bytearray(TYPE_RECORD.byte_size)
    tag_at, layout_at, params_layout_at, params_at, types_at = TYPE_RECORD.offsets
    STRING.from_json(record, tag_at, tag, origins)
    STRING.from_json(record, layout_at, str(layout), origins)
    STRING.from_json(record, params_layout_at, str(params_record), origins)
    origins += [data, types, record]
    if data:
        Pointer.struct.pack_into(record, params_at, _bytearray_address(data))
    if types:
        PARAM_TYPES.struct.pack_into(
            record, types_at, _bytearray_address(types), len(params)
        )
    return _bytearray_address(record)


def type_record_parts(address: int) -> tuple[str, Record, bytes]:
    """The tag, the parameters' record layout and its bytes, of the type record at address."""
    record = _read(address, TYPE_RECORD.byte_size)
    tag_at, _, params_layout_at, params_at, _ = TYPE_RECORD.offsets
    params_record = parse(STRING.to_json(record, params_layout_at))
    assert isinstance(params_record, Record)
    (params,) = Pointer.struct.unpack_from(record, params_at)
    return (
        STRING.to_json(record, tag_at),
        params_record,
        _read(params, params_record.byte_size),
    )


def param_type_values(address: int) -> list[bytes]:
    """The type value of each parameter of the type record at address."""
    record = _read(address, TYPE_RECORD.byte_size)
    types, count = PARAM_TYPES.struct.unpack_from(record, TYPE_RECORD.offsets[4])
    data = _read(types, count * TypeValue.struct.size)
    return [data[at : at + 8] for at in range(0, len(data), 8)]


class Image:
    """A value in one buffer, with each address replaced by an offset from the buffer's start.

    Invariants:
      - each part starts at a multiple of its layout's alignment
      - relocations lists the offset of every non-null address slot, in the order written
    """

    def __init__(self, start: int) -> None:
        self.data = bytearray(start)
        self.relocations: list[int] = []

    def copy(
        self, address: int, layout: Layout, *, count: int = 1, walk: bool = True
    ) -> int:
        """Append count values of layout from address, and return their offset."""
        start = _aligned(len(self.data), layout.alignment)
        self.data += bytes(start - len(self.data))
        self.data += _read(address, layout.byte_size * count)
        if walk:
            for i in range(count):
                layout.write(self, start + i * layout.byte_size)
        return start

    def link(self, slot: int, target: int) -> None:
        Pointer.struct.pack_into(self.data, slot, target)
        self.relocations.append(slot)


def _read(address: int, size: int) -> bytes:
    return bytes((ctypes.c_char * size).from_address(address))


class _Parser:
    """A recursive descent parser for layout descriptors, eg. "span<record<x: int64>>"."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def take(self, token: str) -> None:
        if not self.text.startswith(token, self.at):
            raise ValueError(f"expected {token!r} at {self.at} in {self.text!r}")
        self.at += len(token)

    def name(self) -> str:
        start = self.at
        while self.at < len(self.text) and (
            self.text[self.at].isalnum() or self.text[self.at] == "_"
        ):
            self.at += 1
        return self.text[start : self.at]

    def layout(self) -> Layout:
        match self.name():
            case "pointer":
                self.take("<")
                result: Layout = Pointer(self.layout())
            case "span":
                self.take("<")
                result = Span(self.layout())
            case "array":
                self.take("<")
                element = self.layout()
                self.take(", ")
                result = Array(element, int(self.name()))
            case "record":
                self.take("<")
                fields: list[tuple[str, Layout]] = []
                while not self.text.startswith(">", self.at):
                    if fields:
                        self.take(", ")
                    name = self.name()
                    self.take(": ")
                    fields.append((name, self.layout()))
                result = Record(fields)
            case leaf:
                if leaf not in LEAVES:
                    raise ValueError(f"unknown layout {leaf!r} in {self.text!r}")
                return LEAVES[leaf]
        self.take(">")
        return result


LEAVES: dict[str, Layout] = {
    str(leaf): leaf
    for leaf in [Void(), Byte(), Int(), Float64(), String(), TypeValue()]
}


def parse(text: str) -> Layout:
    """The layout a descriptor names, ie. the inverse of str(layout)."""
    parser = _Parser(text)
    result = parser.layout()
    if parser.at != len(text):
        raise ValueError(f"unexpected {text[parser.at :]!r} in {text!r}")
    return result
