"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte(), Int(), Float64() are leaf types;
Array, Pointer, FatPointer are parameterized via constructors.

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


class Layout:
    """Base for memory layout types."""

    struct: Struct

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
        raise NotImplementedError

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        raise NotImplementedError


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    struct = Struct("0s")

    def to_json(self, buf: bytes | bytearray, offset: int) -> None:
        return None

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        pass


class Byte(Layout):
    struct = Struct("B")

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Int(Layout):
    """64-bit integer (i64)."""

    struct = Struct("q")

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, int)
        self.struct.pack_into(buf, offset, value)


class Float64(Layout):
    """64-bit float (f64)."""

    struct = Struct("d")

    def to_json(self, buf: bytes | bytearray, offset: int) -> float:
        return self.struct.unpack_from(buf, offset)[0]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, (int, float))
        self.struct.pack_into(buf, offset, float(value))


class Array(Layout):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    def __init__(self, element: Layout, count: int) -> None:
        self.element = element
        self.count = count
        self.struct = Struct(f"{count}{element.struct.format}")

    def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
        return [
            self.element.to_json(buf, offset + i * self.element.struct.size)
            for i in range(self.count)
        ]

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
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

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        pointee = self.pointee
        origin = bytearray(pointee.struct.size)
        origins.append(origin)
        pointee.from_json(origin, 0, value, origins)
        ptr = _bytearray_address(origin)
        self.struct.pack_into(buf, offset, ptr)


class FatPointer(Layout):
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

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
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
        result: dict[str, object] = {}
        for (name, lay), field_offset in zip(self.fields, self._offsets):
            result[name] = lay.to_json(buf, offset + field_offset)
        return result

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, dict)
        for (name, lay), field_offset in zip(self.fields, self._offsets):
            lay.from_json(buf, offset + field_offset, value[name], origins)


class TypeValue(Layout):
    """Pointer to a self-describing type value.

    Type values are Records starting with ("tag", StringLayout()). The tag
    identifies the type class, which determines the full Record layout.
    Fixed size (8-byte pointer) regardless of concrete type.
    """

    struct = Struct("P")

    @staticmethod
    def _is_type_kinded(param_type: type) -> bool:
        """Check if a __params__ type entry represents a type-kinded parameter.

        Type-kinded params are TypeType, Type (base class = "any type"), or
        TypeType subclasses (e.g. Natural trait). Concrete types like Index
        are NOT type-kinded — they inherit from Type but not from TypeType.
        """
        from .type import Type, TypeType

        if param_type is Type or param_type is TypeType:
            return True
        return hasattr(param_type, "__mro__") and TypeType in param_type.__mro__

    def _resolve_layout(self, tag: str) -> Record:
        """Look up a type class by tag and return its TypeType Record layout."""
        from .dialect import Dialect

        dialect_name, type_name = tag.split(".")
        dialect = Dialect.get(dialect_name)
        cls = dialect.types[type_name]
        fields: list[tuple[str, Layout]] = [("tag", String())]
        for param_name, param_type in cls.__params__:
            if self._is_type_kinded(param_type):
                # Type-kinded param: its layout is itself a TypeValue pointer
                fields.append((param_name, TypeValue()))
            elif hasattr(param_type, "__params__") and any(
                self._is_type_kinded(pt) for _, pt in param_type.__params__
            ):
                # List of types (e.g. Tuple's types: List param with TypeType element)
                fields.append((param_name, FatPointer(TypeValue())))
            else:
                # Value-kinded param: use the param type's layout
                fields.append((param_name, param_type().__layout__))
        return Record(fields)

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, dict)
        tag = value["tag"]
        assert isinstance(tag, str)
        pointee_layout = self._resolve_layout(tag)
        origin = bytearray(pointee_layout.byte_size)
        origins.append(origin)
        pointee_layout.from_json(origin, 0, value, origins)
        self.struct.pack_into(buf, offset, _bytearray_address(origin))

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        (ptr,) = self.struct.unpack_from(buf, offset)
        # Read just the tag first (String = FatPointer<Byte> at offset 0)
        tag_layout = String()
        tag_data = bytes((ctypes.c_char * tag_layout.struct.size).from_address(ptr))
        tag = tag_layout.to_json(tag_data, 0)
        assert isinstance(tag, str)
        # Now resolve full layout and read entire buffer
        pointee_layout = self._resolve_layout(tag)
        data = bytes((ctypes.c_char * pointee_layout.byte_size).from_address(ptr))
        return pointee_layout.to_json(data, 0)


class String(FatPointer):
    """FatPointer(Byte()) with str ↔ list[int] conversion in to_json/from_json."""

    def __init__(self, pointee: Layout | None = None) -> None:
        super().__init__(pointee or Byte())

    def to_json(self, buf: bytes | bytearray, offset: int) -> str:
        byte_list = super().to_json(buf, offset)
        return bytes(bytearray(byte_list)).decode("utf-8")

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        if isinstance(value, str):
            value = list(value.encode("utf-8"))
        elif isinstance(value, bytes):
            value = list(value)
        super().from_json(buf, offset, value, origins)
