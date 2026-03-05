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
