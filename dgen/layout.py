"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte, Int, Float64 are singletons;
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

    def parse(self, obj: object) -> object:
        raise NotImplementedError

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

    def parse(self, obj: object) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj

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

    def parse(self, obj: object) -> float:
        assert isinstance(obj, (int, float)), (
            f"expected number, got {type(obj).__name__}"
        )
        return float(obj)

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

    def parse(self, obj: object) -> list[object]:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]

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


class Bytes(Layout):
    """N raw bytes, stored as a single bytes object."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.struct = Struct(f"{count}s")

    def parse(self, obj: object) -> bytes:
        assert isinstance(obj, bytes), f"expected bytes, got {type(obj).__name__}"
        assert len(obj) == self.count, f"expected {self.count} bytes, got {len(obj)}"
        return obj


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
        if isinstance(value, str):
            value = list(value.encode("utf-8"))
        elif isinstance(value, bytes):
            value = list(value)
        assert isinstance(value, list)
        pointee = self.pointee
        ps = pointee.struct.size
        origin = bytearray(ps * len(value))
        origins.append(origin)
        for i, v in enumerate(value):
            pointee.from_json(origin, i * ps, v, origins)
        ptr = _bytearray_address(origin)
        self.struct.pack_into(buf, offset, ptr, len(value))


# Module-level singletons for primitives
VOID = Void()
BYTE = Byte()
INT = Int()
FLOAT64 = Float64()
