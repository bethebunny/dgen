"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte, Int, Float64 are singletons;
Array, Pointer, FatPointer are parameterized via constructors.

Each layout has a `struct` attribute (a `struct.Struct`) that describes
its binary encoding. The `Memory` class pairs a layout with a buffer
for pack/unpack operations.
"""

from __future__ import annotations

from struct import Struct


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


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    struct = Struct("0s")

    def to_json(self, buf: bytes | bytearray, offset: int) -> None:
        return None


class Byte(Layout):
    struct = Struct("B")

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]


class Int(Layout):
    """64-bit integer (i64)."""

    struct = Struct("q")

    def parse(self, obj: object) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj

    def to_json(self, buf: bytes | bytearray, offset: int) -> int:
        return self.struct.unpack_from(buf, offset)[0]


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


class Array(Layout):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    def __init__(self, element: Layout, count: int) -> None:
        self.element = element
        self.count = count
        self.struct = Struct(f"{count}{element.struct.format}")

    def parse(self, obj: object) -> list[object]:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]


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


class FatPointer(Layout):
    """Pointer + i64 length (16 bytes)."""

    struct = Struct("PQ")

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee


# Module-level singletons for primitives
VOID = Void()
BYTE = Byte()
INT = Int()
FLOAT64 = Float64()
