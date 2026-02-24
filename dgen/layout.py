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

    def byte_size(self) -> int:
        return self.struct.size

    def parse(self, obj: object) -> object:
        raise NotImplementedError


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    struct = Struct("0s")


class Byte(Layout):
    struct = Struct("B")


class Int(Layout):
    """64-bit integer (i64)."""

    struct = Struct("q")

    def parse(self, obj: object) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj


class Float64(Layout):
    """64-bit float (f64)."""

    struct = Struct("d")

    def parse(self, obj: object) -> float:
        assert isinstance(obj, (int, float)), (
            f"expected number, got {type(obj).__name__}"
        )
        return float(obj)


class Array(Layout):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    def __init__(self, element: Layout, count: int) -> None:
        self.element = element
        self.count = count
        self.struct = Struct(f"{count}{element.struct.format}")

    def parse(self, obj: object) -> list[object]:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]


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
