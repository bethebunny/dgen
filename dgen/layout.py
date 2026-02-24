"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte, Int, Float64 are singletons;
Array, Pointer, FatPointer are parameterized via constructors.

Each layout has a `struct` property returning a `struct.Struct` that
describes its binary encoding. The `Memory` class pairs a layout with
a buffer for pack/unpack operations.
"""

from __future__ import annotations

import struct as _struct
from functools import cached_property


class Layout:
    """Base for memory layout types."""

    _format: str | None = None

    @cached_property
    def struct(self) -> _struct.Struct:
        return _struct.Struct(f"@{self._format}")

    def byte_size(self) -> int:
        return self.struct.size

    def parse(self, obj: object) -> object:
        raise NotImplementedError


class Void(Layout):
    """Zero-size layout for types with no runtime representation."""

    _format = "0s"


class Byte(Layout):
    _format = "B"


class Int(Layout):
    """64-bit integer (i64)."""

    _format = "q"

    def parse(self, obj: object) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj


class Float64(Layout):
    """64-bit float (f64)."""

    _format = "d"

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
        self._format = f"{count}{element._format}"

    def parse(self, obj: object) -> list[object]:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]


class Pointer(Layout):
    """8-byte pointer to T."""

    _format = "P"

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee


class FatPointer(Layout):
    """Pointer + i64 length (16 bytes)."""

    _format = "PQ"

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee


# Module-level singletons for primitives
VOID = Void()
BYTE = Byte()
INT = Int()
FLOAT64 = Float64()
