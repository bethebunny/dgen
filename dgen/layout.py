"""Memory layout types for language-agnostic type descriptions.

Layouts are value-level descriptors: Byte, Int, Float64 are singletons;
Array, Pointer, FatPointer are parameterized via constructors.

Each layout has a `struct` property returning a `struct.Struct` that
describes its binary encoding. The `Memory` class pairs a layout with
a buffer for pack/unpack operations.
"""

from __future__ import annotations

import ctypes as _ctypes
import struct as _struct
from functools import cached_property


class Layout:
    """Base for memory layout types."""

    _format = None

    @cached_property
    def struct(self) -> _struct.Struct:
        return _struct.Struct(f'@{self._format}')

    def byte_size(self) -> int:
        return self.struct.size

    def parse(self, obj) -> object:
        raise NotImplementedError

    def prepare_arg(self, value):
        """Convert Python value to (ctypes_arg, refs_to_keep_alive)."""
        return value, []


class Byte(Layout):
    _format = 'B'


class Int(Layout):
    """64-bit integer (i64)."""

    _format = 'q'

    def parse(self, obj) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj


class Float64(Layout):
    """64-bit float (f64)."""

    _format = 'd'

    def parse(self, obj) -> float:
        assert isinstance(obj, (int, float)), (
            f"expected number, got {type(obj).__name__}"
        )
        return float(obj)


class Array(Layout):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    def __init__(self, element: Layout, count: int):
        self.element = element
        self.count = count
        self._format = f'{count}{element._format}'

    def parse(self, obj) -> list:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]

    def prepare_arg(self, value):
        mem = Memory(self)
        mem.pack(*value)
        return mem.ptr, [mem]


class Pointer(Layout):
    """8-byte pointer to T."""

    _format = 'P'

    def __init__(self, pointee: Layout):
        self.pointee = pointee


class FatPointer(Layout):
    """Pointer + i64 length (16 bytes)."""

    _format = 'PQ'

    def __init__(self, pointee: Layout):
        self.pointee = pointee


# ---------------------------------------------------------------------------
# Memory: typed buffer for layout ABI
# ---------------------------------------------------------------------------


class Memory:
    """Typed memory buffer — the ABI for a layout."""

    __slots__ = ('layout', 'buffer', '_ct_buf')

    def __init__(self, layout: Layout, buffer: bytearray | None = None):
        self.layout = layout
        self.buffer = bytearray(layout.struct.size) if buffer is None else buffer
        self._ct_buf = (_ctypes.c_char * len(self.buffer)).from_buffer(self.buffer)

    def pack(self, *values):
        self.layout.struct.pack_into(self.buffer, 0, *values)

    def unpack(self):
        return self.layout.struct.unpack(self.buffer)

    @property
    def ptr(self):
        """ctypes void pointer to the buffer."""
        return _ctypes.cast(self._ct_buf, _ctypes.c_void_p)

    @property
    def address(self) -> int:
        """Raw memory address of the buffer."""
        return _ctypes.addressof(self._ct_buf)


# Module-level singletons for primitives
BYTE = Byte()
INT = Int()
FLOAT64 = Float64()
