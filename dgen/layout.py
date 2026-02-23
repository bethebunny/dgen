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
from typing import Any


class Layout:
    """Base for memory layout types."""

    _format: str | None = None

    @cached_property
    def struct(self) -> _struct.Struct:
        return _struct.Struct(f'@{self._format}')

    def byte_size(self) -> int:
        return self.struct.size

    def parse(self, obj: object) -> object:
        raise NotImplementedError

    def prepare_arg(self, value: object, type: object = None) -> tuple[Any, list[Any]]:
        """Convert Python value to (ctypes_arg, refs_to_keep_alive)."""
        return value, []


class Byte(Layout):
    _format = 'B'


class Int(Layout):
    """64-bit integer (i64)."""

    _format = 'q'

    def parse(self, obj: object) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj


class Float64(Layout):
    """64-bit float (f64)."""

    _format = 'd'

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
        self._format = f'{count}{element._format}'

    def parse(self, obj: object) -> list[object]:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]

    def prepare_arg(self, value: object, type: object = None) -> tuple[Any, list[Memory]]:
        mem = Memory(type) if type is not None else Memory._from_layout(self)
        mem.pack(*value)
        return mem.ptr, [mem]


class Pointer(Layout):
    """8-byte pointer to T."""

    _format = 'P'

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee


class FatPointer(Layout):
    """Pointer + i64 length (16 bytes)."""

    _format = 'PQ'

    def __init__(self, pointee: Layout) -> None:
        self.pointee = pointee


# ---------------------------------------------------------------------------
# Memory: typed buffer for layout ABI
# ---------------------------------------------------------------------------


class _LayoutAsType:
    """Thin wrapper so Memory can treat a bare Layout like a Type."""
    __slots__ = ('__layout__',)
    def __init__(self, layout: Layout) -> None:
        self.__layout__ = layout


class Memory:
    """Typed memory buffer — the ABI for a type."""

    __slots__ = ('type', 'buffer', '_ct_buf')

    def __init__(self, type: object, buffer: bytearray | None = None) -> None:
        self.type = type
        layout = type.__layout__
        self.buffer = bytearray(layout.struct.size) if buffer is None else buffer
        self._ct_buf = (_ctypes.c_char * len(self.buffer)).from_buffer(self.buffer)

    @classmethod
    def _from_layout(cls, layout: Layout) -> Memory:
        """Create Memory directly from a Layout (internal fallback)."""
        return cls(_LayoutAsType(layout))

    @property
    def layout(self) -> Layout:
        return self.type.__layout__

    def pack(self, *values: int | float | bytes) -> None:
        self.layout.struct.pack_into(self.buffer, 0, *values)

    def unpack(self) -> tuple[Any, ...]:
        return self.layout.struct.unpack(self.buffer)

    @classmethod
    def from_value(cls, type: object, value: object) -> Memory:
        """Create Memory from a Type and a Python value."""
        parsed = type.__layout__.parse(value)
        mem = cls(type)
        if isinstance(parsed, list):
            mem.pack(*parsed)
        else:
            mem.pack(parsed)
        return mem

    @classmethod
    def from_asm(cls, type: object, text: str) -> Memory:
        """Create Memory from a Type and an ASM literal string."""
        from dgen.asm.parser import IRParser, parse_expr
        parser = IRParser(text)
        value = parse_expr(parser)
        return cls.from_value(type, value)

    @property
    def ptr(self) -> _ctypes.c_void_p:
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
