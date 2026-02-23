"""Memory layout types for language-agnostic struct descriptions.

Layouts are value-level descriptors: Byte, Int, Float64 are singletons;
Array, Pointer, FatPointer are parameterized via constructors.
"""

from __future__ import annotations

import ctypes


class Struct:
    """Base for memory layout types."""

    ctype = None
    llvm_type = None

    def byte_size(self) -> int:
        raise NotImplementedError

    def parse(self, obj) -> object:
        raise NotImplementedError

    def prepare_arg(self, value):
        """Convert Python value to (ctypes_arg, refs_to_keep_alive)."""
        return value, []


class Byte(Struct):
    def byte_size(self) -> int:
        return 1


class Int(Struct):
    """64-bit integer (i64)."""

    ctype = ctypes.c_int64
    llvm_type = "i64"

    def byte_size(self) -> int:
        return 8

    def parse(self, obj) -> int:
        assert isinstance(obj, int), f"expected int, got {type(obj).__name__}"
        return obj


class Float64(Struct):
    """64-bit float (f64)."""

    ctype = ctypes.c_double
    llvm_type = "double"

    def byte_size(self) -> int:
        return 8

    def parse(self, obj) -> float:
        assert isinstance(obj, (int, float)), f"expected number, got {type(obj).__name__}"
        return float(obj)


class Array(Struct):
    """Fixed-size inline array: n × sizeof(T) bytes."""

    ctype = ctypes.c_void_p
    llvm_type = "ptr"

    def __init__(self, element: Struct, count: int):
        self.element = element
        self.count = count

    def byte_size(self) -> int:
        return self.element.byte_size() * self.count

    def parse(self, obj) -> list:
        assert isinstance(obj, list), f"expected list, got {type(obj).__name__}"
        return [self.element.parse(v) for v in obj]

    def prepare_arg(self, value):
        buf = (self.element.ctype * self.count)(*value)
        return ctypes.cast(buf, ctypes.c_void_p), [buf]


class Pointer(Struct):
    """8-byte pointer to T."""

    ctype = ctypes.c_void_p
    llvm_type = "ptr"

    def __init__(self, pointee: Struct):
        self.pointee = pointee

    def byte_size(self) -> int:
        return 8


class FatPointer(Struct):
    """Pointer + i64 length (16 bytes)."""

    ctype = ctypes.c_void_p
    llvm_type = "ptr"

    def __init__(self, pointee: Struct):
        self.pointee = pointee

    def byte_size(self) -> int:
        return 16


# Module-level singletons for primitives
BYTE = Byte()
INT = Int()
FLOAT64 = Float64()
