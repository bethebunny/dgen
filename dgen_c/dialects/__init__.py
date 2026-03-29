"""C dialect definitions and convenience constructors."""

from __future__ import annotations

from dgen.dialects.index import Index

from dgen_c.dialects.c import CFloat, CInt, CPtr, CVoid


def c_int(bits: int = 32, signed: bool = True) -> CInt:
    """Create a CInt type constant."""
    return CInt(
        bits=Index().constant(bits),
        signed=Index().constant(1 if signed else 0),
    )


def c_float() -> CFloat:
    return CFloat(kind=Index().constant(0))


def c_double() -> CFloat:
    return CFloat(kind=Index().constant(1))


def c_void() -> CVoid:
    return CVoid()


def c_ptr(pointee: CInt | CFloat | CVoid | CPtr) -> CPtr:
    return CPtr(pointee=pointee)
