"""C dialect definitions and convenience constructors.

Maps C types to shared dgen types:
  int/unsigned  -> number.SignedInteger / number.UnsignedInteger
  float/double  -> number.Float64
  void          -> builtin.Nil
  T*            -> memory.Reference<T>
  T[N]          -> builtin.Array<T, N>
  int (*)(...)  -> function.Function<R>

Only CStruct and CUnion remain C-specific.
"""

from __future__ import annotations

import dgen
from dgen.dialects.builtin import Nil
from dgen.dialects.index import Index
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger


def c_int(bits: int = 32, signed: bool = True) -> dgen.Type:
    if signed:
        return SignedInteger(bits=Index().constant(bits))
    return UnsignedInteger(bits=Index().constant(bits))


def c_float() -> Float64:
    return Float64()


def c_double() -> Float64:
    return Float64()


def c_void() -> Nil:
    return Nil()


def c_ptr(pointee: dgen.Type) -> Reference:
    return Reference(element_type=pointee)
