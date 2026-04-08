"""Trait base class for dgen dialects."""

from __future__ import annotations

from .type import TypeType


class Trait(TypeType):
    """Base class for all dgen traits.

    Traits are type values in the type hierarchy — they describe sets
    of types. A trait can appear wherever a type can: as a type annotation,
    in constraint checks, in ASM type position.

    """
