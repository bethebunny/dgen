"""Trait base class and runtime trait-checking API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .op import Op
    from .type import Type


class Trait:
    """Sentinel base class for all dgen traits."""


def has_trait(target: Type | Op, trait: type[Trait]) -> bool:
    """Check whether a type or op implements a trait."""
    return isinstance(target, trait)
