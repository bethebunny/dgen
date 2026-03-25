"""Toy dialect package — helpers and monkey-patches for generated dialects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dgen import Constant, Type
from dgen.dialects.builtin import Index
from dgen.dialects.function import Function

from toy.dialects.memory import Shape

# ===----------------------------------------------------------------------=== #
# Shape helpers
# ===----------------------------------------------------------------------=== #


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[Shape] from a list of dims."""
    rank = Index().constant(len(dims))
    return Shape(rank=rank).constant(dims)


# ===----------------------------------------------------------------------=== #
# FunctionType (not dialect-registered)
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True, eq=False)
class FunctionType(Function):
    """Toy function signature with explicit input types."""

    inputs: list[Type]
