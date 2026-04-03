"""Toy dialect package — helpers and monkey-patches for generated dialects."""

from __future__ import annotations

from collections.abc import Sequence

from dgen import Constant
from dgen.dialects.index import Index

from dgen.dialects.ndbuffer import Shape

# ===----------------------------------------------------------------------=== #
# Shape helpers
# ===----------------------------------------------------------------------=== #


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[Shape] from a list of dims."""
    rank = Index().constant(len(dims))
    return Shape(rank=rank).constant(dims)
