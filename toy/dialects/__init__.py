"""Toy dialect package — helpers and monkey-patches for generated dialects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dgen import Constant, Type, Value
from dgen.dialects.builtin import Index
from dgen.module import Function

from toy.dialects.affine import Shape
from toy.dialects.toy import DimSizeOp, Tensor

# ===----------------------------------------------------------------------=== #
# Shape helpers
# ===----------------------------------------------------------------------=== #


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[Shape] from a list of dims."""
    rank = Index().constant(len(dims))
    return Shape(rank=rank).constant(dims)


# ===----------------------------------------------------------------------=== #
# Tensor helpers
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# DimSizeOp helpers
# ===----------------------------------------------------------------------=== #


def _dim_size_resolve_constant(self: DimSizeOp) -> int | None:
    """Return constant value if input type has a resolved shape."""
    if not isinstance(self.input.type, Tensor):
        return None
    shape: Value[Shape] = self.input.type.shape
    if not shape.ready:
        return None
    return shape.__constant__.to_json()[self.axis.__constant__.to_json()]


DimSizeOp.resolve_constant = _dim_size_resolve_constant  # type: ignore[assignment]


# ===----------------------------------------------------------------------=== #
# FunctionType (not dialect-registered)
# ===----------------------------------------------------------------------=== #


@dataclass
class FunctionType(Function):
    """Toy function signature with explicit input types."""

    inputs: list[Type]
