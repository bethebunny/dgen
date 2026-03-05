"""Toy dialect package — helpers and monkey-patches for generated dialects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from dgen import Constant, Type, Value
from dgen import layout
from dgen.dialects.builtin import Index
from dgen.module import Function
from dgen.layout import Layout
from dgen.type import Memory

from toy.dialects.affine import Shape
from toy.dialects.toy import DimSizeOp, Tensor

# ===----------------------------------------------------------------------=== #
# Shape helpers
# ===----------------------------------------------------------------------=== #


@classmethod  # type: ignore[misc]
def _shape_for_value(cls: type[Shape], value: object) -> Shape:
    if isinstance(value, Constant):
        assert isinstance(value.type, Index)
        return cls(rank=Index().constant(value.__constant__.to_json()))
    assert isinstance(value, list)
    return cls(rank=Index().constant(len(value)))


Shape.for_value = _shape_for_value  # type: ignore[assignment]


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[Shape] from a list of dims."""
    rank = Index().constant(len(dims))
    return Shape(rank=rank).constant(dims)


# ===----------------------------------------------------------------------=== #
# Tensor helpers
# ===----------------------------------------------------------------------=== #


def _tensor_unpack_shape(self: Tensor) -> list[int]:
    """Extract concrete shape dimensions as a list of ints."""
    return self.shape.__constant__.to_json()


Tensor.unpack_shape = _tensor_unpack_shape  # type: ignore[assignment]


@property  # type: ignore[misc]
def _tensor_layout(self: Tensor) -> Layout:
    assert self.shape.ready
    shape: Memory[Shape] = self.shape.__constant__
    return layout.Array(layout.Float64(), prod(shape.to_json()))


Tensor.__layout__ = _tensor_layout  # type: ignore[assignment, misc]


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
