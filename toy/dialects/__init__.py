"""Toy dialect package — helpers and monkey-patches for generated dialects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from dgen import Constant, Type, Value
from dgen.dialects.builtin import IndexType
from dgen.module import Function
from dgen.layout import FLOAT64, Array, Layout
from dgen.type import Memory

from toy.dialects.affine import ShapeType
from toy.dialects.toy import DimSizeOp, TensorType

# ===----------------------------------------------------------------------=== #
# ShapeType helpers
# ===----------------------------------------------------------------------=== #


@classmethod  # type: ignore[misc]
def _shape_for_value(cls: type[ShapeType], value: object) -> ShapeType:
    if isinstance(value, Constant):
        assert isinstance(value.type, IndexType)
        return cls(rank=IndexType().constant(value.__constant__.unpack()[0]))
    assert isinstance(value, list)
    return cls(rank=IndexType().constant(len(value)))


ShapeType.for_value = _shape_for_value  # type: ignore[assignment]


def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[ShapeType] from a list of dims."""
    rank = IndexType().constant(len(dims))
    return ShapeType(rank=rank).constant(dims)


# ===----------------------------------------------------------------------=== #
# TensorType helpers
# ===----------------------------------------------------------------------=== #


def _tensor_unpack_shape(self: TensorType) -> list[int]:
    """Extract concrete shape dimensions as a list of ints."""
    return list(self.shape.__constant__.unpack())


TensorType.unpack_shape = _tensor_unpack_shape  # type: ignore[assignment]


@property  # type: ignore[misc]
def _tensor_layout(self: TensorType) -> Layout:
    assert self.shape.ready
    shape: Memory[ShapeType] = self.shape.__constant__
    return Array(FLOAT64, prod(shape.unpack()))


TensorType.__layout__ = _tensor_layout  # type: ignore[assignment, misc]


# ===----------------------------------------------------------------------=== #
# DimSizeOp helpers
# ===----------------------------------------------------------------------=== #


def _dim_size_resolve_constant(self: DimSizeOp) -> int | None:
    """Return constant value if input type has a resolved shape."""
    if not isinstance(self.input.type, TensorType):
        return None
    shape: Value[ShapeType] = self.input.type.shape
    if not shape.ready:
        return None
    return shape.__constant__.unpack()[self.axis.__constant__.unpack()[0]]


DimSizeOp.resolve_constant = _dim_size_resolve_constant  # type: ignore[assignment]


# ===----------------------------------------------------------------------=== #
# FunctionType (not dialect-registered)
# ===----------------------------------------------------------------------=== #


@dataclass
class FunctionType(Function):
    """Toy function signature with explicit input types."""

    inputs: list[Type]
