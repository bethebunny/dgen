"""Tests for toy-specific memory layout."""

from dgen.layout import FatPointer
from dgen.type import Memory
from toy.dialects import shape_constant
from toy.dialects.toy import Tensor


def test_tensor_type_layout():
    t = Tensor(shape=shape_constant([2, 3]))
    ly = t.__layout__
    assert ly.byte_size == 16  # FatPointer: ptr + length
    assert isinstance(ly, FatPointer)


def test_array_to_json():
    ty = Tensor(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]
