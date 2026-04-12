"""Round-trip tests for register-passable small Records and Arrays.

Aggregates ≤ 16 bytes (typically two 8-byte fields or two i64 array
elements) flow through the JIT as integer pairs in registers under the
SysV ABI rather than via pointer indirection. These tests verify the
end-to-end path: build a layout, run an identity function over a Memory
of that layout, and confirm the bytes round-trip.
"""

from __future__ import annotations

import dgen
from dgen import Block, layout
from dgen.block import BlockArgument
from dgen.builtins import pack as pack_values
from dgen.dialects.builtin import Index
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.ndbuffer import Shape
from dgen.memory import Memory
from dgen.testing import llvm_compile


def _identity_exe(ty: dgen.Type):
    arg = BlockArgument(name="v0", type=ty)
    func = FunctionOp(
        name="main",
        body=Block(result=arg, args=[arg]),
        result_type=ty,
        type=Function(arguments=pack_values([arg.type]), result_type=ty),
    )
    return llvm_compile(func)


def test_small_array_is_register_passable() -> None:
    """``Array<Int, 2>`` (16 bytes) is register-passable."""
    arr = layout.Array(layout.Int(), 2)
    assert arr.byte_size == 16
    assert arr.register_passable


def test_large_array_is_not_register_passable() -> None:
    """An array bigger than 16 bytes stays passed by pointer."""
    arr = layout.Array(layout.Int(), 8)
    assert arr.byte_size == 64
    assert not arr.register_passable


def test_small_record_is_register_passable() -> None:
    """A two-pointer record (16 bytes) is register-passable."""
    rec = layout.Record(
        [("a", layout.Pointer(layout.Int())), ("b", layout.Pointer(layout.Int()))]
    )
    assert rec.byte_size == 16
    assert rec.register_passable


def test_large_record_is_not_register_passable() -> None:
    """A record bigger than 16 bytes stays passed by pointer."""
    rec = layout.Record(
        [
            ("a", layout.Int()),
            ("b", layout.Int()),
            ("c", layout.Int()),
        ]
    )
    assert rec.byte_size == 24
    assert not rec.register_passable


def test_shape_2_jit_identity_roundtrip() -> None:
    """``Shape<2>`` flows through the JIT as a register-passable Array.

    Previously this was passed via pointer indirection; with the
    register-passable change it goes through return registers.
    """
    shape_t = Shape(rank=Index().constant(2))
    assert shape_t.__layout__.register_passable
    exe = _identity_exe(shape_t)
    mem = Memory.from_value(shape_t, [3, 5])
    result = exe.run(mem)
    assert result.to_json() == [3, 5]


def test_shape_4_jit_identity_roundtrip() -> None:
    """``Shape<4>`` is 32 bytes — non-register-passable, still works via pointer."""
    shape_t = Shape(rank=Index().constant(4))
    assert not shape_t.__layout__.register_passable
    exe = _identity_exe(shape_t)
    mem = Memory.from_value(shape_t, [1, 2, 3, 4])
    result = exe.run(mem)
    assert result.to_json() == [1, 2, 3, 4]
