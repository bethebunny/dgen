"""Tests for the ndbuffer dialect: types, lowering, and end-to-end JIT.

NDBuffer<shape, dtype> has Span layout (pointer + length). These tests
verify the type system, the NDBufferToMemory lowering, and full JIT
execution of alloc/load/store/dealloc on 1-D, 2-D, and 3-D buffers.
"""

from __future__ import annotations

import pytest

from dgen.asm.parser import parse
from dgen.dialects import ndbuffer
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.layout import Span
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import strip_prefix

import dgen


def _compile(ir_text: str):
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            NDBufferToMemory(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    ).run(parse(ir_text))


def _lower(ir_text: str) -> dgen.Value:
    """Lower through NDBufferToMemory only."""
    return Compiler(
        [NDBufferToMemory()],
        IdentityPass(),
    ).run(parse(ir_text))


# ---------------------------------------------------------------------------
# Type and layout
# ---------------------------------------------------------------------------


def test_ndbuffer_has_span_layout() -> None:
    """NDBuffer<Shape<1>([3]), Float64> has Span layout."""
    shape = ndbuffer.Shape(rank=Index().constant(1))
    nb = ndbuffer.NDBuffer(shape=shape.constant([3]), dtype=Float64())
    assert isinstance(nb.__layout__, Span)
    assert nb.__layout__.struct.format == "PQ"


def test_ndbuffer_shape_accessible() -> None:
    """Shape parameter is accessible and resolves to the correct dimensions."""
    shape = ndbuffer.Shape(rank=Index().constant(2))
    nb = ndbuffer.NDBuffer(shape=shape.constant([4, 5]), dtype=Float64())
    assert nb.shape.__constant__.to_json() == [4, 5]


# ---------------------------------------------------------------------------
# End-to-end: 1-D alloc, store, load
# ---------------------------------------------------------------------------


def test_alloc_store_load_1d() -> None:
    """Allocate a 1-D buffer, store a value, load it back."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([3]))
        |     %st : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(0)])
        |     %val : number.Float64 = ndbuffer.load(%st, %buf, [index.Index(0)])
    """)
    )
    assert exe.run(42.0).to_json() == 42.0


def test_alloc_store_load_different_indices() -> None:
    """Store at index 1, load from index 1 — verify correct offset."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([3]))
        |     %st : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(1)])
        |     %val : number.Float64 = ndbuffer.load(%st, %buf, [index.Index(1)])
    """)
    )
    assert exe.run(7.5).to_json() == 7.5


# ---------------------------------------------------------------------------
# End-to-end: 2-D alloc, store, load with linearization
# ---------------------------------------------------------------------------


def test_alloc_store_load_2d() -> None:
    """Allocate a 2x3 buffer, store at [1,2], load it back."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(2)>([2, 3]))
        |     %st : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(1), index.Index(2)])
        |     %val : number.Float64 = ndbuffer.load(%st, %buf, [index.Index(1), index.Index(2)])
    """)
    )
    assert exe.run(99.0).to_json() == 99.0


# ---------------------------------------------------------------------------
# End-to-end: multiple stores, verify independence
# ---------------------------------------------------------------------------


def test_multiple_stores_independent() -> None:
    """Store different values at different indices, read both back."""
    exe = _compile(
        strip_prefix("""
        | import algebra
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64, number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64, %y: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([4]))
        |     %st1 : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(0)])
        |     %st2 : Nil = ndbuffer.store(%st1, %y, %buf, [index.Index(1)])
        |     %a : number.Float64 = ndbuffer.load(%st2, %buf, [index.Index(0)])
        |     %b : number.Float64 = ndbuffer.load(%st2, %buf, [index.Index(1)])
        |     %result : number.Float64 = algebra.add(%a, %b)
    """)
    )
    assert exe.run(10.0, 20.0).to_json() == 30.0


# ---------------------------------------------------------------------------
# Lowering snapshots: verify the IR after NDBufferToMemory + RecordToMemory
# ---------------------------------------------------------------------------


def test_lowering_alloc_store_load_1d(lowering_snapshot) -> None:
    """Lowered IR for 1-D alloc + store + load."""
    lowering_snapshot(
        [NDBufferToMemory()],
        """
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([3]))
        |     %st : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(0)])
        |     %val : number.Float64 = ndbuffer.load(%st, %buf, [index.Index(0)])
        """,
    )


def test_lowering_alloc_store_load_2d(lowering_snapshot) -> None:
    """Lowered IR for 2-D alloc + store + load with linearization."""
    lowering_snapshot(
        [NDBufferToMemory()],
        """
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(2)>([2, 3]))
        |     %st : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(1), index.Index(2)])
        |     %val : number.Float64 = ndbuffer.load(%st, %buf, [index.Index(1), index.Index(2)])
        """,
    )


def test_lowering_multiple_stores(lowering_snapshot) -> None:
    """Lowered IR for multiple stores at different indices."""
    lowering_snapshot(
        [NDBufferToMemory()],
        """
        | import algebra
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[number.Float64, number.Float64], number.Float64> = function.function<number.Float64>() body(%x: number.Float64, %y: number.Float64):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([4]))
        |     %st1 : Nil = ndbuffer.store(%buf, %x, %buf, [index.Index(0)])
        |     %st2 : Nil = ndbuffer.store(%st1, %y, %buf, [index.Index(1)])
        |     %a : number.Float64 = ndbuffer.load(%st2, %buf, [index.Index(0)])
        |     %b : number.Float64 = ndbuffer.load(%st2, %buf, [index.Index(1)])
        |     %result : number.Float64 = algebra.add(%a, %b)
        """,
    )


# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    raises=AssertionError,
    reason="Lowering asserts indices are a PackOp; Span from other sources fails",
)
def test_load_with_indices_from_non_pack() -> None:
    """Indices passed as a function argument (Span<Index>, not a pack literal)."""
    _compile(
        strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %main : function.Function<[Span<index.Index>], number.Float64> = function.function<number.Float64>() body(%indices: Span<index.Index>):
        |     %buf : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([4]))
        |     %val : number.Float64 = ndbuffer.load(%buf, %buf, %indices)
    """)
    )
