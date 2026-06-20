"""Tests for the State effect / Linear Reference memory API.

Each ``memory.load``/``store``/``deallocate`` op consumes the input
``Reference<T>`` and produces a fresh one (or, for ``deallocate``,
discharges the linear thread). The verifier enforces single-consume on
``Reference`` (it has trait ``Linear``).
"""

from __future__ import annotations

import pytest

import dgen
from dgen.asm.parser import parse
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp, Nil, UnpackOp
from dgen.dialects.index import Index
from dgen.ir.verification import (
    DoubleConsumeError,
    LinearLeakError,
    verify_linearity,
)
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.lower_builtin import LowerBuiltin
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.testing import strip_prefix


def _jit(ir: str, *args: object) -> object:
    value = parse(strip_prefix(ir))
    compiler: Compiler[Executable] = Compiler(
        [
            LowerBuiltin(),
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    )
    exe = compiler.compile(value)
    return exe.run(*args).to_json()


# ---------------------------------------------------------------------------
# End-to-end: the user's sketched scalar pattern works through the JIT
# ---------------------------------------------------------------------------


def test_alloc_store_load_unpack_dealloc():
    """The canonical sketch from docs/effects.md works end-to-end."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %ref0 : memory.Reference<index.Index> = memory.heap_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %ref1 : memory.Reference<index.Index> = memory.store(%ref0, %zero)
        |     %loaded : Tuple<[index.Index, memory.Reference<index.Index>]> = memory.load(%ref1)
        |     %result : index.Index = unpack(%loaded) body(%v: index.Index, %ref: memory.Reference<index.Index>):
        |         %d : Nil = memory.deallocate(%ref)
        |         %r : index.Index = chain(%v, %d)
    """)
        == 0
    )


def test_stack_allocate_then_store_load():
    """Stack allocate, store a value, load it back through the linear thread."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %ref0 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v : index.Index = 42
        |     %ref1 : memory.Reference<index.Index> = memory.store(%ref0, %v)
        |     %loaded : Tuple<[index.Index, memory.Reference<index.Index>]> = memory.load(%ref1)
        |     %result : index.Index = unpack(%loaded) body(%out: index.Index, %ref: memory.Reference<index.Index>):
        |         %d : Nil = memory.deallocate(%ref)
        |         %r : index.Index = chain(%out, %d)
    """)
        == 42
    )


def test_two_stores_last_wins():
    """Two sequential stores; loaded value is the last one written."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %r0 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %a : index.Index = 10
        |     %b : index.Index = 20
        |     %r1 : memory.Reference<index.Index> = memory.store(%r0, %a)
        |     %r2 : memory.Reference<index.Index> = memory.store(%r1, %b)
        |     %loaded : Tuple<[index.Index, memory.Reference<index.Index>]> = memory.load(%r2)
        |     %result : index.Index = unpack(%loaded) body(%out: index.Index, %ref: memory.Reference<index.Index>):
        |         %d : Nil = memory.deallocate(%ref)
        |         %r : index.Index = chain(%out, %d)
    """)
        == 20
    )


# ---------------------------------------------------------------------------
# Linearity verifier rejects misuse
# ---------------------------------------------------------------------------


def test_linearity_rejects_double_consume():
    """Deallocating the same Reference twice is a double-consume."""
    value = parse(
        strip_prefix("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %r0 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %d1 : Nil = memory.deallocate(%r0)
        |     %d2 : Nil = memory.deallocate(%r0)
        |     %v : index.Index = 0
        |     %dchain : Nil = chain(%d1, %d2)
        |     %result : index.Index = chain(%v, %dchain)
    """)
    )
    with pytest.raises(DoubleConsumeError):
        verify_linearity(value)


def test_linearity_leak_when_capture_unused():
    """A Reference captured into a sub-block but never consumed there is a leak."""
    ref_type = memory.Reference(element_type=Index())
    ref = memory.StackAllocateOp(element_type=Index(), type=ref_type)
    inner = ChainOp(lhs=Index().constant(0), rhs=Nil().constant(None), type=Index())
    inner_block = dgen.Block(result=inner, captures=[ref])
    tup = Index().constant(0)
    outer = UnpackOp(tuple=tup, body=inner_block, type=Index())
    with pytest.raises(LinearLeakError):
        verify_linearity(outer)
