"""Tests for the State effect / Origin + Ref memory API.

``memory.Origin`` is the linear ``Handler<State>``: evidence for a
region of memory, erased at codegen. ``memory.Reference<T>`` is plain pointer
data. Allocation returns ``Tuple<[Reference<T>, Origin]>``; ``load`` returns
``Tuple<[T, Origin]>``; ``store`` consumes the input origin and returns
a fresh one; ``destroy`` discharges the thread. The verifier enforces
single-consume on ``Origin`` (it has trait ``Linear``). See
``docs/origins.md``.
"""

from __future__ import annotations

import pytest

from dgen import asm
from dgen.asm.parser import parse
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
from dgen.passes.lower_destroy import LowerDestroy
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.testing import assert_ir_equivalent, strip_prefix


def _jit(ir: str, *args: object) -> object:
    value = parse(strip_prefix(ir))
    compiler: Compiler[Executable] = Compiler(
        [
            LowerDestroy(),
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
# End-to-end: the canonical cell pattern works through the JIT
# ---------------------------------------------------------------------------


def test_alloc_store_load_unpack_destroy():
    """The canonical sketch from docs/origins.md works end-to-end. The
    heap cell attaches its destructor, so the destroy frees it."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
        |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |         %dtor : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o: memory.Origin) captures(%ref):
        |             %n : Nil = memory.deallocate(%o, %ref)
        |         %oa : memory.Origin<%dtor> = memory.attach(%o0, %dtor)
        |         %zero : index.Index = 0
        |         %o1 : memory.Origin<%dtor> = memory.store(%oa, %ref, %zero)
        |         %loaded : Tuple<[index.Index, memory.Origin<%dtor>]> = memory.load(%o1, %ref)
        |         %r : index.Index = unpack(%loaded) body(%v: index.Index, %o2: memory.Origin<%dtor>) captures(%ref, %dtor):
        |             %d : Nil = memory.destroy(%o2)
        |             %out : index.Index = chain(%v, %d)
    """)
        == 0
    )


def test_stack_allocate_then_store_load():
    """Stack allocate, store a value, load it back through the origin thread."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |         %v : index.Index = 42
        |         %o1 : memory.Origin = memory.store(%o0, %ref, %v)
        |         %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o1, %ref)
        |         %r : index.Index = unpack(%loaded) body(%out: index.Index, %o2: memory.Origin):
        |             %d : Nil = memory.destroy(%o2)
        |             %rr : index.Index = chain(%out, %d)
    """)
        == 42
    )


def test_two_stores_last_wins():
    """Two sequential stores threading one origin; the last write wins."""
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |         %a : index.Index = 10
        |         %b : index.Index = 20
        |         %o1 : memory.Origin = memory.store(%o0, %ref, %a)
        |         %o2 : memory.Origin = memory.store(%o1, %ref, %b)
        |         %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o2, %ref)
        |         %r : index.Index = unpack(%loaded) body(%out: index.Index, %o3: memory.Origin):
        |             %d : Nil = memory.destroy(%o3)
        |             %rr : index.Index = chain(%out, %d)
    """)
        == 20
    )


# ---------------------------------------------------------------------------
# ASM round-trip
# ---------------------------------------------------------------------------


def test_origin_api_asm_roundtrip():
    """The Origin/Ref cell ops survive format → parse."""
    value = parse(
        strip_prefix("""
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |     %v : index.Index = 7
        |     %o1 : memory.Origin = memory.store(%o0, %ref, %v)
        |     %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o1, %ref)
        |     %r : index.Index = unpack(%loaded) body(%out: index.Index, %o2: memory.Origin):
        |         %d : Nil = memory.destroy(%o2)
        |         %rr : index.Index = chain(%out, %d)
    """)
    )
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


# ---------------------------------------------------------------------------
# Linearity verifier rejects misuse
# ---------------------------------------------------------------------------


def test_linearity_rejects_double_destroy():
    """Destroying the same Origin twice is a double-consume."""
    value = parse(
        strip_prefix("""
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |     %d1 : Nil = memory.destroy(%o0)
        |     %d2 : Nil = memory.destroy(%o0)
        |     %v : index.Index = 0
        |     %dchain : Nil = chain(%d1, %d2)
        |     %r : index.Index = chain(%v, %dchain)
    """)
    )
    with pytest.raises(DoubleConsumeError):
        verify_linearity(value)


def test_linearity_rejects_leaked_origin():
    """An Origin that is never consumed is a leak at block exit."""
    value = parse(
        strip_prefix("""
        | import index
        | import memory
        | %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
        | %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
        |     %v : index.Index = 5
    """)
    )
    with pytest.raises(LinearLeakError):
        verify_linearity(value)
