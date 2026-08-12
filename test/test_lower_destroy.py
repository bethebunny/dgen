"""Tests for LowerDestroy. destroy applies the destructor attached to
its origin's type. Function-literal destructors inline at the destroy
site, bare origins destroy with no runtime action, and origin-threading
ops must annotate the origin type they consume.
"""

from __future__ import annotations

import pytest

from dgen.asm.parser import parse
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen, emit_llvm_ir
from dgen.llvm.memory_to_llvm import MemoryToLLVM, UnloweredDestroyError
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.lower_builtin import LowerBuiltin
from dgen.passes.lower_destroy import LowerDestroy, OriginThreadTypeError
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.testing import strip_prefix


def _lowered_llvm(ir: str) -> str:
    value = parse(strip_prefix(ir))
    compiler: Compiler[object] = Compiler(
        [
            LowerDestroy(),
            LowerBuiltin(),
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        IdentityPass(),
    )
    text, _ = emit_llvm_ir(compiler.compile(value))
    return text


ATTACHED_CELL = """
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
    |         %dtor : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o: memory.Origin) captures(%ref):
    |             %n : Nil = memory.deallocate(%o, %ref)
    |         %o1 : memory.Origin<%dtor> = memory.attach(%o0, %dtor)
    |         %v : index.Index = 7
    |         %o2 : memory.Origin<%dtor> = memory.store(%o1, %ref, %v)
    |         %loaded : Tuple<[index.Index, memory.Origin<%dtor>]> = memory.load(%o2, %ref)
    |         %r : index.Index = unpack(%loaded) body(%out: index.Index, %o3: memory.Origin<%dtor>) captures(%ref, %dtor):
    |             %d : Nil = memory.destroy(%o3)
    |             %rr : index.Index = chain(%out, %d)
"""


BARE_CELL = """
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
    |         %v : index.Index = 7
    |         %o1 : memory.Origin = memory.store(%o0, %ref, %v)
    |         %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o1, %ref)
    |         %r : index.Index = unpack(%loaded) body(%out: index.Index, %o2: memory.Origin):
    |             %d : Nil = memory.destroy(%o2)
    |             %rr : index.Index = chain(%out, %d)
"""


def test_attached_destructor_frees():
    text = _lowered_llvm(ATTACHED_CELL)
    assert "@free" in text


def test_free_is_ordered_after_the_store():
    text = _lowered_llvm(ATTACHED_CELL)
    lines = text.splitlines()
    store_line = next(i for i, l in enumerate(lines) if " store " in l)
    free_line = next(i for i, l in enumerate(lines) if "@free" in l and "call" in l)
    assert store_line < free_line


def test_destructor_function_is_not_emitted():
    """The inlined destructor literal is unlinked from the graph, so
    codegen emits only main."""
    text = _lowered_llvm(ATTACHED_CELL)
    assert text.count("define ") == 1


def test_bare_origin_destroy_has_no_runtime_action():
    text = _lowered_llvm(BARE_CELL)
    assert "@free" not in text


def test_destroy_reaching_memory_to_llvm_is_an_error():
    value = parse(strip_prefix(BARE_CELL))
    compiler: Compiler[object] = Compiler(
        [
            LowerBuiltin(),
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            MemoryToLLVM(),
        ],
        IdentityPass(),
    )
    with pytest.raises(UnloweredDestroyError):
        compiler.compile(value)


def test_store_dropping_the_destructor_is_an_error():
    """A store that consumes Origin<%dtor> but annotates bare Origin
    breaks the thread and would silently drop the free."""
    ir = """
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
    |         %dtor : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o: memory.Origin) captures(%ref):
    |             %n : Nil = memory.deallocate(%o, %ref)
    |         %o1 : memory.Origin<%dtor> = memory.attach(%o0, %dtor)
    |         %v : index.Index = 7
    |         %o2 : memory.Origin = memory.store(%o1, %ref, %v)
    |         %d : Nil = memory.destroy(%o2)
    |         %r : index.Index = chain(%v, %d)
    """
    value = parse(strip_prefix(ir))
    compiler: Compiler[object] = Compiler([LowerDestroy()], IdentityPass())
    with pytest.raises(OriginThreadTypeError):
        compiler.compile(value)


def test_runtime_selected_destructor_survives_to_runtime():
    """A destructor selected at runtime is data, not an inline splice.

    The origin's destructor parameter depends on the entry argument, so
    staging defers the function to a callback thunk and specializes it
    against the actual argument. The destroy then applies whichever
    destructor the run selected. This is the shared-pointer
    prerequisite, with the destructor as the origin's runtime data."""
    ir = """
    | import control_flow
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%which: index.Index):
    |     %fsel : function.Function<[memory.Origin], Nil> = control_flow.if(%which, [], []) then_body():
    |         %d0 : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o0: memory.Origin):
    |             %n0 : Nil = chain((), %o0)
    |     else_body():
    |         %d1 : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o1: memory.Origin):
    |             %n1 : Nil = chain((), %o1)
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.stack_allocate<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %oa0: memory.Origin) captures(%fsel):
    |         %oa : memory.Origin<%fsel> = memory.attach(%oa0, %fsel)
    |         %v : index.Index = 7
    |         %ob : memory.Origin<%fsel> = memory.store(%oa, %ref, %v)
    |         %d : Nil = memory.destroy(%ob)
    |         %r : index.Index = chain(%v, %d)
    """
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
    assert exe.run(1).to_json() == 7
    assert exe.run(0).to_json() == 7


def test_destructor_body_is_ordinary_ir():
    """A destructor that does extra work before its free. The body is
    plain IR, so composition needs no destructor ops in the dialect.
    Cleanup of other linear values from a destructor body awaits the
    destructor-stack design, since function bodies reject linear
    captures."""
    ir = """
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %flag : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin) captures(%flag):
    |         %dtor : function.Function<[memory.Origin], Nil> = function.function<Nil>() body(%o: memory.Origin) captures(%ref, %flag):
    |             %one : index.Index = 1
    |             %zero : index.Index = 0
    |             %mark : Nil = memory.buffer_store(%flag, %flag, %zero, %one)
    |             %freed : Nil = memory.deallocate(%o, %ref)
    |             %both : Nil = chain(%mark, %freed)
    |         %o1 : memory.Origin<%dtor> = memory.attach(%o0, %dtor)
    |         %v : index.Index = 7
    |         %o2 : memory.Origin<%dtor> = memory.store(%o1, %ref, %v)
    |         %d : Nil = memory.destroy(%o2)
    |         %r : index.Index = chain(%v, %d)
    """
    text = _lowered_llvm(ir)
    assert "@free" in text
    lines = text.splitlines()
    flag_store = next(i for i, l in enumerate(lines) if " store " in l and "i64 1" in l)
    free_line = next(i for i, l in enumerate(lines) if "@free" in l and "call" in l)
    assert flag_store < free_line
