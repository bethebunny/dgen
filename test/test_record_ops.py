"""End-to-end tests for record ops (record_pack, record_get, record_set).

Tests through the full JIT pipeline: ASM parse → lowering → LLVM codegen →
JIT execution → result verification.
"""

from __future__ import annotations

from dgen.asm.parser import parse
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.existential_to_memory import ExistentialToMemory
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import strip_prefix


def _compile(ir_text: str):
    """Parse ASM and compile through the record-aware pipeline."""
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            ExistentialToMemory(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    ).run(parse(ir_text))


# ---------------------------------------------------------------------------
# record_pack + record_get: construct and read back
# ---------------------------------------------------------------------------


def test_record_pack_get_field_zero() -> None:
    """Pack two values, record_get field 0."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %result : index.Index = record_get<index.Index(0)>(%packed)
    """)
    )
    assert exe.run(10, 20).to_json() == 10


def test_record_pack_get_field_one() -> None:
    """Pack two values, record_get field 1."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %result : index.Index = record_get<index.Index(1)>(%packed)
    """)
    )
    assert exe.run(10, 20).to_json() == 20


def test_record_pack_get_multiple_values() -> None:
    """Verify the round-trip works for several input values."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %result : index.Index = record_get<index.Index(1)>(%packed)
    """)
    )
    for a, b in [(0, 0), (1, 2), (42, 99), (2**32, 2**48)]:
        assert exe.run(a, b).to_json() == b


# ---------------------------------------------------------------------------
# record_set: mutate a field in a Reference, verify via load + record_get
# ---------------------------------------------------------------------------


def test_record_set_then_load_get() -> None:
    """Pack, store to ref, set field 1, load back, get field 1."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %ref : memory.Reference<Span<index.Index>> = memory.stack_allocate<Span<index.Index>>()
        |     %st : Nil = memory.store(%ref, %packed, %ref)
        |     %st2 : Nil = record_set<index.Index(1)>(%st, %ref, %new_b)
        |     %updated : Span<index.Index> = memory.load(%st2, %ref)
        |     %result : index.Index = record_get<index.Index(1)>(%updated)
    """)
    )
    assert exe.run(10, 20, 77).to_json() == 77


def test_record_set_preserves_other_field() -> None:
    """Set field 1, verify field 0 is unchanged."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %ref : memory.Reference<Span<index.Index>> = memory.stack_allocate<Span<index.Index>>()
        |     %st : Nil = memory.store(%ref, %packed, %ref)
        |     %st2 : Nil = record_set<index.Index(1)>(%st, %ref, %new_b)
        |     %updated : Span<index.Index> = memory.load(%st2, %ref)
        |     %result : index.Index = record_get<index.Index(0)>(%updated)
    """)
    )
    assert exe.run(10, 20, 77).to_json() == 10


def test_record_set_field_zero() -> None:
    """Set field 0, read it back."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_a: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %ref : memory.Reference<Span<index.Index>> = memory.stack_allocate<Span<index.Index>>()
        |     %st : Nil = memory.store(%ref, %packed, %ref)
        |     %st2 : Nil = record_set<index.Index(0)>(%st, %ref, %new_a)
        |     %updated : Span<index.Index> = memory.load(%st2, %ref)
        |     %result : index.Index = record_get<index.Index(0)>(%updated)
    """)
    )
    assert exe.run(10, 20, 99).to_json() == 99


def test_record_set_field_zero_preserves_field_one() -> None:
    """Set field 0, verify field 1 is unchanged."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_a: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %ref : memory.Reference<Span<index.Index>> = memory.stack_allocate<Span<index.Index>>()
        |     %st : Nil = memory.store(%ref, %packed, %ref)
        |     %st2 : Nil = record_set<index.Index(0)>(%st, %ref, %new_a)
        |     %updated : Span<index.Index> = memory.load(%st2, %ref)
        |     %result : index.Index = record_get<index.Index(1)>(%updated)
    """)
    )
    assert exe.run(10, 20, 99).to_json() == 20
