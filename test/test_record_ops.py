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
import pytest

from dgen.testing import strip_prefix

# Span<Index> has layout "PQ" (ptr + u64, 16 bytes, register-passable).
# We use it as the record type in tests because it's a two-field aggregate
# available in the builtin dialect.


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
        |     %result : index.Index = record_get<index.Index(0)>(%packed, %packed)
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
        |     %result : index.Index = record_get<index.Index(1)>(%packed, %packed)
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
        |     %result : index.Index = record_get<index.Index(1)>(%packed, %packed)
    """)
    )
    for a, b in [(0, 0), (1, 2), (42, 99), (2**32, 2**48)]:
        assert exe.run(a, b).to_json() == b


# ---------------------------------------------------------------------------
# record_set: mutate a field, verify via record_get on the same record
# ---------------------------------------------------------------------------


def test_record_set_then_get() -> None:
    """Pack, set field 1 to a new value, get it back from the same record."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %st : Nil = record_set<index.Index(1)>(%packed, %packed, %new_b)
        |     %result : index.Index = record_get<index.Index(1)>(%st, %packed)
    """)
    )
    assert exe.run(10, 20, 77).to_json() == 77


def test_record_set_preserves_other_field() -> None:
    """Set field 1, verify field 0 is unchanged."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %st : Nil = record_set<index.Index(1)>(%packed, %packed, %new_b)
        |     %result : index.Index = record_get<index.Index(0)>(%st, %packed)
    """)
    )
    assert exe.run(10, 20, 77).to_json() == 10


def test_record_set_field_zero() -> None:
    """Mutate field 0, read it back."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_a: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %st : Nil = record_set<index.Index(0)>(%packed, %packed, %new_a)
        |     %result : index.Index = record_get<index.Index(0)>(%st, %packed)
    """)
    )
    assert exe.run(10, 20, 99).to_json() == 99


def test_record_set_field_zero_preserves_field_one() -> None:
    """Mutate field 0, verify field 1 is unchanged."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_a: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %st : Nil = record_set<index.Index(0)>(%packed, %packed, %new_a)
        |     %result : index.Index = record_get<index.Index(1)>(%st, %packed)
    """)
    )
    assert exe.run(10, 20, 99).to_json() == 20


# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------


def test_record_pack_snapshot_after_set() -> None:
    """record_pack returns a snapshot; record_set mutates the backing slot
    but the pack's LoadOp value is already materialized. To observe the
    mutation, use record_get — not the original pack value.

    This test documents the current behavior (C copy semantics).
    """
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %new_b: index.Index):
        |     %packed : Span<index.Index> = record_pack([%a, %b])
        |     %st : Nil = record_set<index.Index(1)>(%packed, %packed, %new_b)
        |     %result : index.Index = record_get<index.Index(1)>(%st, %packed)
    """)
    )
    # record_get reads from the mutated slot — sees the new value.
    assert exe.run(10, 20, 77).to_json() == 77


@pytest.mark.xfail(
    reason="Non-pack values get independent spill slots; set+get don't share storage"
)
def test_record_set_get_on_function_arg() -> None:
    """record_set then record_get on a function argument (not from record_pack).

    Currently each creates its own spill slot, so the mutation is invisible.
    """
    exe = _compile(
        strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[existential.Some<index.Index>, index.Index], index.Index> = function.function<index.Index>() body(%some: existential.Some<index.Index>, %new_witness: index.Index):
        |     %st : Nil = record_set<index.Index(0)>(%some, %some, %new_witness)
        |     %result : index.Index = record_get<index.Index(0)>(%st, %some)
    """)
    )
    from dgen.dialects import existential
    from dgen.dialects.index import Index
    from dgen.memory import Memory

    some_index = existential.Some(bound=Index())
    payload = {"type": Index().to_json(), "value": 42}
    mem = Memory.from_json(some_index, payload)
    assert exe.run(mem, 999).to_json() == 999
