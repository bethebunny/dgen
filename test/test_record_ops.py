"""End-to-end tests for record ops (record.pack, record.get, record.set).

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
from dgen.passes.existential_to_record import ExistentialToRecord
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import strip_prefix


def _compile(ir_text: str):
    """Parse ASM and compile through the record-aware pipeline."""
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            ExistentialToRecord(),
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
        | import record
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Array<index.Index, index.Index(2)> = record.pack([%a, %b])
        |     %result : index.Index = record.get<index.Index(0)>(%packed)
    """)
    )
    assert exe.run(10, 20).to_json() == 10


def test_record_pack_get_field_one() -> None:
    """Pack two values, record_get field 1."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import record
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Array<index.Index, index.Index(2)> = record.pack([%a, %b])
        |     %result : index.Index = record.get<index.Index(1)>(%packed)
    """)
    )
    assert exe.run(10, 20).to_json() == 20


def test_record_pack_get_multiple_values() -> None:
    """Verify the round-trip works for several input values."""
    exe = _compile(
        strip_prefix("""
        | import function
        | import index
        | import record
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Array<index.Index, index.Index(2)> = record.pack([%a, %b])
        |     %result : index.Index = record.get<index.Index(1)>(%packed)
    """)
    )
    for a, b in [(0, 0), (1, 2), (42, 99), (2**32, 2**48)]:
        assert exe.run(a, b).to_json() == b


# NOTE: record.set was removed alongside the State-effect / Linear
# Reference refactor (see docs/effects.md). Records are SSA aggregates
# now; in-place field mutation is expressed at the memory level via the
# linear store/load pattern, not via a record op.
