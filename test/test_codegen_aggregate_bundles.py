"""Tests for the aggregate-bundle codegen path.

The codegen treats bundles uniformly: a ``PackOp`` constructed inline (the
common case from inline ``[a, b]`` IR syntax) and a runtime aggregate
value (a ``Tuple`` returned by a function call, etc.) are both valid
bundles. PackOps short-circuit; runtime aggregates use ``extractvalue``.

These tests pin down the runtime-aggregate behaviour end-to-end.
"""

from __future__ import annotations

import dgen
from dgen.asm.parser import parse
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import llvm_compile, strip_prefix


def _record_aware_compile(value: dgen.Value) -> Executable:
    """Like ``llvm_compile`` but lowers ``record.pack`` through
    ``RecordToMemory`` + ``MemoryToLLVM`` for tests that build runtime
    aggregates with ``record.pack``."""
    return Compiler(
        [
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    ).run(value)


def test_call_with_runtime_tuple_as_args():
    """A function whose ``arguments`` come from a runtime Tuple value
    (here, the result of another function call) should still execute
    correctly — codegen extracts each LLVM arg via ``extractvalue``
    rather than reading inline PackOp values."""
    exe = _record_aware_compile(
        parse(
            strip_prefix("""
        | import algebra
        | import index
        | import record
        | import function
        |
        | %add : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %sum : index.Index = algebra.add(%a, %b)
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: index.Index) captures(%add):
        |     %t : Tuple<[index.Index, index.Index]> = record.pack([%x, %y])
        |     %r : index.Index = function.call<%add>(%t)
    """)
        )
    )
    assert exe.run(7, 35).to_json() == 42
    assert exe.run(0, 0).to_json() == 0
    assert exe.run(100, 200).to_json() == 300


def test_packop_aggregate_round_trips_through_inline_pack_call():
    """Sanity check the PackOp short-circuit: an inline ``[%a, %b]`` carrying
    two homogeneous values still passes through extractvalue-free at the
    call site (no aggregate constructed)."""
    exe = llvm_compile(
        parse(
            strip_prefix("""
        | import algebra
        | import index
        | import function
        |
        | %add : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %sum : index.Index = algebra.add(%a, %b)
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: index.Index) captures(%add):
        |     %r : index.Index = function.call<%add>([%x, %y])
    """)
        )
    )
    assert exe.run(11, 31).to_json() == 42


def test_call_with_heterogeneous_tuple_args():
    """Mixed-type runtime tuple flowing into a call. Verifies that
    ``extractvalue`` honours each field's individual type."""
    exe = _record_aware_compile(
        parse(
            strip_prefix("""
        | import index
        | import number
        | import record
        | import function
        |
        | %first : function.Function<[index.Index, number.Float64], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: number.Float64):
        |     %_ : index.Index = chain(%a, %b)
        | %main : function.Function<[index.Index, number.Float64], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: number.Float64) captures(%first):
        |     %t : Tuple<[index.Index, number.Float64]> = record.pack([%x, %y])
        |     %r : index.Index = function.call<%first>(%t)
    """)
        )
    )
    assert exe.run(7, 0.5).to_json() == 7
