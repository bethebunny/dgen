"""Tests for builtin.unpack op: ASM round-trip, lowering, and end-to-end JIT.

Lowering snapshots verify the IR after LowerBuiltin only.
End-to-end tests verify the full JIT pipeline.
"""

from __future__ import annotations

import dgen
from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import builtin
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.lower_builtin import LowerBuiltin
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import assert_ir_equivalent, strip_prefix


def _lower(ir_text: str) -> dgen.Value:
    """Lower through LowerBuiltin only."""
    return Compiler([LowerBuiltin()], IdentityPass()).run(parse(ir_text))


def _compile(ir_text: str):
    """Parse ASM and compile through the unpack-aware pipeline."""
    return Compiler(
        passes=[
            LowerBuiltin(),
            NormalizeRegionTerminators(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    ).run(parse(ir_text))


# -- ASM round-trip ---------------------------------------------------------


def test_unpack_roundtrip() -> None:
    """builtin.unpack with two-element tuple round-trips through ASM."""
    ir = strip_prefix("""
        | import algebra
        | import index
        | import record
        |
        | %a : index.Index = 10
        | %b : index.Index = 20
        | %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
        | %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index):
        |     %sum : index.Index = algebra.add(%x, %y)
    """)
    value = parse(ir)
    assert isinstance(value, builtin.UnpackOp)
    assert [a.name for a in value.body.args] == ["x", "y"]
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


# -- Lowering snapshots -----------------------------------------------------


def test_lowering_two_elements(ir_snapshot) -> None:
    """unpack(%t) over a two-tuple lowers to a goto.region capturing %t."""
    assert (
        _lower(
            strip_prefix("""
        | import algebra
        | import index
        | import record
        |
        | %a : index.Index = 10
        | %b : index.Index = 20
        | %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
        | %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index):
        |     %sum : index.Index = algebra.add(%x, %y)
    """)
        )
        == ir_snapshot
    )


def test_lowering_first_element(ir_snapshot) -> None:
    """Body that only references the first arg still lowers cleanly."""
    assert (
        _lower(
            strip_prefix("""
        | import index
        | import record
        |
        | %a : index.Index = 99
        | %b : index.Index = 1
        | %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
        | %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index):
        |     %first : index.Index = chain(%x, %y)
    """)
        )
        == ir_snapshot
    )


def test_lowering_three_elements(ir_snapshot) -> None:
    """A three-element unpack produces three record.get ops in the body."""
    assert (
        _lower(
            strip_prefix("""
        | import algebra
        | import index
        | import record
        |
        | %a : index.Index = 1
        | %b : index.Index = 2
        | %c : index.Index = 3
        | %t : Tuple<[index.Index, index.Index, index.Index]> = record.pack([%a, %b, %c])
        | %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index, %z: index.Index):
        |     %xy : index.Index = algebra.add(%x, %y)
        |     %sum : index.Index = algebra.add(%xy, %z)
    """)
        )
        == ir_snapshot
    )


# -- End-to-end JIT ---------------------------------------------------------


def test_unpack_jit_sum() -> None:
    """unpack a Tuple, sum the two elements end-to-end."""
    exe = _compile(
        strip_prefix("""
        | import algebra
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
        |     %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index):
        |         %sum : index.Index = algebra.add(%x, %y)
    """)
    )
    assert exe.run(7, 35).to_json() == 42
    assert exe.run(0, 0).to_json() == 0
    assert exe.run(1, 2).to_json() == 3


def test_unpack_jit_first_element() -> None:
    """unpack and return the first element (verifies field-0 extraction)."""
    exe = _compile(
        strip_prefix("""
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
        |     %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index):
        |         %first : index.Index = chain(%x, %y)
    """)
    )
    assert exe.run(99, 1).to_json() == 99
