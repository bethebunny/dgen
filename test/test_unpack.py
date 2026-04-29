"""Tests for builtin.unpack op: ASM round-trip, lowering to goto.region, JIT."""

from __future__ import annotations

from dgen import asm
from dgen.asm.parser import parse
from dgen import builtins as builtin_mod
from dgen.dialects import builtin, function, goto, record
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.passes.record_to_memory import RecordToMemory
from dgen.passes.unpack_to_goto import UnpackToGoto
from dgen.testing import assert_ir_equivalent, strip_prefix


def _full_pipeline():
    """Pipeline including record + unpack lowering for end-to-end JIT."""
    return Compiler(
        passes=[
            UnpackToGoto(),
            NormalizeRegionTerminators(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    )


# -- ASM round-trip ---------------------------------------------------------


def test_unpack_roundtrip():
    """builtin.unpack with two-element tuple round-trips through ASM."""
    ir = strip_prefix("""
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
    fn = parse(ir)
    assert isinstance(fn, function.FunctionOp)
    unpack_op = fn.body.result
    assert isinstance(unpack_op, builtin.UnpackOp)
    assert len(unpack_op.body.args) == 2
    assert [a.name for a in unpack_op.body.args] == ["x", "y"]
    assert_ir_equivalent(fn, asm.parse(asm.format(fn)))


# -- Lowering to goto.region ------------------------------------------------


def test_unpack_lowers_to_region():
    """UnpackToGoto rewrites builtin.unpack as goto.region(record.get<i>...).

    The body block is reused (same args), %self/%exit parameters are
    prepended, and one record.GetOp is created per body arg as the
    region's initial_arguments.
    """
    ir = strip_prefix("""
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
    fn = parse(ir)
    lowered = Compiler([UnpackToGoto()], IdentityPass()).run(fn)
    assert isinstance(lowered, function.FunctionOp)

    region = lowered.body.result
    assert isinstance(region, goto.RegionOp)
    assert len(region.body.parameters) == 2
    assert region.body.parameters[0].name == "self"
    exit_name = region.body.parameters[1].name
    assert exit_name is not None and exit_name.startswith("unpack_exit")
    # initial_arguments is a pack of two record.GetOps reading fields 0 and 1.
    initial = region.initial_arguments
    assert isinstance(initial, builtin_mod.PackOp)
    init_args = initial.values
    assert len(init_args) == 2
    assert all(isinstance(g, record.GetOp) for g in init_args)


# -- End-to-end JIT ---------------------------------------------------------


def test_unpack_jit_sum():
    """unpack a Tuple, sum the elements end-to-end."""
    exe = _full_pipeline().run(
        parse(
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
    )
    assert exe.run(7, 35).to_json() == 42
    assert exe.run(0, 0).to_json() == 0
    assert exe.run(1, 2).to_json() == 3


def test_unpack_jit_first_element():
    """unpack and return the first element (verifies field-0 extraction)."""
    exe = _full_pipeline().run(
        parse(
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
    )
    assert exe.run(99, 1).to_json() == 99


def test_unpack_three_elements_lowering():
    """A three-element unpack lowers to three record.GetOps as initial_arguments."""
    ir = strip_prefix("""
        | import algebra
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index, %c: index.Index):
        |     %t : Tuple<[index.Index, index.Index, index.Index]> = record.pack([%a, %b, %c])
        |     %r : index.Index = unpack(%t) body(%x: index.Index, %y: index.Index, %z: index.Index):
        |         %xy : index.Index = algebra.add(%x, %y)
        |         %sum : index.Index = algebra.add(%xy, %z)
    """)
    fn = parse(ir)
    lowered = Compiler([UnpackToGoto()], IdentityPass()).run(fn)
    assert isinstance(lowered, function.FunctionOp)
    region = lowered.body.result
    assert isinstance(region, goto.RegionOp)
    initial = region.initial_arguments
    assert isinstance(initial, builtin_mod.PackOp)
    init_args = initial.values
    assert len(init_args) == 3
    # Each initial arg is a record.GetOp pulling field i from the same tuple.
    first = init_args[0]
    assert isinstance(first, record.GetOp)
    first_source = first.record
    for i, g in enumerate(init_args):
        assert isinstance(g, record.GetOp)
        assert g.index.__constant__.to_json() == i
        assert g.record is first_source
