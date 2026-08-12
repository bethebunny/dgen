"""Tests for LowerDestroy: destroys resolve to their allocation's base
destructor. Heap destroys free, stack destroys keep no runtime action,
and unresolvable origin threads are an error rather than a leak.
"""

from __future__ import annotations

import pytest

import dgen
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import emit_llvm_ir
from dgen.llvm.memory_to_llvm import MemoryToLLVM, UnloweredDestroyError
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.lower_builtin import LowerBuiltin
from dgen.passes.lower_destroy import LowerDestroy, UnresolvableOriginError
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


CELL_ROUNDTRIP = """
    | import function
    | import index
    | import memory
    |
    | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.{alloc}<index.Index>()
    |     %result : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o0: memory.Origin):
    |         %v : index.Index = 7
    |         %o1 : memory.Origin = memory.store(%o0, %ref, %v)
    |         %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o1, %ref)
    |         %r : index.Index = unpack(%loaded) body(%out: index.Index, %o2: memory.Origin):
    |             %d : Nil = memory.destroy(%o2)
    |             %rr : index.Index = chain(%out, %d)
"""


def test_heap_destroy_frees():
    text = _lowered_llvm(CELL_ROUNDTRIP.format(alloc="heap_allocate"))
    assert "@free" in text


def test_free_is_ordered_after_the_store():
    text = _lowered_llvm(CELL_ROUNDTRIP.format(alloc="heap_allocate"))
    lines = text.splitlines()
    store_line = next(i for i, l in enumerate(lines) if " store " in l)
    free_line = next(i for i, l in enumerate(lines) if "@free" in l and "call" in l)
    assert store_line < free_line


def test_stack_destroy_does_not_free():
    text = _lowered_llvm(CELL_ROUNDTRIP.format(alloc="stack_allocate"))
    assert "@free" not in text


def test_destroy_reaching_memory_to_llvm_is_an_error():
    value = parse(strip_prefix(CELL_ROUNDTRIP.format(alloc="heap_allocate")))
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


def test_origin_from_function_parameter_is_unresolvable():
    """A callee receiving ownership cannot yet free it. The destructor
    design in docs/origins.md resolves this."""
    o_arg = BlockArgument(name="o", type=memory.Origin())
    destroy = memory.DestroyOp(origin=o_arg, type=Nil())
    body_result = ChainOp(lhs=Index().constant(0), rhs=destroy, type=Index())
    func = FunctionOp(
        name="f",
        result_type=Index(),
        body=dgen.Block(result=body_result, args=[o_arg]),
        type=Function(arguments=pack([memory.Origin()]), result_type=Index()),
    )
    compiler: Compiler[object] = Compiler([LowerDestroy()], IdentityPass())
    with pytest.raises(UnresolvableOriginError):
        compiler.compile(func)
