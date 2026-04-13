"""Ch6 tests: Structured IR to LLVM-like IR lowering."""

import dgen
from dgen.asm.parser import parse
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.record_to_memory import RecordToMemory
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import strip_prefix


def compile_to_llvm(ir_text: str) -> dgen.Value:
    m = parse(ir_text)
    return Compiler(
        [ToyToStructured(), ControlFlowToGoto(), NDBufferToMemory(), RecordToMemory(), MemoryToLLVM()],
        IdentityPass(),
    ).run(m)


def test_simple_constant(ir_snapshot):
    """Tensor constant passes through to LLVM level."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_constant_preserved(ir_snapshot):
    """Constants are preserved as tensor constants (not expanded to scalar stores)."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_2d_constant_preserved(ir_snapshot):
    """2D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_load_store_linearization(ir_snapshot):
    """Load/store with multi-dim indices are linearized."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%0)
        |     %2 : Nil = toy.print(%1)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_3d_constant_preserved(ir_snapshot):
    """3D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(3)>([2, 2, 2]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_3d_load_store_linearization(ir_snapshot):
    """3D load/store indices are linearized with stride multiplication."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(3)>([2, 2, 2]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<ndbuffer.Shape<index.Index(3)>([2, 2, 2]), number.Float64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<ndbuffer.Shape<index.Index(3)>([2, 2, 2]), number.Float64> = toy.add(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_full_example(ir_snapshot):
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
    ir_text = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%0)
        |     %2 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%2)
        |     %4 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.mul(%1, %3)
        |     %5 : Nil = toy.print(%4)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot
