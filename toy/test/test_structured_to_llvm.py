"""Ch6 tests: Affine IR to LLVM-like IR lowering."""

from dgen.asm.parser import parse_module
from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from toy.passes.structured_to_llvm import StructuredToLLVM
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import strip_prefix

_compiler = Compiler([], IdentityPass())


def compile_to_llvm(ir_text: str) -> Module:
    m = parse_module(ir_text)
    affine = ToyToStructured().run(m, _compiler)
    return StructuredToLLVM().run(affine, _compiler)


def test_simple_constant(ir_snapshot):
    """Tensor constant passes through to LLVM level."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<1>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_constant_preserved(ir_snapshot):
    """Constants are preserved as tensor constants (not expanded to scalar stores)."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<1>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_2d_constant_preserved(ir_snapshot):
    """2D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_load_store_linearization(ir_snapshot):
    """Load/store with multi-dim indices are linearized."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([3, 2]), number.Float64> = toy.transpose(%0)
        |     %2 : Nil = toy.print(%1)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_3d_constant_preserved(ir_snapshot):
    """3D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<3>([2, 2, 2]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_3d_load_store_linearization(ir_snapshot):
    """3D load/store indices are linearized with stride multiplication."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<3>([2, 2, 2]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<memory.Shape<3>([2, 2, 2]), number.Float64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<memory.Shape<3>([2, 2, 2]), number.Float64> = toy.add(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot


def test_full_example(ir_snapshot):
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([3, 2]), number.Float64> = toy.transpose(%0)
        |     %2 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<memory.Shape<2>([3, 2]), number.Float64> = toy.transpose(%2)
        |     %4 : toy.Tensor<memory.Shape<2>([3, 2]), number.Float64> = toy.mul(%1, %3)
        |     %5 : Nil = toy.print(%4)
    """)
    assert compile_to_llvm(ir_text) == ir_snapshot
