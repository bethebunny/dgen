"""Ch5 tests: Toy IR to Affine IR lowering."""

from dgen.asm.parser import parse_module
from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import strip_prefix

_compiler = Compiler([], IdentityPass())


def lower_to_affine(m: Module) -> Module:
    return ToyToStructured().run(m, _compiler)


def test_simple_constant(ir_snapshot):
    """Tensor constant passes through as-is (no alloc/store expansion)."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_transpose(ir_snapshot):
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([3, 2]), F64> = toy.transpose(%0)
        |     %2 : Nil = toy.print(%1)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_mul(ir_snapshot):
    """Mul lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = toy.mul(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_add(ir_snapshot):
    """Add lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<memory.Shape<2>([2, 2]), F64> = toy.add(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_print(ir_snapshot):
    """Print maps to print_memref."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_3d_constant(ir_snapshot):
    """3D tensor constant passes through as-is."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : Nil = toy.print(%0)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_3d_add(ir_snapshot):
    """3D add lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = toy.add(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_3d_mul(ir_snapshot):
    """3D mul lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<memory.Shape<3>([2, 2, 2]), F64> = toy.mul(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot


def test_full_example(ir_snapshot):
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = strip_prefix("""
        | import function
        | import toy
        |
        | %main : function.Function<Nil> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([3, 2]), F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<memory.Shape<2>([3, 2]), F64> = toy.transpose(%2)
        |     %4 : toy.Tensor<memory.Shape<2>([3, 2]), F64> = toy.mul(%1, %3)
        |     %5 : Nil = toy.print(%4)
    """)
    m = parse_module(ir_text)
    assert lower_to_affine(m) == ir_snapshot
