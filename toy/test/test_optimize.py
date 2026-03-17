"""Ch3 tests: IR optimization passes."""

from dgen.asm.parser import parse_module
from toy.passes.optimize import optimize
from toy.test.helpers import strip_prefix


def test_transpose_elimination(ir_snapshot):
    """transpose(transpose(x)) -> x"""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%2)
        |     %_ : Nil = return(%3)
    """)
    m = parse_module(ir_text)
    assert optimize(m) == ir_snapshot


def test_reshape_of_matching_constant(ir_snapshot):
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.reshape(%0)
        |     %2 : Nil = toy.print(%1)
        |     %_ : Nil = return(%2)
    """)
    m = parse_module(ir_text)
    assert optimize(m) == ir_snapshot


def test_constant_folding_reshape(ir_snapshot):
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<1>([6]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.reshape(%0)
        |     %2 : Nil = toy.print(%1)
        |     %_ : Nil = return(%2)
    """)
    m = parse_module(ir_text)
    assert optimize(m) == ir_snapshot


def test_dce(ir_snapshot):
    """Dead code elimination removes unused ops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %2 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%0)
        |     %_ : Nil = chain(%3, %2)
    """)
    m = parse_module(ir_text)
    assert optimize(m) == ir_snapshot


def test_full_pipeline(ir_snapshot):
    """Full optimization on multiply_transpose-like example."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.reshape(%0)
        |     %2 : toy.Tensor<affine.Shape<1>([6]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.reshape(%2)
        |     %4 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%1)
        |     %5 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%3)
        |     %6 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.mul(%4, %5)
        |     %7 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%3)
        |     %8 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.transpose(%1)
        |     %9 : toy.Tensor<affine.Shape<2>([3, 2]), F64> = toy.mul(%7, %8)
        |     %10 : Nil = toy.print(%9)
        |     %_ : Nil = chain(%10, %6)
    """)
    m = parse_module(ir_text)
    assert optimize(m) == ir_snapshot
