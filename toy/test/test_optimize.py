"""Ch3 tests: IR optimization passes."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.passes.optimize import optimize
from toy.test.helpers import strip_prefix


def test_transpose_elimination():
    """transpose(transpose(x)) -> x"""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<[2, 3], F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%2)
        |     %_ : Nil = return(%3)
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : Nil = toy.print(%0)
        |     %_ : Nil = return(%3)
    """)
    assert result == expected


def test_reshape_of_matching_constant():
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
        |     %2 : Nil = toy.print(%1)
        |     %_ : Nil = return(%2)
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : Nil = toy.print(%0)
        |     %_ : Nil = return(%2)
    """)
    assert result == expected


def test_constant_folding_reshape():
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[6], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
        |     %2 : Nil = toy.print(%1)
        |     %_ : Nil = return(%2)
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : Nil = toy.print(%0)
        |     %_ : Nil = return(%2)
    """)
    assert result == expected


def test_dce():
    """Dead code elimination removes unused ops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %2 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%0)
        |     %_ : Nil = return(%3)
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : Nil = toy.print(%0)
        |     %_ : Nil = return(%3)
    """)
    assert result == expected


def test_full_pipeline():
    """Full optimization on multiply_transpose-like example."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
        |     %2 : toy.Tensor<[6], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<[2, 3], F64> = toy.reshape(%2)
        |     %4 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
        |     %5 : toy.Tensor<[3, 2], F64> = toy.transpose(%3)
        |     %6 : toy.Tensor<[3, 2], F64> = toy.mul(%4, %5)
        |     %7 : toy.Tensor<[3, 2], F64> = toy.transpose(%3)
        |     %8 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
        |     %9 : toy.Tensor<[3, 2], F64> = toy.mul(%7, %8)
        |     %10 : Nil = toy.print(%9)
        |     %_ : Nil = return(%10)
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %7 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %8 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %9 : toy.Tensor<[3, 2], F64> = toy.mul(%7, %8)
        |     %10 : Nil = toy.print(%9)
        |     %_ : Nil = return(%10)
    """)
    assert result == expected
