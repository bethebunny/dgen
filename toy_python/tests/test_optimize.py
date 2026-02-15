"""Ch3 tests: IR optimization passes."""

from toy_python import asm
from toy_python.passes.optimize import optimize
from toy_python.dialects.toy import parse_toy_module as parse_module
from toy_python.tests.helpers import strip_prefix


def test_transpose_elimination():
    """transpose(transpose(x)) -> x"""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.transpose(%0) : tensor<*xf64>
        |     %2 = toy.transpose(%1) : tensor<*xf64>
        |     toy.print(%2)
        |     return()
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     toy.print(%0)
        |     return()
    """)
    assert result == expected


def test_reshape_of_matching_constant():
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.reshape(%0) : tensor<2x3xf64>
        |     toy.print(%1)
        |     return()
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     toy.print(%0)
        |     return()
    """)
    assert result == expected


def test_constant_folding_reshape():
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>
        |     %1 = toy.reshape(%0) : tensor<2x3xf64>
        |     toy.print(%1)
        |     return()
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %1 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     toy.print(%1)
        |     return()
    """)
    assert result == expected


def test_dce():
    """Dead code elimination removes unused ops."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.constant(<2x3>, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) : tensor<2x3xf64>
        |     %2 = toy.transpose(%1) : tensor<*xf64>
        |     toy.print(%0)
        |     return()
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     toy.print(%0)
        |     return()
    """)
    assert result == expected


def test_full_pipeline():
    """Full optimization on multiply_transpose-like example."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.reshape(%0) : tensor<2x3xf64>
        |     %2 = toy.constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>
        |     %3 = toy.reshape(%2) : tensor<2x3xf64>
        |     %4 = toy.transpose(%1) : tensor<*xf64>
        |     %5 = toy.transpose(%3) : tensor<*xf64>
        |     %6 = toy.mul(%4, %5) : tensor<*xf64>
        |     %7 = toy.transpose(%3) : tensor<*xf64>
        |     %8 = toy.transpose(%1) : tensor<*xf64>
        |     %9 = toy.mul(%7, %8) : tensor<*xf64>
        |     toy.print(%9)
        |     return()
    """)
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %3 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %7 = toy.transpose(%3) : tensor<*xf64>
        |     %8 = toy.transpose(%0) : tensor<*xf64>
        |     %9 = toy.mul(%7, %8) : tensor<*xf64>
        |     toy.print(%9)
        |     return()
    """)
    assert result == expected
