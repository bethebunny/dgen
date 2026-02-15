"""Ch3 tests: IR optimization passes."""

from toy_python import asm
from toy_python.passes.optimize import optimize
from toy_python.dialects.toy import parse_toy_module as parse_module


def test_transpose_elimination():
    """transpose(transpose(x)) -> x"""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = transpose(%0) : tensor<*xf64>\n"
        "    %2 = transpose(%1) : tensor<*xf64>\n"
        "    print(%2)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    assert result == expected


def test_reshape_of_matching_constant():
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = reshape(%0) : tensor<2x3xf64>\n"
        "    print(%1)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    assert result == expected


def test_constant_folding_reshape():
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %1 = reshape(%0) : tensor<2x3xf64>\n"
        "    print(%1)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "%main = function () -> ():\n"
        "    %1 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%1)\n"
        "    return()\n"
    )
    assert result == expected


def test_dce():
    """Dead code elimination removes unused ops."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = constant(<2x3>, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) : tensor<2x3xf64>\n"
        "    %2 = transpose(%1) : tensor<*xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    assert result == expected


def test_full_pipeline():
    """Full optimization on multiply_transpose-like example."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = transpose(%1) : tensor<*xf64>\n"
        "    %5 = transpose(%3) : tensor<*xf64>\n"
        "    %6 = mul(%4, %5) : tensor<*xf64>\n"
        "    %7 = transpose(%3) : tensor<*xf64>\n"
        "    %8 = transpose(%1) : tensor<*xf64>\n"
        "    %9 = mul(%7, %8) : tensor<*xf64>\n"
        "    print(%9)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %7 = transpose(%3) : tensor<*xf64>\n"
        "    %8 = transpose(%0) : tensor<*xf64>\n"
        "    %9 = mul(%7, %8) : tensor<*xf64>\n"
        "    print(%9)\n"
        "    return()\n"
    )
    assert result == expected
