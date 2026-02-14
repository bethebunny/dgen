"""Ch3 tests: IR optimization passes."""

from toy_python import asm
from toy_python.passes.optimize import optimize
from toy_python.ir_parser import parse_module


def test_transpose_elimination():
    """transpose(transpose(x)) -> x"""
    ir_text = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<*xf64>\n"
        "    %2 = Transpose(%1) : tensor<*xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return"
    )
    assert result == expected


def test_reshape_of_matching_constant():
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return"
    )
    assert result == expected


def test_constant_folding_reshape():
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %1 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return"
    )
    assert result == expected


def test_dce():
    """Dead code elimination removes unused ops."""
    ir_text = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Constant(<2x3> [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) : tensor<2x3xf64>\n"
        "    %2 = Transpose(%1) : tensor<*xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return"
    )
    assert result == expected


def test_full_pipeline():
    """Full optimization on multiply_transpose-like example."""
    ir_text = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = Reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = Transpose(%1) : tensor<*xf64>\n"
        "    %5 = Transpose(%3) : tensor<*xf64>\n"
        "    %6 = Mul(%4, %5) : tensor<*xf64>\n"
        "    %7 = Transpose(%3) : tensor<*xf64>\n"
        "    %8 = Transpose(%1) : tensor<*xf64>\n"
        "    %9 = Mul(%7, %8) : tensor<*xf64>\n"
        "    Print(%9)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    opt = optimize(m)
    result = asm.format(opt)
    expected = (
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %7 = Transpose(%3) : tensor<*xf64>\n"
        "    %8 = Transpose(%0) : tensor<*xf64>\n"
        "    %9 = Mul(%7, %8) : tensor<*xf64>\n"
        "    Print(%9)\n"
        "    return"
    )
    assert result == expected
