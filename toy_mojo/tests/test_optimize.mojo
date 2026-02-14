"""Ch3 tests: IR optimization passes."""

from testing import assert_equal, TestSuite

from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
)
from toy.dialects.toy_printer import print_module
from toy.passes.optimize import optimize
from toy.ir_parser import parse_module
from collections import Optional


fn make_module(var func: FuncOp) -> Module:
    var funcs = List[FuncOp]()
    funcs.append(func^)
    return Module(functions=funcs^)


fn unranked() -> AnyToyType:
    return AnyToyType(UnrankedTensorType())


fn ranked(var shape: List[Int]) -> AnyToyType:
    return AnyToyType(RankedTensorType(shape=shape^))


def test_transpose_elimination():
    """transpose(transpose(x)) -> x"""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<*xf64>\n"
        "    %2 = Transpose(%1) : tensor<*xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var opt = optimize(m)
    var result = print_module(opt)
    var expected = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    assert_equal(result, expected)


def test_reshape_of_matching_constant():
    """Reshape to same shape as constant -> remove reshape."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var opt = optimize(m)
    var result = print_module(opt)
    var expected = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    assert_equal(result, expected)


def test_constant_folding_reshape():
    """Reshape of constant with different shape -> fold into new constant."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var opt = optimize(m)
    var result = print_module(opt)
    var expected = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %1 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    assert_equal(result, expected)


def test_dce():
    """Dead code elimination removes unused ops."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Constant(<2x3> [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) : tensor<2x3xf64>\n"
        "    %2 = Transpose(%1) : tensor<*xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var opt = optimize(m)
    var result = print_module(opt)
    var expected = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    assert_equal(result, expected)


def test_full_pipeline():
    """Full optimization on multiply_transpose-like example."""
    var ir_text = String(
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
    var m = parse_module(ir_text)
    var opt = optimize(m)
    var result = print_module(opt)
    # After optimization:
    # - Reshape %1 of matching constant -> removed, uses rewritten to %0
    # - Reshape %3 of different shape -> folded into new constant
    # - Transpose of constant produces ops that are used directly
    # - DCE removes unused intermediate ops
    var expected = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %7 = Transpose(%3) : tensor<*xf64>\n"
        "    %8 = Transpose(%0) : tensor<*xf64>\n"
        "    %9 = Mul(%7, %8) : tensor<*xf64>\n"
        "    Print(%9)\n"
        "    return\n"
    )
    assert_equal(result, expected)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
