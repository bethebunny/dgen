"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from testing import assert_equal, TestSuite

from toy.ir_parser import parse_module
from toy.printer import print_module


def test_roundtrip_transpose():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_reshape():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>) -> tensor<2x3xf64>:\n"
        "    %0 = Reshape(%a) : tensor<2x3xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_constant():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function () -> tensor<2x3xf64>:\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_mul():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Mul(%a, %b) : tensor<*xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_add():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Add(%a, %b) : tensor<*xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_generic_call():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = GenericCall @helper(%a) : tensor<*xf64>\n"
        "    return %0\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_print():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function (%a: tensor<*xf64>):\n"
        "    Print(%a)\n"
        "    return\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_void_return():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%f = function ():\n"
        "    return\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def test_roundtrip_full_program():
    var ir = String(
        "from toy use *\n"
        "\n"
        "%multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    %1 = Transpose(%b) : tensor<*xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    return %2\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = Reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>\n"
        "    %5 = GenericCall @multiply_transpose(%3, %1) : tensor<*xf64>\n"
        "    Print(%5)\n"
        "    return\n"
    )
    var module = parse_module(ir)
    assert_equal(print_module(module), ir)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
