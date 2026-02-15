"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from toy_python.dialects.toy import parse_toy_module as parse_module
from toy_python import asm


def test_roundtrip_transpose():
    ir = (
        "%f = function (%a: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_reshape():
    ir = (
        "%f = function (%a: tensor<*xf64>) -> tensor<2x3xf64>:\n"
        "    %0 = Reshape(%a) : tensor<2x3xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_constant():
    ir = (
        "%f = function () -> tensor<2x3xf64>:\n"
        "    %0 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_mul():
    ir = (
        "%f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Mul(%a, %b) : tensor<*xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add():
    ir = (
        "%f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Add(%a, %b) : tensor<*xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_generic_call():
    ir = (
        "%f = function (%a: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = GenericCall(@helper, [%a]) : tensor<*xf64>\n"
        "    Return(%0)\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print():
    ir = (
        "%f = function (%a: tensor<*xf64>) -> ():\n"
        "    Print(%a)\n"
        "    Return()\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_void_return():
    ir = (
        "%f = function () -> ():\n"
        "    Return()\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_full_program():
    ir = (
        "%multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    %1 = Transpose(%b) : tensor<*xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    Return(%2)\n"
        "\n"
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = Constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = Reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = GenericCall(@multiply_transpose, [%1, %3]) : tensor<*xf64>\n"
        "    %5 = GenericCall(@multiply_transpose, [%3, %1]) : tensor<*xf64>\n"
        "    Print(%5)\n"
        "    Return()\n"
    )
    module = parse_module(ir)
    assert asm.format(module) == ir
