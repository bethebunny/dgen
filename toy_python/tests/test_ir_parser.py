"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from toy_python.asm.parser import parse_module
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.transpose(%a) : tensor<*xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<2x3xf64>:
        |     %0 = toy.reshape(%a) : tensor<2x3xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> tensor<2x3xf64>:
        |     %0 = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.mul(%a, %b) : tensor<*xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.add(%a, %b) : tensor<*xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_generic_call():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.generic_call(@helper, [%a]) : tensor<*xf64>
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> ():
        |     %_ = toy.print(%a)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_void_return():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_full_program():
    ir = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.transpose(%a) : tensor<*xf64>
        |     %1 = toy.transpose(%b) : tensor<*xf64>
        |     %2 = toy.mul(%0, %1) : tensor<*xf64>
        |     %_ = return(%2)
        |
        | %main = function () -> ():
        |     %0 = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.reshape(%0) : tensor<2x3xf64>
        |     %2 = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>
        |     %3 = toy.reshape(%2) : tensor<2x3xf64>
        |     %4 = toy.generic_call(@multiply_transpose, [%1, %3]) : tensor<*xf64>
        |     %5 = toy.generic_call(@multiply_transpose, [%3, %1]) : tensor<*xf64>
        |     %_ = toy.print(%5)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
