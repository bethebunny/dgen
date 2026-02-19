"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from toy_python.asm.parser import parse_module
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 : tensor<*xf64> = toy.transpose(%a)
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<2x3xf64>:
        |     %0 : tensor<2x3xf64> = toy.reshape(%a)
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> tensor<2x3xf64>:
        |     %0 : tensor<2x3xf64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 : tensor<*xf64> = toy.mul(%a, %b)
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 : tensor<*xf64> = toy.add(%a, %b)
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_generic_call():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 : tensor<*xf64> = toy.generic_call(@helper, [%a])
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
        |     %0 : tensor<*xf64> = toy.transpose(%a)
        |     %1 : tensor<*xf64> = toy.transpose(%b)
        |     %2 : tensor<*xf64> = toy.mul(%0, %1)
        |     %_ = return(%2)
        |
        | %main = function () -> ():
        |     %0 : tensor<2x3xf64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %1 : tensor<2x3xf64> = toy.reshape(%0)
        |     %2 : tensor<6xf64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %3 : tensor<2x3xf64> = toy.reshape(%2)
        |     %4 : tensor<*xf64> = toy.generic_call(@multiply_transpose, [%1, %3])
        |     %5 : tensor<*xf64> = toy.generic_call(@multiply_transpose, [%3, %1])
        |     %_ = toy.print(%5)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
