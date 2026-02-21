"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.InferredShapeTensor[f64]) -> toy.InferredShapeTensor[f64]:
        |     %0 : toy.InferredShapeTensor[f64] = toy.transpose(%a)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.InferredShapeTensor[f64]) -> toy.Tensor[(2, 3), f64]:
        |     %0 : toy.Tensor[(2, 3), f64] = toy.reshape(%a)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> toy.Tensor[(2, 3), f64]:
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_explicit_constant():
    """Explicit constant(...) syntax is accepted and normalizes to implicit form."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> toy.Tensor[(2, 3), f64]:
        |     %0 : toy.Tensor[(2, 3), f64] = constant((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    expected = strip_prefix("""
        | import toy
        |
        | %f = function () -> toy.Tensor[(2, 3), f64]:
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %_ : () = return(%0)
    """)
    assert asm.format(module) == expected


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.InferredShapeTensor[f64], %b: toy.InferredShapeTensor[f64]) -> toy.InferredShapeTensor[f64]:
        |     %0 : toy.InferredShapeTensor[f64] = toy.mul(%a, %b)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.InferredShapeTensor[f64], %b: toy.InferredShapeTensor[f64]) -> toy.InferredShapeTensor[f64]:
        |     %0 : toy.InferredShapeTensor[f64] = toy.add(%a, %b)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_generic_call():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.InferredShapeTensor[f64]) -> toy.InferredShapeTensor[f64]:
        |     %0 : toy.InferredShapeTensor[f64] = toy.generic_call(@helper, [%a])
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print():
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%a: toy.Tensor[(2, 3), f64]) -> ():
        |     %_ : () = toy.print(%a)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_void_return():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_full_program():
    ir = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.InferredShapeTensor[f64], %b: toy.InferredShapeTensor[f64]) -> toy.InferredShapeTensor[f64]:
        |     %0 : toy.InferredShapeTensor[f64] = toy.transpose(%a)
        |     %1 : toy.InferredShapeTensor[f64] = toy.transpose(%b)
        |     %2 : toy.InferredShapeTensor[f64] = toy.mul(%0, %1)
        |     %_ : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %1 : toy.Tensor[(2, 3), f64] = toy.reshape(%0)
        |     %2 : toy.Tensor[(6), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %3 : toy.Tensor[(2, 3), f64] = toy.reshape(%2)
        |     %4 : toy.InferredShapeTensor[f64] = toy.generic_call(@multiply_transpose, [%1, %3])
        |     %5 : toy.InferredShapeTensor[f64] = toy.generic_call(@multiply_transpose, [%3, %1])
        |     %_ : () = toy.print(%5)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
