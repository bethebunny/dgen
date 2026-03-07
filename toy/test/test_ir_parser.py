"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.transpose(%a)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[2, 3], F64>>() (%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.Tensor<[2, 3], F64> = toy.reshape(%a)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[2, 3], F64>>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_explicit_constant():
    """Explicit constant(...) syntax is accepted and normalizes to implicit form."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[2, 3], F64>>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    expected = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[2, 3], F64>>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ : Nil = return(%0)
    """)
    assert asm.format(module) == expected


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.mul(%a, %b)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.add(%a, %b)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_generic_call():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.generic_call<"helper">([%a])
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<Nil>() (%a: toy.Tensor<[2, 3], F64>):
        |     %_ : Nil = toy.print(%a)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_void_return():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() ():
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_concat():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[2, 8], F64>>() (%a: toy.Tensor<[2, 3], F64>, %b: toy.Tensor<[2, 5], F64>):
        |     %0 : toy.Tensor<[2, 8], F64> = toy.concat<1>(%a, %b)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_tile():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[4, 3], F64>>() (%a: toy.Tensor<[3], F64>, %n: Index):
        |     %0 : toy.Tensor<[4, 3], F64> = toy.tile<%a>(%n)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_tile_with_index_constant():
    """Tile where count is an index constant — shape inference can peek through."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<[4, 3], F64>>() (%a: toy.Tensor<[3], F64>):
        |     %0 : Index = 4
        |     %1 : toy.Tensor<[4, 3], F64> = toy.tile<%a>(%0)
        |     %_ : Nil = return(%1)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_tile_with_computed_count():
    """Tile where count is computed — shape inference CANNOT resolve without evaluation."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.Tensor<[3], F64>):
        |     %0 : Index = 2
        |     %1 : Index = 2
        |     %2 : Index = add_index(%0, %1)
        |     %3 : toy.InferredShapeTensor<F64> = toy.tile<%a>(%2)
        |     %_ : Nil = return(%3)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add_index():
    ir = strip_prefix("""
        | %f : Nil = function<Index>() (%x: Index, %y: Index):
        |     %0 : Index = add_index(%x, %y)
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_full_program():
    ir = strip_prefix("""
        | import toy
        |
        | %multiply_transpose : Nil = function<toy.InferredShapeTensor<F64>>() (%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.transpose(%a)
        |     %1 : toy.InferredShapeTensor<F64> = toy.transpose(%b)
        |     %2 : toy.InferredShapeTensor<F64> = toy.mul(%0, %1)
        |     %_ : Nil = return(%2)
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
        |     %2 : toy.Tensor<[6], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<[2, 3], F64> = toy.reshape(%2)
        |     %4 : toy.InferredShapeTensor<F64> = toy.generic_call<"multiply_transpose">([%1, %3])
        |     %5 : toy.InferredShapeTensor<F64> = toy.generic_call<"multiply_transpose">([%3, %1])
        |     %_ : Nil = toy.print(%5)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
