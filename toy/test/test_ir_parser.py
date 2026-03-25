"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent
from toy.test.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.transpose(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([2, 3]), F64>>() body(%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = toy.reshape(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([2, 3]), F64>>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_explicit_constant(ir_snapshot):
    """Explicit constant(...) syntax is accepted and normalizes to implicit form."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([2, 3]), F64>>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    """)
    module = parse_module(ir)
    assert module == ir_snapshot


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.mul(%a, %b)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_add():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.add(%a, %b)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_call():
    ir = strip_prefix("""
        | import toy
        |
        | %helper : Nil = function<toy.InferredShapeTensor<F64>>() body(%x: toy.InferredShapeTensor<F64>):
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = call<%helper>([%a])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_print():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<Nil>() body(%a: toy.Tensor<memory.Shape<2>([2, 3]), F64>):
        |     %0 : Nil = toy.print(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_void_return():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() body():
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_concat():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([2, 8]), F64>>() body(%a: toy.Tensor<memory.Shape<2>([2, 3]), F64>, %b: toy.Tensor<memory.Shape<2>([2, 5]), F64>):
        |     %0 : toy.Tensor<memory.Shape<2>([2, 8]), F64> = toy.concat<1>(%a, %b)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_tile():
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([4, 3]), F64>>() body(%a: toy.Tensor<memory.Shape<1>([3]), F64>, %n: Index):
        |     %0 : toy.Tensor<memory.Shape<2>([4, 3]), F64> = toy.tile<%n>(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_tile_with_index_constant():
    """Tile where count is an index constant — shape inference can peek through."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.Tensor<memory.Shape<2>([4, 3]), F64>>() body(%a: toy.Tensor<memory.Shape<1>([3]), F64>):
        |     %0 : Index = 4
        |     %1 : toy.Tensor<memory.Shape<2>([4, 3]), F64> = toy.tile<%0>(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_tile_with_computed_count():
    """Tile where count is computed — shape inference CANNOT resolve without evaluation."""
    ir = strip_prefix("""
        | import toy
        |
        | %f : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.Tensor<memory.Shape<1>([3]), F64>):
        |     %0 : Index = 2
        |     %1 : Index = 2
        |     %2 : Index = add_index(%0, %1)
        |     %3 : toy.InferredShapeTensor<F64> = toy.tile<%2>(%a)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_add_index():
    ir = strip_prefix("""
        | %f : Nil = function<Index>() body(%x: Index, %y: Index):
        |     %0 : Index = add_index(%x, %y)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_full_program():
    ir = strip_prefix("""
        | import toy
        |
        | %multiply_transpose : Nil = function<toy.InferredShapeTensor<F64>>() body(%a: toy.InferredShapeTensor<F64>, %b: toy.InferredShapeTensor<F64>):
        |     %0 : toy.InferredShapeTensor<F64> = toy.transpose(%a)
        |     %1 : toy.InferredShapeTensor<F64> = toy.transpose(%b)
        |     %2 : toy.InferredShapeTensor<F64> = toy.mul(%0, %1)
        |
        | %main : Nil = function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = toy.reshape(%0)
        |     %2 : toy.Tensor<memory.Shape<1>([6]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<memory.Shape<2>([2, 3]), F64> = toy.reshape(%2)
        |     %4 : toy.InferredShapeTensor<F64> = call<%multiply_transpose>([%1, %3])
        |     %5 : toy.InferredShapeTensor<F64> = call<%multiply_transpose>([%3, %1])
        |     %6 : toy.InferredShapeTensor<F64> = chain(%5, %4)
        |     %7 : Nil = toy.print(%6)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
