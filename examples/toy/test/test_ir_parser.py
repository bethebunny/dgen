"""Phase 2 tests: parse IR text -> reprint -> compare (round-trip)."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.testing import assert_ir_equivalent
from toy.test.helpers import strip_prefix


def test_roundtrip_transpose():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        | import toy
        |
        | %f : function.Function<[toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>):
        |     %0 : toy.InferredShapeTensor<number.Float64> = toy.transpose(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_reshape():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.InferredShapeTensor<number.Float64>], toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>):
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_constant():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[], toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_explicit_constant(ir_snapshot):
    """Explicit constant(...) syntax is accepted and normalizes to implicit form."""
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[], toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    """)
    value = parse(ir)
    assert value == ir_snapshot


def test_roundtrip_mul():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        | import toy
        |
        | %f : function.Function<[toy.InferredShapeTensor<number.Float64>, toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>, %b: toy.InferredShapeTensor<number.Float64>):
        |     %0 : toy.InferredShapeTensor<number.Float64> = toy.mul(%a, %b)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_add():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        | import toy
        |
        | %f : function.Function<[toy.InferredShapeTensor<number.Float64>, toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>, %b: toy.InferredShapeTensor<number.Float64>):
        |     %0 : toy.InferredShapeTensor<number.Float64> = toy.add(%a, %b)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_call():
    ir = strip_prefix("""
        | import function
        | import index
        | import number
        | import toy
        |
        | %helper : function.Function<[toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%x: toy.InferredShapeTensor<number.Float64>):
        |     %_ : Nil = ()
        |
        | %f : function.Function<[toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>) captures(%helper):
        |     %0 : toy.InferredShapeTensor<number.Float64> = function.call(%helper, [%a])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_print():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>], Nil> = function.function<Nil>() body(%a: toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>):
        |     %0 : Nil = toy.print(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_void_return():
    ir = strip_prefix("""
        | import function
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %_ : Nil = ()
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_concat():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>, toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 5]), number.Float64>], toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 8]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 8]), number.Float64>>() body(%a: toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64>, %b: toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 5]), number.Float64>):
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 8]), number.Float64> = toy.concat<index.Index(1)>(%a, %b)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_tile():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>, index.Index], toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64>>() body(%a: toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>, %n: index.Index):
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64> = toy.tile<%n>(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_tile_with_index_constant():
    """Tile where count is an index constant — shape inference can peek through."""
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>], toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64>> = function.function<toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64>>() body(%a: toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>):
        |     %0 : index.Index = 4
        |     %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([4, 3]), number.Float64> = toy.tile<%0>(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_tile_with_computed_count():
    """Tile where count is computed — shape inference CANNOT resolve without evaluation."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.Tensor<ndbuffer.Shape<index.Index(1)>([3]), number.Float64>):
        |     %0 : index.Index = 2
        |     %1 : index.Index = 2
        |     %2 : index.Index = algebra.add(%0, %1)
        |     %3 : toy.InferredShapeTensor<number.Float64> = toy.tile<%2>(%a)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_add_index():
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %f : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: index.Index):
        |     %0 : index.Index = algebra.add(%x, %y)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_full_program():
    ir = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %multiply_transpose : function.Function<[toy.InferredShapeTensor<number.Float64>, toy.InferredShapeTensor<number.Float64>], toy.InferredShapeTensor<number.Float64>> = function.function<toy.InferredShapeTensor<number.Float64>>() body(%a: toy.InferredShapeTensor<number.Float64>, %b: toy.InferredShapeTensor<number.Float64>):
        |     %0 : toy.InferredShapeTensor<number.Float64> = toy.transpose(%a)
        |     %1 : toy.InferredShapeTensor<number.Float64> = toy.transpose(%b)
        |     %2 : toy.InferredShapeTensor<number.Float64> = toy.mul(%0, %1)
        |
        | %main : function.Function<[], Nil> = function.function<Nil>() body() captures(%multiply_transpose):
        |     %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%0)
        |     %2 : toy.Tensor<ndbuffer.Shape<index.Index(1)>([6]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%2)
        |     %4 : toy.InferredShapeTensor<number.Float64> = function.call(%multiply_transpose, [%1, %3])
        |     %5 : toy.InferredShapeTensor<number.Float64> = function.call(%multiply_transpose, [%3, %1])
        |     %6 : toy.InferredShapeTensor<number.Float64> = chain(%5, %4)
        |     %7 : Nil = toy.print(%6)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
