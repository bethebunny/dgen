"""Staging tests: dependent types that require compile-time evaluation.

These tests describe behavior that the staging evaluator should provide.
Shape inference alone cannot resolve them because the shape depends on
a computed value, not a constant literal.
"""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.staging import evaluate_stage0
from toy.passes.shape_inference import infer_shapes
from toy.test.helpers import strip_prefix


def test_tile_add_index():
    """tile count = add_index(2, 2) should resolve to Tensor([4, 3])."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [1.0, 2.0, 3.0]
        |     %1 : index = 2
        |     %2 : index = 2
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.InferredShapeTensor(f64) = toy.tile(%0, %3)
        |     %5 : () = toy.print(%4)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([4, 3], f64) = toy.tile" in out


def test_tile_chained_add():
    """tile count = add_index(add_index(1, 1), add_index(1, 1)) -> 4."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [1.0, 2.0, 3.0]
        |     %1 : index = 1
        |     %2 : index = 1
        |     %3 : index = add_index(%1, %2)
        |     %4 : index = 1
        |     %5 : index = 1
        |     %6 : index = add_index(%4, %5)
        |     %7 : index = add_index(%3, %6)
        |     %8 : toy.InferredShapeTensor(f64) = toy.tile(%0, %7)
        |     %9 : () = toy.print(%8)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([4, 3], f64) = toy.tile" in out


def test_concat_after_computed_tile():
    """concat with a tile whose count is computed — both shapes need evaluation.

    tile(Tensor([3]), add_index(2, 1)) -> Tensor([3, 3])
    concat(Tensor([2, 3]), Tensor([3, 3]), axis=0) -> Tensor([5, 3])
    """
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %2 : index = 2
        |     %3 : index = 1
        |     %4 : index = add_index(%2, %3)
        |     %5 : toy.InferredShapeTensor(f64) = toy.tile(%1, %4)
        |     %6 : toy.InferredShapeTensor(f64) = toy.concat(%0, %5, 0)
        |     %7 : () = toy.print(%6)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([5, 3], f64) = toy.concat" in out


def test_tile_shape_propagates_to_mul():
    """Shape from computed tile propagates to downstream ops.

    tile(Tensor([3]), add_index(2, 2)) -> Tensor([4, 3])
    mul(Tensor([4, 3]), Tensor([4, 3])) -> Tensor([4, 3])
    """
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [1.0, 2.0, 3.0]
        |     %1 : index = 2
        |     %2 : index = 2
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.InferredShapeTensor(f64) = toy.tile(%0, %3)
        |     %5 : toy.InferredShapeTensor(f64) = toy.tile(%0, %3)
        |     %6 : toy.InferredShapeTensor(f64) = toy.mul(%4, %5)
        |     %7 : () = toy.print(%6)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([4, 3], f64) = toy.mul" in out
