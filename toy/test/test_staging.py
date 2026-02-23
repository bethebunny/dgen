"""Staging tests: dependent types that require compile-time evaluation.

These tests describe behavior that the staging evaluator should provide.
Shape inference alone cannot resolve them because the shape depends on
a computed value, not a constant literal.
"""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.codegen import compile_and_run
from dgen.staging import compile_and_run_staged, evaluate_stage0
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine
from toy.test.helpers import strip_prefix


def _stage_lower(m):
    return lower_to_llvm(lower_to_affine(m))


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
    staged = evaluate_stage0(module, _stage_lower)
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
    staged = evaluate_stage0(module, _stage_lower)
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
    staged = evaluate_stage0(module, _stage_lower)
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
    staged = evaluate_stage0(module, _stage_lower)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([4, 3], f64) = toy.mul" in out


def test_tile_nonzero_count():
    """tile count = nonzero_count([1.0, 0.0, 3.0, 0.0]) -> 2."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([4], f64) = [1.0, 0.0, 3.0, 0.0]
        |     %1 : index = toy.nonzero_count(%0)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module, _stage_lower)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([2, 3], f64) = toy.tile" in out


def test_tile_nonzero_plus_add():
    """tile count = add_index(nonzero_count([1.0, 0.0, 3.0, 0.0]), 1) -> 3."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([4], f64) = [1.0, 0.0, 3.0, 0.0]
        |     %1 : index = toy.nonzero_count(%0)
        |     %2 : index = 1
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.Tensor([2], f64) = [5.0, 6.0]
        |     %5 : toy.InferredShapeTensor(f64) = toy.tile(%4, %3)
        |     %6 : () = toy.print(%5)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module, _stage_lower)
    result = infer_shapes(staged)
    out = asm.format(result)
    assert "toy.Tensor([3, 2], f64) = toy.tile" in out


# ===----------------------------------------------------------------------=== #
# Phase 2: TileOp lowering — full pipeline with JIT execution
# ===----------------------------------------------------------------------=== #


def test_tile_constant():
    """tile([1.0, 2.0, 3.0], 2) prints two copies of the tensor."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [1.0, 2.0, 3.0]
        |     %1 : index = 2
        |     %2 : toy.Tensor([2, 3], f64) = toy.tile(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    typed = infer_shapes(module)
    affine = lower_to_affine(typed)
    ll = lower_to_llvm(affine)
    output = compile_and_run(ll, capture_output=True)
    assert output is not None
    assert output.strip() == "1, 2, 3, 1, 2, 3"


def test_tile_constant_3copies():
    """tile([1.0, 2.0], 3) prints three copies."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([2], f64) = [4.0, 5.0]
        |     %1 : index = 3
        |     %2 : toy.Tensor([3, 2], f64) = toy.tile(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    typed = infer_shapes(module)
    affine = lower_to_affine(typed)
    ll = lower_to_llvm(affine)
    output = compile_and_run(ll, capture_output=True)
    assert output is not None
    assert output.strip() == "4, 5, 4, 5, 4, 5"


def test_tile_staged_constant():
    """tile with computed count (add_index) through full JIT pipeline."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %1 : index = 1
        |     %2 : index = 1
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.InferredShapeTensor(f64) = toy.tile(%0, %3)
        |     %5 : () = toy.print(%4)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module, _stage_lower)
    typed = infer_shapes(staged)
    affine = lower_to_affine(typed)
    ll = lower_to_llvm(affine)
    output = compile_and_run(ll, capture_output=True)
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


# ===----------------------------------------------------------------------=== #
# Phase 3: Stage-1 JIT — runtime-dependent comptime values
# ===----------------------------------------------------------------------=== #


def test_stage1_nonzero_count():
    """nonzero_count on a function parameter — stage-1 JIT resolves tile shape."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %1 : index = toy.nonzero_count(%x)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    tensor = [1.0, 0.0, 3.0, 0.0]  # 2 nonzero elements
    output = compile_and_run_staged(
        parse_module(ir),
        infer=infer_shapes,
        lower=_stage_lower,
        args=[tensor],
    )
    # tile([7,8,9], 2) → print 7, 8, 9, 7, 8, 9
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


def test_stage1_nonzero_count_different_input():
    """Different tensor, different nonzero count — stage-1 adapts."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %1 : index = toy.nonzero_count(%x)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    tensor = [1.0, 2.0, 3.0, 4.0]  # 4 nonzero elements
    output = compile_and_run_staged(
        parse_module(ir),
        infer=infer_shapes,
        lower=_stage_lower,
        args=[tensor],
    )
    # tile([7,8,9], 4) → 4 copies
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9"


# ===----------------------------------------------------------------------=== #
# Phase 4: Stage-0 nonzero_count full pipeline
# ===----------------------------------------------------------------------=== #


def test_stage0_nonzero_count_jit():
    """nonzero_count on a constant tensor — stage-0 JIT through full pipeline."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([4], f64) = [1.0, 0.0, 3.0, 0.0]
        |     %1 : index = toy.nonzero_count(%0)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    staged = evaluate_stage0(module, _stage_lower)
    typed = infer_shapes(staged)
    affine = lower_to_affine(typed)
    ll = lower_to_llvm(affine)
    output = compile_and_run(ll, capture_output=True)
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


# ===----------------------------------------------------------------------=== #
# Phase 5: Staging extensions — currently expected to fail
# ===----------------------------------------------------------------------=== #

import pytest


def test_stage1_param_in_stage2():
    """Stage-2 code uses the original function parameter, not just the comptime scalar."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %1 : index = toy.nonzero_count(%x)
        |     %2 : toy.InferredShapeTensor(f64) = toy.tile(%x, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return()
    """)
    tensor = [1.0, 0.0, 3.0, 0.0]  # 2 nonzero elements
    output = compile_and_run_staged(
        parse_module(ir),
        infer=infer_shapes,
        lower=_stage_lower,
        args=[tensor],
    )
    # tile([1,0,3,0], 2) → 2 copies of the 4-element input
    assert output is not None
    assert output.strip() == "1, 0, 3, 0, 1, 0, 3, 0"


def test_stage1_two_tiles():
    """Two TileOps with independent runtime-dependent counts in the same function."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64), %y: toy.Tensor([3], f64)) -> ():
        |     %c1 : index = toy.nonzero_count(%x)
        |     %d1 : toy.Tensor([2], f64) = [1.0, 2.0]
        |     %t1 : toy.InferredShapeTensor(f64) = toy.tile(%d1, %c1)
        |     %p1 : () = toy.print(%t1)
        |     %c2 : index = toy.nonzero_count(%y)
        |     %d2 : toy.Tensor([2], f64) = [3.0, 4.0]
        |     %t2 : toy.InferredShapeTensor(f64) = toy.tile(%d2, %c2)
        |     %p2 : () = toy.print(%t2)
        |     %_ : () = return()
    """)
    tensor_x = [1.0, 0.0, 3.0, 0.0]  # 2 nonzero
    tensor_y = [1.0, 2.0, 0.0]  # 2 nonzero
    output = compile_and_run_staged(
        parse_module(ir),
        infer=infer_shapes,
        lower=_stage_lower,
        args=[tensor_x, tensor_y],
    )
    assert output is not None
    lines = output.strip().split("\n")
    assert lines[0] == "1, 2, 1, 2"
    assert lines[1] == "3, 4, 3, 4"


def test_stage1_chained_nonzero_tile():
    """Second tile count depends on shape resolved by the first tile."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %c1 : index = toy.nonzero_count(%x)
        |     %base : toy.Tensor([2], f64) = [10.0, 20.0]
        |     %t1 : toy.InferredShapeTensor(f64) = toy.tile(%base, %c1)
        |     %len : index = toy.dim_size(%t1, 0)
        |     %d2 : toy.Tensor([1], f64) = [5.0]
        |     %t2 : toy.InferredShapeTensor(f64) = toy.tile(%d2, %len)
        |     %p : () = toy.print(%t2)
        |     %_ : () = return()
    """)
    tensor = [1.0, 0.0, 3.0, 0.0]  # 2 nonzero → c1=2, t1=[2,2], len=2
    output = compile_and_run_staged(
        parse_module(ir),
        infer=infer_shapes,
        lower=_stage_lower,
        args=[tensor],
    )
    # tile([5.0], 2) → "5, 5"
    assert output is not None
    assert output.strip() == "5, 5"
