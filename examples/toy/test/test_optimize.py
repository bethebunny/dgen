"""Ch3 tests: IR optimization passes."""

import pytest

import dgen
from dgen import asm
from dgen.passes.compiler import Compiler, IdentityPass
from toy.passes.optimize import ToyOptimize
from toy.test.helpers import strip_prefix


def optimize(m: dgen.Value) -> dgen.Value:
    return Compiler([ToyOptimize()], IdentityPass()).run(m)


def test_transpose_elimination(ir_snapshot):
    """transpose(transpose(x)) -> x"""
    ir_text = strip_prefix("""
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%0)
        | %2 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.transpose(%1)
        | %3 : Nil = toy.print(%2)
    """)
    m = asm.parse(ir_text)
    assert optimize(m) == ir_snapshot


@pytest.mark.skip(reason="Needs constant folding")
def test_reshape_of_matching_constant(ir_snapshot):
    """Reshape to same shape as constant -> remove reshape."""
    ir_text = strip_prefix("""
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%0)
        | %2 : Nil = toy.print(%1)
    """)
    m = asm.parse(ir_text)
    assert optimize(m) == ir_snapshot


@pytest.mark.skip(reason="Needs constant folding")
def test_constant_folding_reshape(ir_snapshot):
    """Reshape of constant with different shape -> fold into new constant."""
    ir_text = strip_prefix("""
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %0 : toy.Tensor<ndbuffer.Shape<index.Index(1)>([6]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%0)
        | %2 : Nil = toy.print(%1)
    """)
    m = asm.parse(ir_text)
    assert optimize(m) == ir_snapshot


def test_dce(ir_snapshot):
    """Dead code elimination removes unused ops."""
    ir_text = strip_prefix("""
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        | %2 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%1)
        | %3 : Nil = toy.print(%0)
        | %_ : Nil = chain(%3, %2)
    """)
    m = asm.parse(ir_text)
    assert optimize(m) == ir_snapshot


@pytest.mark.skip(reason="Needs constant folding")
def test_full_pipeline(ir_snapshot):
    """Full optimization on multiply_transpose-like example."""
    ir_text = strip_prefix("""
        | import ndbuffer
        | import number
        | import toy
        | import index
        |
        | %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%0)
        | %2 : toy.Tensor<ndbuffer.Shape<index.Index(1)>([6]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        | %3 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.reshape(%2)
        | %4 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%1)
        | %5 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%3)
        | %6 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.mul(%4, %5)
        | %7 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%3)
        | %8 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%1)
        | %9 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.mul(%7, %8)
        | %10 : Nil = toy.print(%9)
        | %_ : Nil = chain(%10, %6)
    """)
    m = asm.parse(ir_text)
    assert optimize(m) == ir_snapshot
