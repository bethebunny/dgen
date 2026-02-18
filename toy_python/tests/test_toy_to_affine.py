"""Ch5 tests: Toy IR to Affine IR lowering."""

from toy_python.asm.parser import parse_module
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_simple_constant():
    """Constant op lowers to alloc + stores."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.alloc(<2x3>)" in result, "Should have alloc"
    assert "affine.store" in result, "Should have stores"
    assert "affine.arith_constant" in result, "Should have constants"
    assert "affine.print_memref" in result, "Should have print"
    assert "affine.dealloc" in result, "Should have dealloc"
    assert "return" in result, "Should have return"


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.transpose(%0) : tensor<3x2xf64>
        |     %_ = toy.print(%1)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("affine.alloc(")
    assert alloc_count >= 2, "Should have at least 2 allocs"
    assert "affine.load" in result, "Should have loads for transpose"
    assert "affine.alloc(<3x2>)" in result, "Should have 3x2 alloc for transposed result"


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x2>, [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>
        |     %1 = toy.constant(<2x2>, [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>
        |     %2 = toy.mul(%0, %1) : tensor<2x2xf64>
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.mul_f" in result, "Should have mul_f op"
    alloc_count = result.count("affine.alloc(")
    assert alloc_count >= 3, "Should have 3 allocs (2 constants + 1 result)"


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x2>, [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>
        |     %1 = toy.constant(<2x2>, [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>
        |     %2 = toy.add(%0, %1) : tensor<2x2xf64>
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.add_f" in result, "Should have add_f op"


def test_print():
    """Print maps to print_memref."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.print_memref" in result, "Should have print_memref"


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.transpose(%0) : tensor<3x2xf64>
        |     %2 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %3 = toy.transpose(%2) : tensor<3x2xf64>
        |     %4 = toy.mul(%1, %3) : tensor<3x2xf64>
        |     %_ = toy.print(%4)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("affine.alloc(")
    assert alloc_count >= 5, "Should have at least 5 allocs"
    assert "affine.mul_f" in result, "Should have mul_f"
    assert "affine.load" in result, "Should have loads"
    assert "affine.store" in result, "Should have stores"
    assert "affine.print_memref" in result, "Should have print"
    assert "return" in result, "Should have return"
