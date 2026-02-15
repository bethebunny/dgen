"""Ch5 tests: Toy IR to Affine IR lowering."""

from toy_python.ir_parser import parse_module
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python import asm


def test_simple_constant():
    """Constant op lowers to alloc + stores."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "Alloc<2x3>()" in result, "Should have alloc"
    assert "AffineFor" in result, "Should have for loop"
    assert "AffineStore" in result, "Should have stores"
    assert "ArithConstant" in result, "Should have constants"
    assert "PrintMemRef" in result, "Should have print"
    assert "Dealloc" in result, "Should have dealloc"
    assert "return" in result, "Should have return"


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("Alloc<")
    assert alloc_count >= 2, "Should have at least 2 allocs"
    assert "AffineLoad" in result, "Should have loads for transpose"
    assert "Alloc<3x2>()" in result, "Should have 3x2 alloc for transposed result"


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x2> [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = Constant(<2x2> [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<2x2xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "MulF" in result, "Should have MulF op"
    alloc_count = result.count("Alloc<")
    assert alloc_count >= 3, "Should have 3 allocs (2 constants + 1 result)"


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x2> [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = Constant(<2x2> [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = Add(%0, %1) : tensor<2x2xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "AddF" in result, "Should have AddF op"


def test_print():
    """Print maps to PrintMemRef."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "PrintMemRef" in result, "Should have PrintMemRef"


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    %2 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = Transpose(%2) : tensor<3x2xf64>\n"
        "    %4 = Mul(%1, %3) : tensor<3x2xf64>\n"
        "    Print(%4)\n"
        "    return\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("Alloc<")
    assert alloc_count >= 5, "Should have at least 5 allocs"
    assert "MulF" in result, "Should have MulF"
    assert "AffineLoad" in result, "Should have loads"
    assert "AffineStore" in result, "Should have stores"
    assert "PrintMemRef" in result, "Should have print"
    assert "return" in result, "Should have return"
