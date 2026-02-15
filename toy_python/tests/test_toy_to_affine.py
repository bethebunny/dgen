"""Ch5 tests: Toy IR to Affine IR lowering."""

from toy_python.dialects.toy import parse_toy_module as parse_module
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python import asm


def test_simple_constant():
    """Constant op lowers to alloc + stores."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "alloc(<2x3>)" in result, "Should have alloc"
    assert "affine_for" in result, "Should have for loop"
    assert "affine_store" in result, "Should have stores"
    assert "arith_constant" in result, "Should have constants"
    assert "print_memref" in result, "Should have print"
    assert "dealloc" in result, "Should have dealloc"
    assert "return" in result, "Should have return"


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = transpose(%0) : tensor<3x2xf64>\n"
        "    print(%1)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("alloc(")
    assert alloc_count >= 2, "Should have at least 2 allocs"
    assert "affine_load" in result, "Should have loads for transpose"
    assert "alloc(<3x2>)" in result, "Should have 3x2 alloc for transposed result"


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x2>, [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = constant(<2x2>, [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = mul(%0, %1) : tensor<2x2xf64>\n"
        "    print(%2)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "mul_f" in result, "Should have mul_f op"
    alloc_count = result.count("alloc(")
    assert alloc_count >= 3, "Should have 3 allocs (2 constants + 1 result)"


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x2>, [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = constant(<2x2>, [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = add(%0, %1) : tensor<2x2xf64>\n"
        "    print(%2)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "add_f" in result, "Should have add_f op"


def test_print():
    """Print maps to print_memref."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    print(%0)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "print_memref" in result, "Should have print_memref"


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = transpose(%0) : tensor<3x2xf64>\n"
        "    %2 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = transpose(%2) : tensor<3x2xf64>\n"
        "    %4 = mul(%1, %3) : tensor<3x2xf64>\n"
        "    print(%4)\n"
        "    return()\n"
    )
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("alloc(")
    assert alloc_count >= 5, "Should have at least 5 allocs"
    assert "mul_f" in result, "Should have mul_f"
    assert "affine_load" in result, "Should have loads"
    assert "affine_store" in result, "Should have stores"
    assert "print_memref" in result, "Should have print"
    assert "return" in result, "Should have return"
