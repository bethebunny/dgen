"""Ch5 tests: Toy IR to Affine IR lowering."""

from testing import assert_equal, assert_true, TestSuite

from toy.ir_parser import parse_module
from toy.toy_to_affine import lower_to_affine
from toy.affine_printer import print_affine_module


def test_simple_constant():
    """Constant op lowers to alloc + stores."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    # Verify key structures are present
    assert_true("Alloc<2x3>()" in result, "Should have alloc")
    assert_true("AffineFor" in result, "Should have for loop")
    assert_true("AffineStore" in result, "Should have stores")
    assert_true("ArithConstant" in result, "Should have constants")
    assert_true("PrintMemRef" in result, "Should have print")
    assert_true("Dealloc" in result, "Should have dealloc")
    assert_true("return" in result, "Should have return")


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    # Should have two allocs: one for constant, one for transpose result
    var alloc_count = _count_substr(result, "Alloc<")
    assert_true(alloc_count >= 2, "Should have at least 2 allocs")
    assert_true("AffineLoad" in result, "Should have loads for transpose")
    assert_true("Alloc<3x2>()" in result, "Should have 3x2 alloc for transposed result")


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x2> [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = Constant(<2x2> [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<2x2xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    assert_true("MulF" in result, "Should have MulF op")
    var alloc_count = _count_substr(result, "Alloc<")
    assert_true(alloc_count >= 3, "Should have 3 allocs (2 constants + 1 result)")


def test_add():
    """Add lowers to alloc + element-wise loop."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x2> [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = Constant(<2x2> [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = Add(%0, %1) : tensor<2x2xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    assert_true("AddF" in result, "Should have AddF op")


def test_print():
    """Print maps to PrintMemRef."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    assert_true("PrintMemRef" in result, "Should have PrintMemRef")


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    %2 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = Transpose(%2) : tensor<3x2xf64>\n"
        "    %4 = Mul(%1, %3) : tensor<3x2xf64>\n"
        "    Print(%4)\n"
        "    return\n"
    )
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var result = print_affine_module(affine)
    # Should have: 2 constant allocs, 2 transpose allocs, 1 mul alloc = 5 allocs
    var alloc_count = _count_substr(result, "Alloc<")
    assert_true(alloc_count >= 5, "Should have at least 5 allocs")
    assert_true("MulF" in result, "Should have MulF")
    assert_true("AffineLoad" in result, "Should have loads")
    assert_true("AffineStore" in result, "Should have stores")
    assert_true("PrintMemRef" in result, "Should have print")
    assert_true("return" in result, "Should have return")
    # Print the result for visual inspection
    print(result)


fn _count_substr(s: String, sub: String) -> Int:
    var count = 0
    for i in range(len(s) - len(sub) + 1):
        var found = True
        for j in range(len(sub)):
            if String(s[byte=i + j]) != String(sub[byte=j]):
                found = False
                break
        if found:
            count += 1
    return count


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
