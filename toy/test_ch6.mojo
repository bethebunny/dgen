"""Ch6 tests: Affine IR to LLVM-like IR lowering."""

from testing import assert_equal, assert_true, TestSuite

from toy.ir_parser import parse_module
from toy.toy_to_affine import lower_to_affine
from toy.affine_to_llvm import lower_to_llvm
from toy.llvm_printer import print_llvm_module


fn compile_to_llvm(ir_text: String) raises -> String:
    var m = parse_module(ir_text)
    var affine = lower_to_affine(m)
    var llvm = lower_to_llvm(affine)
    return print_llvm_module(llvm)


def test_simple_constant_store():
    """Constant store lowers to alloca + fconst + gep + store."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<3> [1.0, 2.0, 3.0]) : tensor<3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var result = compile_to_llvm(ir_text)
    assert_true("alloca f64, 3" in result, "Should have alloca for 3 elements")
    assert_true("fconst 1.0" in result, "Should have fconst 1.0")
    assert_true("gep" in result, "Should have gep")
    assert_true("store" in result, "Should have store")
    assert_true("call @print_memref" in result, "Should have print_memref call")
    assert_true("ret void" in result, "Should have ret void")


def test_single_for_loop():
    """For loop lowers to label/branch/phi pattern."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<3> [1.0, 2.0, 3.0]) : tensor<3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var result = compile_to_llvm(ir_text)
    assert_true("loop_header" in result, "Should have loop header label")
    assert_true("phi" in result, "Should have phi node")
    assert_true("icmp slt" in result, "Should have comparison")
    assert_true("cond_br" in result, "Should have conditional branch")
    assert_true("loop_body" in result, "Should have loop body label")
    assert_true("loop_exit" in result, "Should have loop exit label")


def test_nested_for_loops():
    """Nested for loops produce nested label/branch patterns."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    var result = compile_to_llvm(ir_text)
    # Should have at least 2 loop headers (outer + inner)
    assert_true("loop_header0" in result, "Should have loop_header0")
    assert_true("loop_header1" in result, "Should have loop_header1")
    assert_true("loop_body0" in result, "Should have loop_body0")
    assert_true("loop_body1" in result, "Should have loop_body1")


def test_load_store_linearization():
    """Load/store with multi-dim indices are linearized."""
    var ir_text = String(
        "from toy use *\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    var result = compile_to_llvm(ir_text)
    # Transpose requires load + store with linearized indices
    assert_true("gep" in result, "Should have gep for pointer arithmetic")
    assert_true("load" in result, "Should have load")
    assert_true("mul" in result, "Should have mul for index linearization")


def test_full_example():
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
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
    var result = compile_to_llvm(ir_text)
    assert_true("define void @main" in result, "Should have function def")
    assert_true("alloca" in result, "Should have alloca")
    assert_true("fconst" in result, "Should have fconst")
    assert_true("fmul" in result, "Should have fmul for Mul op")
    assert_true("call @print_memref" in result, "Should have print_memref")
    assert_true("ret void" in result, "Should have ret void")
    # Print for visual inspection
    print(result)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
