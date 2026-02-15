"""Ch6 tests: Affine IR to LLVM-like IR lowering."""

from toy_python.dialects.toy import parse_toy_module as parse_module
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python.passes.affine_to_llvm import lower_to_llvm
from toy_python import asm


def compile_to_llvm(ir_text: str) -> str:
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_simple_constant_store():
    """Constant store lowers to alloca + fconst + gep + store."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<3>, [1.0, 2.0, 3.0]) : tensor<3xf64>\n"
        "    Print(%0)\n"
        "    Return()\n"
    )
    result = compile_to_llvm(ir_text)
    assert "Alloca(3)" in result, "Should have alloca for 3 elements"
    assert "FConst(1.0)" in result, "Should have FConst 1.0"
    assert "Gep(" in result, "Should have gep"
    assert "Store(" in result, "Should have store"
    assert "Call(@print_memref" in result, "Should have print_memref call"
    assert "Return()" in result, "Should have Return()"


def test_single_for_loop():
    """For loop lowers to label/branch/phi pattern."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<3>, [1.0, 2.0, 3.0]) : tensor<3xf64>\n"
        "    Print(%0)\n"
        "    Return()\n"
    )
    result = compile_to_llvm(ir_text)
    assert "loop_header" in result, "Should have loop header label"
    assert "Phi(" in result, "Should have phi node"
    assert "Icmp(slt" in result, "Should have comparison"
    assert "CondBr(" in result, "Should have conditional branch"
    assert "loop_body" in result, "Should have loop body label"
    assert "loop_exit" in result, "Should have loop exit label"


def test_nested_for_loops():
    """Nested for loops produce nested label/branch patterns."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    Return()\n"
    )
    result = compile_to_llvm(ir_text)
    assert "loop_header0" in result, "Should have loop_header0"
    assert "loop_header1" in result, "Should have loop_header1"
    assert "loop_body0" in result, "Should have loop_body0"
    assert "loop_body1" in result, "Should have loop_body1"


def test_load_store_linearization():
    """Load/store with multi-dim indices are linearized."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    Print(%1)\n"
        "    Return()\n"
    )
    result = compile_to_llvm(ir_text)
    assert "Gep(" in result, "Should have gep for pointer arithmetic"
    assert "Load(" in result, "Should have load"
    assert "Mul(" in result, "Should have mul for index linearization"


def test_full_example():
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
    ir_text = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<3x2xf64>\n"
        "    %2 = Constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %3 = Transpose(%2) : tensor<3x2xf64>\n"
        "    %4 = Mul(%1, %3) : tensor<3x2xf64>\n"
        "    Print(%4)\n"
        "    Return()\n"
    )
    result = compile_to_llvm(ir_text)
    assert "%main = function () -> ():" in result, "Should have function def"
    assert "Alloca(" in result, "Should have alloca"
    assert "FConst(" in result, "Should have fconst"
    assert "FMul(" in result, "Should have fmul for Mul op"
    assert "Call(@print_memref" in result, "Should have print_memref"
    assert "Return()" in result, "Should have Return()"
