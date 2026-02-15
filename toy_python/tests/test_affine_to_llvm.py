"""Ch6 tests: Affine IR to LLVM-like IR lowering."""

from toy_python.asm.parser import parse_module
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python.passes.affine_to_llvm import lower_to_llvm
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def compile_to_llvm(ir_text: str) -> str:
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_simple_constant_store():
    """Constant store lowers to alloca + fconst + gep + store."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<3>, [1.0, 2.0, 3.0]) : tensor<3xf64>
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "llvm.alloca(3)" in result, "Should have alloca for 3 elements"
    assert "llvm.fconst(1.0)" in result, "Should have fconst 1.0"
    assert "llvm.gep(" in result, "Should have gep"
    assert "llvm.store(" in result, "Should have store"
    assert "llvm.call(@print_memref" in result, "Should have print_memref call"
    assert "return()" in result, "Should have return()"


def test_single_for_loop():
    """For loop lowers to label/branch/phi pattern."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<3>, [1.0, 2.0, 3.0]) : tensor<3xf64>
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "loop_header" in result, "Should have loop header label"
    assert "llvm.phi(" in result, "Should have phi node"
    assert "llvm.icmp(slt" in result, "Should have comparison"
    assert "llvm.cond_br(" in result, "Should have conditional branch"
    assert "loop_body" in result, "Should have loop body label"
    assert "loop_exit" in result, "Should have loop exit label"


def test_nested_for_loops():
    """Nested for loops produce nested label/branch patterns."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "loop_header0" in result, "Should have loop_header0"
    assert "loop_header1" in result, "Should have loop_header1"
    assert "loop_body0" in result, "Should have loop_body0"
    assert "loop_body1" in result, "Should have loop_body1"


def test_load_store_linearization():
    """Load/store with multi-dim indices are linearized."""
    ir_text = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.transpose(%0) : tensor<3x2xf64>
        |     %_ = toy.print(%1)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "llvm.gep(" in result, "Should have gep for pointer arithmetic"
    assert "llvm.load(" in result, "Should have load"
    assert "llvm.mul(" in result, "Should have mul for index linearization"


def test_full_example():
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
    ir_text = strip_prefix("""
        | from builtin import function, return
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
    result = compile_to_llvm(ir_text)
    assert "%main = function () -> ():" in result, "Should have function def"
    assert "llvm.alloca(" in result, "Should have alloca"
    assert "llvm.fconst(" in result, "Should have fconst"
    assert "llvm.fmul(" in result, "Should have fmul for Mul op"
    assert "llvm.call(@print_memref" in result, "Should have print_memref"
    assert "return()" in result, "Should have return()"
