"""Ch6 tests: Affine IR to LLVM-like IR lowering."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.toy_to_affine import lower_to_affine
from toy.test.helpers import strip_prefix


def compile_to_llvm(ir_text: str) -> str:
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_simple_constant():
    """Tensor constant passes through to LLVM level."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(3), f64] = (1.0, 2.0, 3.0)
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "= (" in result, "Tensor constant should pass through"
    assert 'llvm.call("print_memref"' in result, "Should have print_memref call"
    assert "return()" in result, "Should have return()"


def test_constant_preserved():
    """Constants are preserved as tensor constants (not expanded to scalar stores)."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(3), f64] = (1.0, 2.0, 3.0)
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "= (1.0, 2.0, 3.0)" in result
    assert result.count("llvm.store(") == 0, "No stores at dgen IR level for constants"


def test_2d_constant_preserved():
    """2D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "= (" in result, "Tensor constant should pass through"


def test_load_store_linearization():
    """Load/store with multi-dim indices are linearized."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %1 : toy.Tensor[(3, 2), f64] = toy.transpose(%0)
        |     %_ = toy.print(%1)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "llvm.gep(" in result, "Should have gep for pointer arithmetic"
    assert "llvm.load(" in result, "Should have load"
    assert "llvm.mul(" in result, "Should have mul for index linearization"


def test_3d_constant_preserved():
    """3D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 2, 2), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "= (" in result, "Tensor constant should pass through"


def test_3d_load_store_linearization():
    """3D load/store indices are linearized with stride multiplication."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 2, 2), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        |     %1 : toy.Tensor[(2, 2, 2), f64] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        |     %2 : toy.Tensor[(2, 2, 2), f64] = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "llvm.mul(" in result, "Should have mul for stride linearization"
    assert "llvm.add(" in result, "Should have add for index accumulation"
    assert "llvm.gep(" in result, "Should have gep for pointer arithmetic"
    assert "llvm.load(" in result, "Should have load"
    assert "llvm.fadd(" in result, "Should have fadd for add op"


def test_full_example():
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %1 : toy.Tensor[(3, 2), f64] = toy.transpose(%0)
        |     %2 : toy.Tensor[(2, 3), f64] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        |     %3 : toy.Tensor[(3, 2), f64] = toy.transpose(%2)
        |     %4 : toy.Tensor[(3, 2), f64] = toy.mul(%1, %3)
        |     %_ = toy.print(%4)
        |     %_ = return()
    """)
    result = compile_to_llvm(ir_text)
    assert "%main = function () -> ():" in result, "Should have function def"
    assert "llvm.alloca(" in result, "Should have alloca for non-constant allocs"
    assert "= (" in result, "Should have tensor constants"
    assert "llvm.fmul(" in result, "Should have fmul for Mul op"
    assert 'llvm.call("print_memref"' in result, "Should have print_memref"
    assert "return()" in result, "Should have return()"
