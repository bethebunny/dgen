"""Ch5 tests: Toy IR to Affine IR lowering."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.passes.toy_to_affine import lower_to_affine
from toy.test.helpers import strip_prefix


def test_simple_constant():
    """Tensor constant passes through as-is (no alloc/store expansion)."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "= [" in result, "Tensor constant should pass through"
    assert "affine.alloc(" not in result, "No alloc for constants"
    assert "affine.store" not in result, "No stores for constants"
    assert "affine.print_memref" in result, "Should have print"
    assert "affine.dealloc" in result, "Should have dealloc"
    assert "return" in result, "Should have return"


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor([3, 2], f64) = toy.transpose(%0)
        |     %_ = toy.print(%1)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert result.count("affine.alloc(") == 1, "Should have 1 alloc (transpose result)"
    assert "affine.alloc([3, 2])" in result, "Should have 3x2 alloc for transposed result"
    assert "affine.load" in result, "Should have loads for transpose"
    assert "affine.store" in result, "Should have stores for transpose"


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2], f64) = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor([2, 2], f64) = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor([2, 2], f64) = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.mul_f" in result, "Should have mul_f op"
    assert result.count("affine.alloc(") == 1, "Should have 1 alloc (result only)"


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2], f64) = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor([2, 2], f64) = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor([2, 2], f64) = toy.add(%0, %1)
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
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.print_memref" in result, "Should have print_memref"


def test_3d_constant():
    """3D tensor constant passes through as-is."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2, 2], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %_ = toy.print(%0)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "= [" in result, "Tensor constant should pass through"
    assert "affine.alloc(" not in result, "No alloc for constants"
    assert "affine.store" not in result, "No stores for constants"


def test_3d_add():
    """3D add lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2, 2], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor([2, 2, 2], f64) = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor([2, 2, 2], f64) = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.add_f" in result, "Should have add_f op"
    assert result.count("affine.alloc(") == 1, "Should have 1 alloc (result only)"
    assert "affine.alloc([2, 2, 2])" in result, "Should have 2x2x2 alloc for result"


def test_3d_mul():
    """3D mul lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2, 2], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor([2, 2, 2], f64) = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor([2, 2, 2], f64) = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    assert "affine.mul_f" in result, "Should have mul_f op"
    assert result.count("affine.alloc(") == 1, "Should have 1 alloc (result only)"


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor([3, 2], f64) = toy.transpose(%0)
        |     %2 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor([3, 2], f64) = toy.transpose(%2)
        |     %4 : toy.Tensor([3, 2], f64) = toy.mul(%1, %3)
        |     %_ = toy.print(%4)
        |     %_ = return()
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    alloc_count = result.count("affine.alloc(")
    assert alloc_count == 3, "Should have 3 allocs (2 transpose + 1 mul result)"
    assert "affine.mul_f" in result, "Should have mul_f"
    assert "affine.load" in result, "Should have loads"
    assert "affine.store" in result, "Should have stores"
    assert "affine.print_memref" in result, "Should have print"
    assert "return" in result, "Should have return"
