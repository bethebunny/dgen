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
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : () = affine.print_memref(%0)
        |     %2 : () = affine.dealloc(%0)
        |     %3 : () = return(())
    """)
    assert result == expected


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], f64> = toy.transpose(%0)
        |     %_ = toy.print(%1)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : affine.MemRef<[3, 2], f64> = affine.alloc([3, 2])
        |     %2 : () = affine.for<0, 2>() (%3: index):
        |         %4 : () = affine.for<0, 3>() (%5: index):
        |             %6 : f64 = affine.load(%0, [%3, %5])
        |             %7 : () = affine.store(%6, %1, [%5, %3])
        |     %8 : () = affine.print_memref(%1)
        |     %9 : () = affine.dealloc(%0)
        |     %10 : () = affine.dealloc(%1)
        |     %11 : () = return(())
    """)
    assert result == expected


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<[2, 2], f64> = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : affine.MemRef<[2, 2], f64> = affine.alloc([2, 2])
        |     %3 : () = affine.for<0, 2>() (%4: index):
        |         %5 : () = affine.for<0, 2>() (%6: index):
        |             %7 : f64 = affine.load(%0, [%4, %6])
        |             %8 : f64 = affine.load(%1, [%4, %6])
        |             %9 : f64 = affine.mul_f(%7, %8)
        |             %10 : () = affine.store(%9, %2, [%4, %6])
        |     %11 : () = affine.print_memref(%2)
        |     %12 : () = affine.dealloc(%0)
        |     %13 : () = affine.dealloc(%1)
        |     %14 : () = affine.dealloc(%2)
        |     %15 : () = return(())
    """)
    assert result == expected


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<[2, 2], f64> = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : affine.MemRef<[2, 2], f64> = affine.alloc([2, 2])
        |     %3 : () = affine.for<0, 2>() (%4: index):
        |         %5 : () = affine.for<0, 2>() (%6: index):
        |             %7 : f64 = affine.load(%0, [%4, %6])
        |             %8 : f64 = affine.load(%1, [%4, %6])
        |             %9 : f64 = affine.add_f(%7, %8)
        |             %10 : () = affine.store(%9, %2, [%4, %6])
        |     %11 : () = affine.print_memref(%2)
        |     %12 : () = affine.dealloc(%0)
        |     %13 : () = affine.dealloc(%1)
        |     %14 : () = affine.dealloc(%2)
        |     %15 : () = return(())
    """)
    assert result == expected


def test_print():
    """Print maps to print_memref."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : () = affine.print_memref(%0)
        |     %2 : () = affine.dealloc(%0)
        |     %3 : () = return(())
    """)
    assert result == expected


def test_3d_constant():
    """3D tensor constant passes through as-is."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : () = affine.print_memref(%0)
        |     %2 : () = affine.dealloc(%0)
        |     %3 : () = return(())
    """)
    assert result == expected


def test_3d_add():
    """3D add lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], f64> = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : affine.MemRef<[2, 2, 2], f64> = affine.alloc([2, 2, 2])
        |     %3 : () = affine.for<0, 2>() (%4: index):
        |         %5 : () = affine.for<0, 2>() (%6: index):
        |             %7 : () = affine.for<0, 2>() (%8: index):
        |                 %9 : f64 = affine.load(%0, [%4, %6, %8])
        |                 %10 : f64 = affine.load(%1, [%4, %6, %8])
        |                 %11 : f64 = affine.add_f(%9, %10)
        |                 %12 : () = affine.store(%11, %2, [%4, %6, %8])
        |     %13 : () = affine.print_memref(%2)
        |     %14 : () = affine.dealloc(%0)
        |     %15 : () = affine.dealloc(%1)
        |     %16 : () = affine.dealloc(%2)
        |     %17 : () = return(())
    """)
    assert result == expected


def test_3d_mul():
    """3D mul lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], f64> = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : affine.MemRef<[2, 2, 2], f64> = affine.alloc([2, 2, 2])
        |     %3 : () = affine.for<0, 2>() (%4: index):
        |         %5 : () = affine.for<0, 2>() (%6: index):
        |             %7 : () = affine.for<0, 2>() (%8: index):
        |                 %9 : f64 = affine.load(%0, [%4, %6, %8])
        |                 %10 : f64 = affine.load(%1, [%4, %6, %8])
        |                 %11 : f64 = affine.mul_f(%9, %10)
        |                 %12 : () = affine.store(%11, %2, [%4, %6, %8])
        |     %13 : () = affine.print_memref(%2)
        |     %14 : () = affine.dealloc(%0)
        |     %15 : () = affine.dealloc(%1)
        |     %16 : () = affine.dealloc(%2)
        |     %17 : () = return(())
    """)
    assert result == expected


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], f64> = toy.transpose(%0)
        |     %2 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<[3, 2], f64> = toy.transpose(%2)
        |     %4 : toy.Tensor<[3, 2], f64> = toy.mul(%1, %3)
        |     %_ = toy.print(%4)
        |     %_ = return(())
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : affine.MemRef<[3, 2], f64> = affine.alloc([3, 2])
        |     %2 : () = affine.for<0, 2>() (%3: index):
        |         %4 : () = affine.for<0, 3>() (%5: index):
        |             %6 : f64 = affine.load(%0, [%3, %5])
        |             %7 : () = affine.store(%6, %1, [%5, %3])
        |     %8 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %9 : affine.MemRef<[3, 2], f64> = affine.alloc([3, 2])
        |     %10 : () = affine.for<0, 2>() (%11: index):
        |         %12 : () = affine.for<0, 3>() (%13: index):
        |             %14 : f64 = affine.load(%8, [%11, %13])
        |             %15 : () = affine.store(%14, %9, [%13, %11])
        |     %16 : affine.MemRef<[3, 2], f64> = affine.alloc([3, 2])
        |     %17 : () = affine.for<0, 3>() (%18: index):
        |         %19 : () = affine.for<0, 2>() (%20: index):
        |             %21 : f64 = affine.load(%1, [%18, %20])
        |             %22 : f64 = affine.load(%9, [%18, %20])
        |             %23 : f64 = affine.mul_f(%21, %22)
        |             %24 : () = affine.store(%23, %16, [%18, %20])
        |     %25 : () = affine.print_memref(%16)
        |     %26 : () = affine.dealloc(%0)
        |     %27 : () = affine.dealloc(%1)
        |     %28 : () = affine.dealloc(%8)
        |     %29 : () = affine.dealloc(%9)
        |     %30 : () = affine.dealloc(%16)
        |     %31 : () = return(())
    """)
    assert result == expected
