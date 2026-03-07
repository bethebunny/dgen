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
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = affine.print_memref(%0)
        |     %2 : Nil = affine.dealloc(%0)
        |     %3 : Nil = return(Nil)
    """)
    assert result == expected


def test_transpose():
    """Transpose lowers to alloc + transposed loop nest."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %_ = toy.print(%1)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : affine.MemRef<[3, 2], F64> = affine.alloc([3, 2])
        |     %2 : Nil = affine.for<0, 2>() (%3: Index):
        |         %4 : Nil = affine.for<0, 3>() (%5: Index):
        |             %6 : F64 = affine.load(%0, [%3, %5])
        |             %7 : Nil = affine.store(%6, %1, [%5, %3])
        |     %8 : Nil = affine.print_memref(%1)
        |     %9 : Nil = affine.dealloc(%0)
        |     %10 : Nil = affine.dealloc(%1)
        |     %11 : Nil = return(Nil)
    """)
    assert result == expected


def test_mul():
    """Mul lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<[2, 2], F64> = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : affine.MemRef<[2, 2], F64> = affine.alloc([2, 2])
        |     %3 : Nil = affine.for<0, 2>() (%4: Index):
        |         %5 : Nil = affine.for<0, 2>() (%6: Index):
        |             %7 : F64 = affine.load(%0, [%4, %6])
        |             %8 : F64 = affine.load(%1, [%4, %6])
        |             %9 : F64 = affine.mul_f(%7, %8)
        |             %10 : Nil = affine.store(%9, %2, [%4, %6])
        |     %11 : Nil = affine.print_memref(%2)
        |     %12 : Nil = affine.dealloc(%0)
        |     %13 : Nil = affine.dealloc(%1)
        |     %14 : Nil = affine.dealloc(%2)
        |     %15 : Nil = return(Nil)
    """)
    assert result == expected


def test_add():
    """Add lowers to alloc + element-wise loop."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<[2, 2], F64> = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : affine.MemRef<[2, 2], F64> = affine.alloc([2, 2])
        |     %3 : Nil = affine.for<0, 2>() (%4: Index):
        |         %5 : Nil = affine.for<0, 2>() (%6: Index):
        |             %7 : F64 = affine.load(%0, [%4, %6])
        |             %8 : F64 = affine.load(%1, [%4, %6])
        |             %9 : F64 = affine.add_f(%7, %8)
        |             %10 : Nil = affine.store(%9, %2, [%4, %6])
        |     %11 : Nil = affine.print_memref(%2)
        |     %12 : Nil = affine.dealloc(%0)
        |     %13 : Nil = affine.dealloc(%1)
        |     %14 : Nil = affine.dealloc(%2)
        |     %15 : Nil = return(Nil)
    """)
    assert result == expected


def test_print():
    """Print maps to print_memref."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = affine.print_memref(%0)
        |     %2 : Nil = affine.dealloc(%0)
        |     %3 : Nil = return(Nil)
    """)
    assert result == expected


def test_3d_constant():
    """3D tensor constant passes through as-is."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %_ = toy.print(%0)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : Nil = affine.print_memref(%0)
        |     %2 : Nil = affine.dealloc(%0)
        |     %3 : Nil = return(Nil)
    """)
    assert result == expected


def test_3d_add():
    """3D add lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], F64> = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : affine.MemRef<[2, 2, 2], F64> = affine.alloc([2, 2, 2])
        |     %3 : Nil = affine.for<0, 2>() (%4: Index):
        |         %5 : Nil = affine.for<0, 2>() (%6: Index):
        |             %7 : Nil = affine.for<0, 2>() (%8: Index):
        |                 %9 : F64 = affine.load(%0, [%4, %6, %8])
        |                 %10 : F64 = affine.load(%1, [%4, %6, %8])
        |                 %11 : F64 = affine.add_f(%9, %10)
        |                 %12 : Nil = affine.store(%11, %2, [%4, %6, %8])
        |     %13 : Nil = affine.print_memref(%2)
        |     %14 : Nil = affine.dealloc(%0)
        |     %15 : Nil = affine.dealloc(%1)
        |     %16 : Nil = affine.dealloc(%2)
        |     %17 : Nil = return(Nil)
    """)
    assert result == expected


def test_3d_mul():
    """3D mul lowers to alloc + element-wise nested loops."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], F64> = toy.mul(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : affine.MemRef<[2, 2, 2], F64> = affine.alloc([2, 2, 2])
        |     %3 : Nil = affine.for<0, 2>() (%4: Index):
        |         %5 : Nil = affine.for<0, 2>() (%6: Index):
        |             %7 : Nil = affine.for<0, 2>() (%8: Index):
        |                 %9 : F64 = affine.load(%0, [%4, %6, %8])
        |                 %10 : F64 = affine.load(%1, [%4, %6, %8])
        |                 %11 : F64 = affine.mul_f(%9, %10)
        |                 %12 : Nil = affine.store(%11, %2, [%4, %6, %8])
        |     %13 : Nil = affine.print_memref(%2)
        |     %14 : Nil = affine.dealloc(%0)
        |     %15 : Nil = affine.dealloc(%1)
        |     %16 : Nil = affine.dealloc(%2)
        |     %17 : Nil = return(Nil)
    """)
    assert result == expected


def test_full_example():
    """Full pipeline: constant + reshape + transpose + mul + print."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<[3, 2], F64> = toy.transpose(%2)
        |     %4 : toy.Tensor<[3, 2], F64> = toy.mul(%1, %3)
        |     %_ = toy.print(%4)
        |     %_ = return(Nil)
    """)
    m = parse_module(ir_text)
    affine = lower_to_affine(m)
    result = asm.format(affine)
    expected = strip_prefix("""
        | import affine
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : affine.MemRef<[3, 2], F64> = affine.alloc([3, 2])
        |     %2 : Nil = affine.for<0, 2>() (%3: Index):
        |         %4 : Nil = affine.for<0, 3>() (%5: Index):
        |             %6 : F64 = affine.load(%0, [%3, %5])
        |             %7 : Nil = affine.store(%6, %1, [%5, %3])
        |     %8 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %9 : affine.MemRef<[3, 2], F64> = affine.alloc([3, 2])
        |     %10 : Nil = affine.for<0, 2>() (%11: Index):
        |         %12 : Nil = affine.for<0, 3>() (%13: Index):
        |             %14 : F64 = affine.load(%8, [%11, %13])
        |             %15 : Nil = affine.store(%14, %9, [%13, %11])
        |     %16 : affine.MemRef<[3, 2], F64> = affine.alloc([3, 2])
        |     %17 : Nil = affine.for<0, 3>() (%18: Index):
        |         %19 : Nil = affine.for<0, 2>() (%20: Index):
        |             %21 : F64 = affine.load(%1, [%18, %20])
        |             %22 : F64 = affine.load(%9, [%18, %20])
        |             %23 : F64 = affine.mul_f(%21, %22)
        |             %24 : Nil = affine.store(%23, %16, [%18, %20])
        |     %25 : Nil = affine.print_memref(%16)
        |     %26 : Nil = affine.dealloc(%0)
        |     %27 : Nil = affine.dealloc(%1)
        |     %28 : Nil = affine.dealloc(%8)
        |     %29 : Nil = affine.dealloc(%9)
        |     %30 : Nil = affine.dealloc(%16)
        |     %31 : Nil = return(Nil)
    """)
    assert result == expected
