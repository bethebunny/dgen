"""End-to-end tests: Toy source -> parse -> lower -> optimize -> affine -> LLVM IR."""

from dgen import asm
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine
from toy.test.helpers import strip_prefix


def compile(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    opt = optimize(ir)
    typed = infer_shapes(opt)
    affine = lower_to_affine(typed)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_constant_print():
    """Constant tensor + print: tensor constant passes through, codegen materializes."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : Index = 6
        |     %3 : Nil = llvm.call<"print_memref">([%1, %2])
        |     %4 : Nil = return(%3)
    """)
    assert result == expected


def test_transpose():
    """Transpose produces a second alloc and transposed load/store pattern."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<6>()
        |     %1 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : llvm.Ptr = llvm.load(%1)
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %33)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_body1">(%3, %29)
        |     %14 : Index = 3
        |     %15 : llvm.Int<1> = llvm.icmp<"slt">(%13, %14)
        |     %16 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%15)
        |     %17 : Nil = llvm.label<"loop_body1">()
        |     %18 : Index = 3
        |     %19 : llvm.Int<64> = llvm.mul(%6, %18)
        |     %20 : llvm.Int<64> = llvm.add(%19, %13)
        |     %21 : llvm.Ptr = llvm.gep(%2, %20)
        |     %22 : llvm.Float = llvm.load(%21)
        |     %23 : Index = 2
        |     %24 : llvm.Int<64> = llvm.mul(%13, %23)
        |     %25 : llvm.Int<64> = llvm.add(%24, %6)
        |     %26 : llvm.Ptr = llvm.gep(%0, %25)
        |     %27 : Nil = llvm.store(%22, %26)
        |     %28 : Index = 1
        |     %29 : llvm.Int<64> = llvm.add(%13, %28)
        |     %30 : Nil = llvm.br<"loop_header1">()
        |     %31 : Nil = llvm.label<"loop_exit1">()
        |     %32 : Index = 1
        |     %33 : llvm.Int<64> = llvm.add(%6, %32)
        |     %34 : Nil = llvm.br<"loop_header0">()
        |     %35 : Nil = llvm.label<"loop_exit0">()
        |     %36 : Index = 6
        |     %37 : Nil = llvm.call<"print_memref">([%0, %36])
        |     %38 : Nil = return(%37)
    """)
    assert result == expected


def test_element_wise_mul():
    """Element-wise multiply produces fmul in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<4>()
        |     %1 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %2 : llvm.Ptr = llvm.load(%1)
        |     %3 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %4 : llvm.Ptr = llvm.load(%3)
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %40)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Nil = llvm.br<"loop_header1">()
        |     %14 : Nil = llvm.label<"loop_header1">()
        |     %15 : Nil = llvm.phi<"loop_body0", "loop_body1">(%5, %36)
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%15, %9)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 2
        |     %20 : llvm.Int<64> = llvm.mul(%8, %19)
        |     %21 : llvm.Int<64> = llvm.add(%20, %15)
        |     %22 : llvm.Ptr = llvm.gep(%2, %21)
        |     %23 : llvm.Float = llvm.load(%22)
        |     %24 : Index = 2
        |     %25 : llvm.Int<64> = llvm.mul(%8, %24)
        |     %26 : llvm.Int<64> = llvm.add(%25, %15)
        |     %27 : llvm.Ptr = llvm.gep(%4, %26)
        |     %28 : llvm.Float = llvm.load(%27)
        |     %29 : llvm.Float = llvm.fmul(%23, %28)
        |     %30 : Index = 2
        |     %31 : llvm.Int<64> = llvm.mul(%8, %30)
        |     %32 : llvm.Int<64> = llvm.add(%31, %15)
        |     %33 : llvm.Ptr = llvm.gep(%0, %32)
        |     %34 : Nil = llvm.store(%29, %33)
        |     %35 : Index = 1
        |     %36 : llvm.Int<64> = llvm.add(%15, %35)
        |     %37 : Nil = llvm.br<"loop_header1">()
        |     %38 : Nil = llvm.label<"loop_exit1">()
        |     %39 : Index = 1
        |     %40 : llvm.Int<64> = llvm.add(%8, %39)
        |     %41 : Nil = llvm.br<"loop_header0">()
        |     %42 : Nil = llvm.label<"loop_exit0">()
        |     %43 : Index = 4
        |     %44 : Nil = llvm.call<"print_memref">([%0, %43])
        |     %45 : Nil = return(%44)
    """)
    assert result == expected


def test_element_wise_add():
    """Element-wise add produces fadd in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<4>()
        |     %1 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %2 : llvm.Ptr = llvm.load(%1)
        |     %3 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %4 : llvm.Ptr = llvm.load(%3)
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %40)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Nil = llvm.br<"loop_header1">()
        |     %14 : Nil = llvm.label<"loop_header1">()
        |     %15 : Nil = llvm.phi<"loop_body0", "loop_body1">(%5, %36)
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%15, %9)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 2
        |     %20 : llvm.Int<64> = llvm.mul(%8, %19)
        |     %21 : llvm.Int<64> = llvm.add(%20, %15)
        |     %22 : llvm.Ptr = llvm.gep(%2, %21)
        |     %23 : llvm.Float = llvm.load(%22)
        |     %24 : Index = 2
        |     %25 : llvm.Int<64> = llvm.mul(%8, %24)
        |     %26 : llvm.Int<64> = llvm.add(%25, %15)
        |     %27 : llvm.Ptr = llvm.gep(%4, %26)
        |     %28 : llvm.Float = llvm.load(%27)
        |     %29 : llvm.Float = llvm.fadd(%23, %28)
        |     %30 : Index = 2
        |     %31 : llvm.Int<64> = llvm.mul(%8, %30)
        |     %32 : llvm.Int<64> = llvm.add(%31, %15)
        |     %33 : llvm.Ptr = llvm.gep(%0, %32)
        |     %34 : Nil = llvm.store(%29, %33)
        |     %35 : Index = 1
        |     %36 : llvm.Int<64> = llvm.add(%15, %35)
        |     %37 : Nil = llvm.br<"loop_header1">()
        |     %38 : Nil = llvm.label<"loop_exit1">()
        |     %39 : Index = 1
        |     %40 : llvm.Int<64> = llvm.add(%8, %39)
        |     %41 : Nil = llvm.br<"loop_header0">()
        |     %42 : Nil = llvm.label<"loop_exit0">()
        |     %43 : Index = 4
        |     %44 : Nil = llvm.call<"print_memref">([%0, %43])
        |     %45 : Nil = return(%44)
    """)
    assert result == expected


def test_3d_constant_print():
    """3D constant tensor + print: tensor constant passes through."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : Index = 8
        |     %3 : Nil = llvm.call<"print_memref">([%1, %2])
        |     %4 : Nil = return(%3)
    """)
    assert result == expected


def test_3d_element_wise_add():
    """3D element-wise add produces fadd in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<8>()
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %2 : llvm.Ptr = llvm.load(%1)
        |     %3 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %4 : llvm.Ptr = llvm.load(%3)
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %59)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Nil = llvm.br<"loop_header1">()
        |     %14 : Nil = llvm.label<"loop_header1">()
        |     %15 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%5, %55)
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%15, %9)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Nil = llvm.br<"loop_header2">()
        |     %20 : Nil = llvm.label<"loop_header2">()
        |     %21 : Nil = llvm.phi<"loop_body1", "loop_body2">(%5, %51)
        |     %22 : llvm.Int<1> = llvm.icmp<"slt">(%21, %9)
        |     %23 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%22)
        |     %24 : Nil = llvm.label<"loop_body2">()
        |     %25 : Index = 4
        |     %26 : llvm.Int<64> = llvm.mul(%8, %25)
        |     %27 : Index = 2
        |     %28 : llvm.Int<64> = llvm.mul(%15, %27)
        |     %29 : llvm.Int<64> = llvm.add(%26, %28)
        |     %30 : llvm.Int<64> = llvm.add(%29, %21)
        |     %31 : llvm.Ptr = llvm.gep(%2, %30)
        |     %32 : llvm.Float = llvm.load(%31)
        |     %33 : Index = 4
        |     %34 : llvm.Int<64> = llvm.mul(%8, %33)
        |     %35 : Index = 2
        |     %36 : llvm.Int<64> = llvm.mul(%15, %35)
        |     %37 : llvm.Int<64> = llvm.add(%34, %36)
        |     %38 : llvm.Int<64> = llvm.add(%37, %21)
        |     %39 : llvm.Ptr = llvm.gep(%4, %38)
        |     %40 : llvm.Float = llvm.load(%39)
        |     %41 : llvm.Float = llvm.fadd(%32, %40)
        |     %42 : Index = 4
        |     %43 : llvm.Int<64> = llvm.mul(%8, %42)
        |     %44 : Index = 2
        |     %45 : llvm.Int<64> = llvm.mul(%15, %44)
        |     %46 : llvm.Int<64> = llvm.add(%43, %45)
        |     %47 : llvm.Int<64> = llvm.add(%46, %21)
        |     %48 : llvm.Ptr = llvm.gep(%0, %47)
        |     %49 : Nil = llvm.store(%41, %48)
        |     %50 : Index = 1
        |     %51 : llvm.Int<64> = llvm.add(%21, %50)
        |     %52 : Nil = llvm.br<"loop_header2">()
        |     %53 : Nil = llvm.label<"loop_exit2">()
        |     %54 : Index = 1
        |     %55 : llvm.Int<64> = llvm.add(%15, %54)
        |     %56 : Nil = llvm.br<"loop_header1">()
        |     %57 : Nil = llvm.label<"loop_exit1">()
        |     %58 : Index = 1
        |     %59 : llvm.Int<64> = llvm.add(%8, %58)
        |     %60 : Nil = llvm.br<"loop_header0">()
        |     %61 : Nil = llvm.label<"loop_exit0">()
        |     %62 : Index = 8
        |     %63 : Nil = llvm.call<"print_memref">([%0, %62])
        |     %64 : Nil = return(%63)
    """)
    assert result == expected


def test_3d_element_wise_mul():
    """3D element-wise multiply produces fmul in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<8>()
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %2 : llvm.Ptr = llvm.load(%1)
        |     %3 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %4 : llvm.Ptr = llvm.load(%3)
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %59)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Nil = llvm.br<"loop_header1">()
        |     %14 : Nil = llvm.label<"loop_header1">()
        |     %15 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%5, %55)
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%15, %9)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Nil = llvm.br<"loop_header2">()
        |     %20 : Nil = llvm.label<"loop_header2">()
        |     %21 : Nil = llvm.phi<"loop_body1", "loop_body2">(%5, %51)
        |     %22 : llvm.Int<1> = llvm.icmp<"slt">(%21, %9)
        |     %23 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%22)
        |     %24 : Nil = llvm.label<"loop_body2">()
        |     %25 : Index = 4
        |     %26 : llvm.Int<64> = llvm.mul(%8, %25)
        |     %27 : Index = 2
        |     %28 : llvm.Int<64> = llvm.mul(%15, %27)
        |     %29 : llvm.Int<64> = llvm.add(%26, %28)
        |     %30 : llvm.Int<64> = llvm.add(%29, %21)
        |     %31 : llvm.Ptr = llvm.gep(%2, %30)
        |     %32 : llvm.Float = llvm.load(%31)
        |     %33 : Index = 4
        |     %34 : llvm.Int<64> = llvm.mul(%8, %33)
        |     %35 : Index = 2
        |     %36 : llvm.Int<64> = llvm.mul(%15, %35)
        |     %37 : llvm.Int<64> = llvm.add(%34, %36)
        |     %38 : llvm.Int<64> = llvm.add(%37, %21)
        |     %39 : llvm.Ptr = llvm.gep(%4, %38)
        |     %40 : llvm.Float = llvm.load(%39)
        |     %41 : llvm.Float = llvm.fmul(%32, %40)
        |     %42 : Index = 4
        |     %43 : llvm.Int<64> = llvm.mul(%8, %42)
        |     %44 : Index = 2
        |     %45 : llvm.Int<64> = llvm.mul(%15, %44)
        |     %46 : llvm.Int<64> = llvm.add(%43, %45)
        |     %47 : llvm.Int<64> = llvm.add(%46, %21)
        |     %48 : llvm.Ptr = llvm.gep(%0, %47)
        |     %49 : Nil = llvm.store(%41, %48)
        |     %50 : Index = 1
        |     %51 : llvm.Int<64> = llvm.add(%21, %50)
        |     %52 : Nil = llvm.br<"loop_header2">()
        |     %53 : Nil = llvm.label<"loop_exit2">()
        |     %54 : Index = 1
        |     %55 : llvm.Int<64> = llvm.add(%15, %54)
        |     %56 : Nil = llvm.br<"loop_header1">()
        |     %57 : Nil = llvm.label<"loop_exit1">()
        |     %58 : Index = 1
        |     %59 : llvm.Int<64> = llvm.add(%8, %58)
        |     %60 : Nil = llvm.br<"loop_header0">()
        |     %61 : Nil = llvm.label<"loop_exit0">()
        |     %62 : Index = 8
        |     %63 : Nil = llvm.call<"print_memref">([%0, %62])
        |     %64 : Nil = return(%63)
    """)
    assert result == expected


def test_reshape_folds_away():
    """Reshape of matching shape is optimized away -- no extra alloc."""
    source = strip_prefix("""
        | def main() {
        |   var x<2, 3> = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : Index = 6
        |     %3 : Nil = llvm.call<"print_memref">([%1, %2])
        |     %4 : Nil = return(%3)
    """)
    assert result == expected


def test_double_transpose_optimized():
    """transpose(transpose(x)) is eliminated by the optimizer."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(transpose(a));
        |   print(b);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : Index = 6
        |     %3 : Nil = llvm.call<"print_memref">([%1, %2])
        |     %4 : Nil = return(%3)
    """)
    assert result == expected


def test_multiply_transpose_inlined():
    """Inlined multiply_transpose: transpose + multiply through full pipeline."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = [[1, 2, 3], [4, 5, 6]];
        |   var c = transpose(a) * transpose(b);
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : llvm.Ptr = llvm.alloca<6>()
        |     %1 : llvm.Ptr = llvm.alloca<6>()
        |     %2 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : llvm.Ptr = llvm.load(%2)
        |     %4 : Index = 0
        |     %5 : Nil = llvm.br<"loop_header0">()
        |     %6 : Nil = llvm.label<"loop_header0">()
        |     %7 : Nil = llvm.phi<"entry", "loop_exit1">(%4, %34)
        |     %8 : Index = 2
        |     %9 : llvm.Int<1> = llvm.icmp<"slt">(%7, %8)
        |     %10 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%9)
        |     %11 : Nil = llvm.label<"loop_body0">()
        |     %12 : Nil = llvm.br<"loop_header1">()
        |     %13 : Nil = llvm.label<"loop_header1">()
        |     %14 : Nil = llvm.phi<"loop_body0", "loop_body1">(%4, %30)
        |     %15 : Index = 3
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%14, %15)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 3
        |     %20 : llvm.Int<64> = llvm.mul(%7, %19)
        |     %21 : llvm.Int<64> = llvm.add(%20, %14)
        |     %22 : llvm.Ptr = llvm.gep(%3, %21)
        |     %23 : llvm.Float = llvm.load(%22)
        |     %24 : Index = 2
        |     %25 : llvm.Int<64> = llvm.mul(%14, %24)
        |     %26 : llvm.Int<64> = llvm.add(%25, %7)
        |     %27 : llvm.Ptr = llvm.gep(%1, %26)
        |     %28 : Nil = llvm.store(%23, %27)
        |     %29 : Index = 1
        |     %30 : llvm.Int<64> = llvm.add(%14, %29)
        |     %31 : Nil = llvm.br<"loop_header1">()
        |     %32 : Nil = llvm.label<"loop_exit1">()
        |     %33 : Index = 1
        |     %34 : llvm.Int<64> = llvm.add(%7, %33)
        |     %35 : Nil = llvm.br<"loop_header0">()
        |     %36 : Nil = llvm.label<"loop_exit0">()
        |     %37 : llvm.Ptr = llvm.alloca<6>()
        |     %38 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %39 : llvm.Ptr = llvm.load(%38)
        |     %40 : Nil = llvm.br<"loop_header2">()
        |     %41 : Nil = llvm.label<"loop_header2">()
        |     %42 : Nil = llvm.phi<"loop_exit0", "loop_exit3">(%4, %67)
        |     %43 : llvm.Int<1> = llvm.icmp<"slt">(%42, %8)
        |     %44 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%43)
        |     %45 : Nil = llvm.label<"loop_body2">()
        |     %46 : Nil = llvm.br<"loop_header3">()
        |     %47 : Nil = llvm.label<"loop_header3">()
        |     %48 : Nil = llvm.phi<"loop_body2", "loop_body3">(%4, %63)
        |     %49 : llvm.Int<1> = llvm.icmp<"slt">(%48, %15)
        |     %50 : Nil = llvm.cond_br<"loop_body3", "loop_exit3">(%49)
        |     %51 : Nil = llvm.label<"loop_body3">()
        |     %52 : Index = 3
        |     %53 : llvm.Int<64> = llvm.mul(%42, %52)
        |     %54 : llvm.Int<64> = llvm.add(%53, %48)
        |     %55 : llvm.Ptr = llvm.gep(%39, %54)
        |     %56 : llvm.Float = llvm.load(%55)
        |     %57 : Index = 2
        |     %58 : llvm.Int<64> = llvm.mul(%48, %57)
        |     %59 : llvm.Int<64> = llvm.add(%58, %42)
        |     %60 : llvm.Ptr = llvm.gep(%37, %59)
        |     %61 : Nil = llvm.store(%56, %60)
        |     %62 : Index = 1
        |     %63 : llvm.Int<64> = llvm.add(%48, %62)
        |     %64 : Nil = llvm.br<"loop_header3">()
        |     %65 : Nil = llvm.label<"loop_exit3">()
        |     %66 : Index = 1
        |     %67 : llvm.Int<64> = llvm.add(%42, %66)
        |     %68 : Nil = llvm.br<"loop_header2">()
        |     %69 : Nil = llvm.label<"loop_exit2">()
        |     %70 : Nil = llvm.br<"loop_header4">()
        |     %71 : Nil = llvm.label<"loop_header4">()
        |     %72 : Nil = llvm.phi<"loop_exit2", "loop_exit5">(%4, %103)
        |     %73 : llvm.Int<1> = llvm.icmp<"slt">(%72, %15)
        |     %74 : Nil = llvm.cond_br<"loop_body4", "loop_exit4">(%73)
        |     %75 : Nil = llvm.label<"loop_body4">()
        |     %76 : Nil = llvm.br<"loop_header5">()
        |     %77 : Nil = llvm.label<"loop_header5">()
        |     %78 : Nil = llvm.phi<"loop_body4", "loop_body5">(%4, %99)
        |     %79 : llvm.Int<1> = llvm.icmp<"slt">(%78, %8)
        |     %80 : Nil = llvm.cond_br<"loop_body5", "loop_exit5">(%79)
        |     %81 : Nil = llvm.label<"loop_body5">()
        |     %82 : Index = 2
        |     %83 : llvm.Int<64> = llvm.mul(%72, %82)
        |     %84 : llvm.Int<64> = llvm.add(%83, %78)
        |     %85 : llvm.Ptr = llvm.gep(%1, %84)
        |     %86 : llvm.Float = llvm.load(%85)
        |     %87 : Index = 2
        |     %88 : llvm.Int<64> = llvm.mul(%72, %87)
        |     %89 : llvm.Int<64> = llvm.add(%88, %78)
        |     %90 : llvm.Ptr = llvm.gep(%37, %89)
        |     %91 : llvm.Float = llvm.load(%90)
        |     %92 : llvm.Float = llvm.fmul(%86, %91)
        |     %93 : Index = 2
        |     %94 : llvm.Int<64> = llvm.mul(%72, %93)
        |     %95 : llvm.Int<64> = llvm.add(%94, %78)
        |     %96 : llvm.Ptr = llvm.gep(%0, %95)
        |     %97 : Nil = llvm.store(%92, %96)
        |     %98 : Index = 1
        |     %99 : llvm.Int<64> = llvm.add(%78, %98)
        |     %100 : Nil = llvm.br<"loop_header5">()
        |     %101 : Nil = llvm.label<"loop_exit5">()
        |     %102 : Index = 1
        |     %103 : llvm.Int<64> = llvm.add(%72, %102)
        |     %104 : Nil = llvm.br<"loop_header4">()
        |     %105 : Nil = llvm.label<"loop_exit4">()
        |     %106 : Index = 6
        |     %107 : Nil = llvm.call<"print_memref">([%0, %106])
        |     %108 : Nil = return(%107)
    """)
    assert result == expected
