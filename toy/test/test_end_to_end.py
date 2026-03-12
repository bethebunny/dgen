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
        |     %1 : Index = 6
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(%2)
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
        |     %2 : Index = 0
        |     %3 : Nil = llvm.br<"loop_header0">()
        |     %4 : Nil = llvm.label<"loop_header0">()
        |     %5 : Nil = llvm.phi<"entry", "loop_exit1">(%2, %32)
        |     %6 : Index = 2
        |     %7 : llvm.Int<1> = llvm.icmp<"slt">(%5, %6)
        |     %8 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%7)
        |     %9 : Nil = llvm.label<"loop_body0">()
        |     %10 : Nil = llvm.br<"loop_header1">()
        |     %11 : Nil = llvm.label<"loop_header1">()
        |     %12 : Nil = llvm.phi<"loop_body0", "loop_body1">(%2, %28)
        |     %13 : Index = 3
        |     %14 : llvm.Int<1> = llvm.icmp<"slt">(%12, %13)
        |     %15 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%14)
        |     %16 : Nil = llvm.label<"loop_body1">()
        |     %17 : Index = 3
        |     %18 : llvm.Int<64> = llvm.mul(%5, %17)
        |     %19 : llvm.Int<64> = llvm.add(%18, %12)
        |     %20 : llvm.Ptr = llvm.gep(%1, %19)
        |     %21 : llvm.Float = llvm.load(%20)
        |     %22 : Index = 2
        |     %23 : llvm.Int<64> = llvm.mul(%12, %22)
        |     %24 : llvm.Int<64> = llvm.add(%23, %5)
        |     %25 : llvm.Ptr = llvm.gep(%0, %24)
        |     %26 : Nil = llvm.store(%21, %25)
        |     %27 : Index = 1
        |     %28 : llvm.Int<64> = llvm.add(%12, %27)
        |     %29 : Nil = llvm.br<"loop_header1">()
        |     %30 : Nil = llvm.label<"loop_exit1">()
        |     %31 : Index = 1
        |     %32 : llvm.Int<64> = llvm.add(%5, %31)
        |     %33 : Nil = llvm.br<"loop_header0">()
        |     %34 : Nil = llvm.label<"loop_exit0">()
        |     %35 : Index = 6
        |     %36 : Nil = llvm.call<"print_memref">([%0, %35])
        |     %37 : Nil = return(%36)
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
        |     %2 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %38)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_body1">(%3, %34)
        |     %14 : llvm.Int<1> = llvm.icmp<"slt">(%13, %7)
        |     %15 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%14)
        |     %16 : Nil = llvm.label<"loop_body1">()
        |     %17 : Index = 2
        |     %18 : llvm.Int<64> = llvm.mul(%6, %17)
        |     %19 : llvm.Int<64> = llvm.add(%18, %13)
        |     %20 : llvm.Ptr = llvm.gep(%1, %19)
        |     %21 : llvm.Float = llvm.load(%20)
        |     %22 : Index = 2
        |     %23 : llvm.Int<64> = llvm.mul(%6, %22)
        |     %24 : llvm.Int<64> = llvm.add(%23, %13)
        |     %25 : llvm.Ptr = llvm.gep(%2, %24)
        |     %26 : llvm.Float = llvm.load(%25)
        |     %27 : llvm.Float = llvm.fmul(%21, %26)
        |     %28 : Index = 2
        |     %29 : llvm.Int<64> = llvm.mul(%6, %28)
        |     %30 : llvm.Int<64> = llvm.add(%29, %13)
        |     %31 : llvm.Ptr = llvm.gep(%0, %30)
        |     %32 : Nil = llvm.store(%27, %31)
        |     %33 : Index = 1
        |     %34 : llvm.Int<64> = llvm.add(%13, %33)
        |     %35 : Nil = llvm.br<"loop_header1">()
        |     %36 : Nil = llvm.label<"loop_exit1">()
        |     %37 : Index = 1
        |     %38 : llvm.Int<64> = llvm.add(%6, %37)
        |     %39 : Nil = llvm.br<"loop_header0">()
        |     %40 : Nil = llvm.label<"loop_exit0">()
        |     %41 : Index = 4
        |     %42 : Nil = llvm.call<"print_memref">([%0, %41])
        |     %43 : Nil = return(%42)
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
        |     %2 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %38)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_body1">(%3, %34)
        |     %14 : llvm.Int<1> = llvm.icmp<"slt">(%13, %7)
        |     %15 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%14)
        |     %16 : Nil = llvm.label<"loop_body1">()
        |     %17 : Index = 2
        |     %18 : llvm.Int<64> = llvm.mul(%6, %17)
        |     %19 : llvm.Int<64> = llvm.add(%18, %13)
        |     %20 : llvm.Ptr = llvm.gep(%1, %19)
        |     %21 : llvm.Float = llvm.load(%20)
        |     %22 : Index = 2
        |     %23 : llvm.Int<64> = llvm.mul(%6, %22)
        |     %24 : llvm.Int<64> = llvm.add(%23, %13)
        |     %25 : llvm.Ptr = llvm.gep(%2, %24)
        |     %26 : llvm.Float = llvm.load(%25)
        |     %27 : llvm.Float = llvm.fadd(%21, %26)
        |     %28 : Index = 2
        |     %29 : llvm.Int<64> = llvm.mul(%6, %28)
        |     %30 : llvm.Int<64> = llvm.add(%29, %13)
        |     %31 : llvm.Ptr = llvm.gep(%0, %30)
        |     %32 : Nil = llvm.store(%27, %31)
        |     %33 : Index = 1
        |     %34 : llvm.Int<64> = llvm.add(%13, %33)
        |     %35 : Nil = llvm.br<"loop_header1">()
        |     %36 : Nil = llvm.label<"loop_exit1">()
        |     %37 : Index = 1
        |     %38 : llvm.Int<64> = llvm.add(%6, %37)
        |     %39 : Nil = llvm.br<"loop_header0">()
        |     %40 : Nil = llvm.label<"loop_exit0">()
        |     %41 : Index = 4
        |     %42 : Nil = llvm.call<"print_memref">([%0, %41])
        |     %43 : Nil = return(%42)
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
        |     %1 : Index = 8
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(%2)
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
        |     %2 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %57)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%3, %53)
        |     %14 : llvm.Int<1> = llvm.icmp<"slt">(%13, %7)
        |     %15 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%14)
        |     %16 : Nil = llvm.label<"loop_body1">()
        |     %17 : Nil = llvm.br<"loop_header2">()
        |     %18 : Nil = llvm.label<"loop_header2">()
        |     %19 : Nil = llvm.phi<"loop_body1", "loop_body2">(%3, %49)
        |     %20 : llvm.Int<1> = llvm.icmp<"slt">(%19, %7)
        |     %21 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%20)
        |     %22 : Nil = llvm.label<"loop_body2">()
        |     %23 : Index = 4
        |     %24 : llvm.Int<64> = llvm.mul(%6, %23)
        |     %25 : Index = 2
        |     %26 : llvm.Int<64> = llvm.mul(%13, %25)
        |     %27 : llvm.Int<64> = llvm.add(%24, %26)
        |     %28 : llvm.Int<64> = llvm.add(%27, %19)
        |     %29 : llvm.Ptr = llvm.gep(%1, %28)
        |     %30 : llvm.Float = llvm.load(%29)
        |     %31 : Index = 4
        |     %32 : llvm.Int<64> = llvm.mul(%6, %31)
        |     %33 : Index = 2
        |     %34 : llvm.Int<64> = llvm.mul(%13, %33)
        |     %35 : llvm.Int<64> = llvm.add(%32, %34)
        |     %36 : llvm.Int<64> = llvm.add(%35, %19)
        |     %37 : llvm.Ptr = llvm.gep(%2, %36)
        |     %38 : llvm.Float = llvm.load(%37)
        |     %39 : llvm.Float = llvm.fadd(%30, %38)
        |     %40 : Index = 4
        |     %41 : llvm.Int<64> = llvm.mul(%6, %40)
        |     %42 : Index = 2
        |     %43 : llvm.Int<64> = llvm.mul(%13, %42)
        |     %44 : llvm.Int<64> = llvm.add(%41, %43)
        |     %45 : llvm.Int<64> = llvm.add(%44, %19)
        |     %46 : llvm.Ptr = llvm.gep(%0, %45)
        |     %47 : Nil = llvm.store(%39, %46)
        |     %48 : Index = 1
        |     %49 : llvm.Int<64> = llvm.add(%19, %48)
        |     %50 : Nil = llvm.br<"loop_header2">()
        |     %51 : Nil = llvm.label<"loop_exit2">()
        |     %52 : Index = 1
        |     %53 : llvm.Int<64> = llvm.add(%13, %52)
        |     %54 : Nil = llvm.br<"loop_header1">()
        |     %55 : Nil = llvm.label<"loop_exit1">()
        |     %56 : Index = 1
        |     %57 : llvm.Int<64> = llvm.add(%6, %56)
        |     %58 : Nil = llvm.br<"loop_header0">()
        |     %59 : Nil = llvm.label<"loop_exit0">()
        |     %60 : Index = 8
        |     %61 : Nil = llvm.call<"print_memref">([%0, %60])
        |     %62 : Nil = return(%61)
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
        |     %2 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %57)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%3, %53)
        |     %14 : llvm.Int<1> = llvm.icmp<"slt">(%13, %7)
        |     %15 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%14)
        |     %16 : Nil = llvm.label<"loop_body1">()
        |     %17 : Nil = llvm.br<"loop_header2">()
        |     %18 : Nil = llvm.label<"loop_header2">()
        |     %19 : Nil = llvm.phi<"loop_body1", "loop_body2">(%3, %49)
        |     %20 : llvm.Int<1> = llvm.icmp<"slt">(%19, %7)
        |     %21 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%20)
        |     %22 : Nil = llvm.label<"loop_body2">()
        |     %23 : Index = 4
        |     %24 : llvm.Int<64> = llvm.mul(%6, %23)
        |     %25 : Index = 2
        |     %26 : llvm.Int<64> = llvm.mul(%13, %25)
        |     %27 : llvm.Int<64> = llvm.add(%24, %26)
        |     %28 : llvm.Int<64> = llvm.add(%27, %19)
        |     %29 : llvm.Ptr = llvm.gep(%1, %28)
        |     %30 : llvm.Float = llvm.load(%29)
        |     %31 : Index = 4
        |     %32 : llvm.Int<64> = llvm.mul(%6, %31)
        |     %33 : Index = 2
        |     %34 : llvm.Int<64> = llvm.mul(%13, %33)
        |     %35 : llvm.Int<64> = llvm.add(%32, %34)
        |     %36 : llvm.Int<64> = llvm.add(%35, %19)
        |     %37 : llvm.Ptr = llvm.gep(%2, %36)
        |     %38 : llvm.Float = llvm.load(%37)
        |     %39 : llvm.Float = llvm.fmul(%30, %38)
        |     %40 : Index = 4
        |     %41 : llvm.Int<64> = llvm.mul(%6, %40)
        |     %42 : Index = 2
        |     %43 : llvm.Int<64> = llvm.mul(%13, %42)
        |     %44 : llvm.Int<64> = llvm.add(%41, %43)
        |     %45 : llvm.Int<64> = llvm.add(%44, %19)
        |     %46 : llvm.Ptr = llvm.gep(%0, %45)
        |     %47 : Nil = llvm.store(%39, %46)
        |     %48 : Index = 1
        |     %49 : llvm.Int<64> = llvm.add(%19, %48)
        |     %50 : Nil = llvm.br<"loop_header2">()
        |     %51 : Nil = llvm.label<"loop_exit2">()
        |     %52 : Index = 1
        |     %53 : llvm.Int<64> = llvm.add(%13, %52)
        |     %54 : Nil = llvm.br<"loop_header1">()
        |     %55 : Nil = llvm.label<"loop_exit1">()
        |     %56 : Index = 1
        |     %57 : llvm.Int<64> = llvm.add(%6, %56)
        |     %58 : Nil = llvm.br<"loop_header0">()
        |     %59 : Nil = llvm.label<"loop_exit0">()
        |     %60 : Index = 8
        |     %61 : Nil = llvm.call<"print_memref">([%0, %60])
        |     %62 : Nil = return(%61)
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
        |     %1 : Index = 6
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(%2)
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
        |     %1 : Index = 6
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(%2)
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
        |     %26 : llvm.Ptr = llvm.gep(%1, %25)
        |     %27 : Nil = llvm.store(%22, %26)
        |     %28 : Index = 1
        |     %29 : llvm.Int<64> = llvm.add(%13, %28)
        |     %30 : Nil = llvm.br<"loop_header1">()
        |     %31 : Nil = llvm.label<"loop_exit1">()
        |     %32 : Index = 1
        |     %33 : llvm.Int<64> = llvm.add(%6, %32)
        |     %34 : Nil = llvm.br<"loop_header0">()
        |     %35 : Nil = llvm.label<"loop_exit0">()
        |     %36 : llvm.Ptr = llvm.alloca<6>()
        |     %37 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %38 : Nil = llvm.br<"loop_header2">()
        |     %39 : Nil = llvm.label<"loop_header2">()
        |     %40 : Nil = llvm.phi<"loop_exit0", "loop_exit3">(%3, %65)
        |     %41 : llvm.Int<1> = llvm.icmp<"slt">(%40, %7)
        |     %42 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%41)
        |     %43 : Nil = llvm.label<"loop_body2">()
        |     %44 : Nil = llvm.br<"loop_header3">()
        |     %45 : Nil = llvm.label<"loop_header3">()
        |     %46 : Nil = llvm.phi<"loop_body2", "loop_body3">(%3, %61)
        |     %47 : llvm.Int<1> = llvm.icmp<"slt">(%46, %14)
        |     %48 : Nil = llvm.cond_br<"loop_body3", "loop_exit3">(%47)
        |     %49 : Nil = llvm.label<"loop_body3">()
        |     %50 : Index = 3
        |     %51 : llvm.Int<64> = llvm.mul(%40, %50)
        |     %52 : llvm.Int<64> = llvm.add(%51, %46)
        |     %53 : llvm.Ptr = llvm.gep(%37, %52)
        |     %54 : llvm.Float = llvm.load(%53)
        |     %55 : Index = 2
        |     %56 : llvm.Int<64> = llvm.mul(%46, %55)
        |     %57 : llvm.Int<64> = llvm.add(%56, %40)
        |     %58 : llvm.Ptr = llvm.gep(%36, %57)
        |     %59 : Nil = llvm.store(%54, %58)
        |     %60 : Index = 1
        |     %61 : llvm.Int<64> = llvm.add(%46, %60)
        |     %62 : Nil = llvm.br<"loop_header3">()
        |     %63 : Nil = llvm.label<"loop_exit3">()
        |     %64 : Index = 1
        |     %65 : llvm.Int<64> = llvm.add(%40, %64)
        |     %66 : Nil = llvm.br<"loop_header2">()
        |     %67 : Nil = llvm.label<"loop_exit2">()
        |     %68 : Nil = llvm.br<"loop_header4">()
        |     %69 : Nil = llvm.label<"loop_header4">()
        |     %70 : Nil = llvm.phi<"loop_exit2", "loop_exit5">(%3, %101)
        |     %71 : llvm.Int<1> = llvm.icmp<"slt">(%70, %14)
        |     %72 : Nil = llvm.cond_br<"loop_body4", "loop_exit4">(%71)
        |     %73 : Nil = llvm.label<"loop_body4">()
        |     %74 : Nil = llvm.br<"loop_header5">()
        |     %75 : Nil = llvm.label<"loop_header5">()
        |     %76 : Nil = llvm.phi<"loop_body4", "loop_body5">(%3, %97)
        |     %77 : llvm.Int<1> = llvm.icmp<"slt">(%76, %7)
        |     %78 : Nil = llvm.cond_br<"loop_body5", "loop_exit5">(%77)
        |     %79 : Nil = llvm.label<"loop_body5">()
        |     %80 : Index = 2
        |     %81 : llvm.Int<64> = llvm.mul(%70, %80)
        |     %82 : llvm.Int<64> = llvm.add(%81, %76)
        |     %83 : llvm.Ptr = llvm.gep(%1, %82)
        |     %84 : llvm.Float = llvm.load(%83)
        |     %85 : Index = 2
        |     %86 : llvm.Int<64> = llvm.mul(%70, %85)
        |     %87 : llvm.Int<64> = llvm.add(%86, %76)
        |     %88 : llvm.Ptr = llvm.gep(%36, %87)
        |     %89 : llvm.Float = llvm.load(%88)
        |     %90 : llvm.Float = llvm.fmul(%84, %89)
        |     %91 : Index = 2
        |     %92 : llvm.Int<64> = llvm.mul(%70, %91)
        |     %93 : llvm.Int<64> = llvm.add(%92, %76)
        |     %94 : llvm.Ptr = llvm.gep(%0, %93)
        |     %95 : Nil = llvm.store(%90, %94)
        |     %96 : Index = 1
        |     %97 : llvm.Int<64> = llvm.add(%76, %96)
        |     %98 : Nil = llvm.br<"loop_header5">()
        |     %99 : Nil = llvm.label<"loop_exit5">()
        |     %100 : Index = 1
        |     %101 : llvm.Int<64> = llvm.add(%70, %100)
        |     %102 : Nil = llvm.br<"loop_header4">()
        |     %103 : Nil = llvm.label<"loop_exit4">()
        |     %104 : Index = 6
        |     %105 : Nil = llvm.call<"print_memref">([%0, %104])
        |     %106 : Nil = return(%105)
    """)
    assert result == expected
