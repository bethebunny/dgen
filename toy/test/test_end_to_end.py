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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : index = 6
        |     %2 : () = llvm.call<"print_memref">([%0, %1])
        |     %3 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : () = llvm.alloca<6>()
        |     %2 : index = 0
        |     %3 : () = llvm.br<"loop_header0">()
        |     %4 : () = llvm.label<"loop_header0">()
        |     %5 : () = llvm.phi<["entry", "loop_exit1"]>([%2, %33])
        |     %6 : index = 2
        |     %7 : () = llvm.icmp<"slt">(%5, %6)
        |     %8 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%7)
        |     %9 : () = llvm.label<"loop_body0">()
        |     %10 : index = 0
        |     %11 : () = llvm.br<"loop_header1">()
        |     %12 : () = llvm.label<"loop_header1">()
        |     %13 : () = llvm.phi<["loop_body0", "loop_body1"]>([%10, %29])
        |     %14 : index = 3
        |     %15 : () = llvm.icmp<"slt">(%13, %14)
        |     %16 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%15)
        |     %17 : () = llvm.label<"loop_body1">()
        |     %18 : index = 3
        |     %19 : () = llvm.mul(%5, %18)
        |     %20 : () = llvm.add(%19, %13)
        |     %21 : () = llvm.gep(%0, %20)
        |     %22 : () = llvm.load(%21)
        |     %23 : index = 2
        |     %24 : () = llvm.mul(%13, %23)
        |     %25 : () = llvm.add(%24, %5)
        |     %26 : () = llvm.gep(%1, %25)
        |     %27 : () = llvm.store(%22, %26)
        |     %28 : index = 1
        |     %29 : () = llvm.add(%13, %28)
        |     %30 : () = llvm.br<"loop_header1">()
        |     %31 : () = llvm.label<"loop_exit1">()
        |     %32 : index = 1
        |     %33 : () = llvm.add(%5, %32)
        |     %34 : () = llvm.br<"loop_header0">()
        |     %35 : () = llvm.label<"loop_exit0">()
        |     %36 : index = 6
        |     %37 : () = llvm.call<"print_memref">([%1, %36])
        |     %38 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : () = llvm.alloca<4>()
        |     %3 : index = 0
        |     %4 : () = llvm.br<"loop_header0">()
        |     %5 : () = llvm.label<"loop_header0">()
        |     %6 : () = llvm.phi<["entry", "loop_exit1"]>([%3, %40])
        |     %7 : index = 2
        |     %8 : () = llvm.icmp<"slt">(%6, %7)
        |     %9 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : () = llvm.label<"loop_body0">()
        |     %11 : index = 0
        |     %12 : () = llvm.br<"loop_header1">()
        |     %13 : () = llvm.label<"loop_header1">()
        |     %14 : () = llvm.phi<["loop_body0", "loop_body1"]>([%11, %36])
        |     %15 : index = 2
        |     %16 : () = llvm.icmp<"slt">(%14, %15)
        |     %17 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : () = llvm.label<"loop_body1">()
        |     %19 : index = 2
        |     %20 : () = llvm.mul(%6, %19)
        |     %21 : () = llvm.add(%20, %14)
        |     %22 : () = llvm.gep(%0, %21)
        |     %23 : () = llvm.load(%22)
        |     %24 : index = 2
        |     %25 : () = llvm.mul(%6, %24)
        |     %26 : () = llvm.add(%25, %14)
        |     %27 : () = llvm.gep(%1, %26)
        |     %28 : () = llvm.load(%27)
        |     %29 : () = llvm.fmul(%23, %28)
        |     %30 : index = 2
        |     %31 : () = llvm.mul(%6, %30)
        |     %32 : () = llvm.add(%31, %14)
        |     %33 : () = llvm.gep(%2, %32)
        |     %34 : () = llvm.store(%29, %33)
        |     %35 : index = 1
        |     %36 : () = llvm.add(%14, %35)
        |     %37 : () = llvm.br<"loop_header1">()
        |     %38 : () = llvm.label<"loop_exit1">()
        |     %39 : index = 1
        |     %40 : () = llvm.add(%6, %39)
        |     %41 : () = llvm.br<"loop_header0">()
        |     %42 : () = llvm.label<"loop_exit0">()
        |     %43 : index = 4
        |     %44 : () = llvm.call<"print_memref">([%2, %43])
        |     %45 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : () = llvm.alloca<4>()
        |     %3 : index = 0
        |     %4 : () = llvm.br<"loop_header0">()
        |     %5 : () = llvm.label<"loop_header0">()
        |     %6 : () = llvm.phi<["entry", "loop_exit1"]>([%3, %40])
        |     %7 : index = 2
        |     %8 : () = llvm.icmp<"slt">(%6, %7)
        |     %9 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : () = llvm.label<"loop_body0">()
        |     %11 : index = 0
        |     %12 : () = llvm.br<"loop_header1">()
        |     %13 : () = llvm.label<"loop_header1">()
        |     %14 : () = llvm.phi<["loop_body0", "loop_body1"]>([%11, %36])
        |     %15 : index = 2
        |     %16 : () = llvm.icmp<"slt">(%14, %15)
        |     %17 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : () = llvm.label<"loop_body1">()
        |     %19 : index = 2
        |     %20 : () = llvm.mul(%6, %19)
        |     %21 : () = llvm.add(%20, %14)
        |     %22 : () = llvm.gep(%0, %21)
        |     %23 : () = llvm.load(%22)
        |     %24 : index = 2
        |     %25 : () = llvm.mul(%6, %24)
        |     %26 : () = llvm.add(%25, %14)
        |     %27 : () = llvm.gep(%1, %26)
        |     %28 : () = llvm.load(%27)
        |     %29 : () = llvm.fadd(%23, %28)
        |     %30 : index = 2
        |     %31 : () = llvm.mul(%6, %30)
        |     %32 : () = llvm.add(%31, %14)
        |     %33 : () = llvm.gep(%2, %32)
        |     %34 : () = llvm.store(%29, %33)
        |     %35 : index = 1
        |     %36 : () = llvm.add(%14, %35)
        |     %37 : () = llvm.br<"loop_header1">()
        |     %38 : () = llvm.label<"loop_exit1">()
        |     %39 : index = 1
        |     %40 : () = llvm.add(%6, %39)
        |     %41 : () = llvm.br<"loop_header0">()
        |     %42 : () = llvm.label<"loop_exit0">()
        |     %43 : index = 4
        |     %44 : () = llvm.call<"print_memref">([%2, %43])
        |     %45 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : index = 8
        |     %2 : () = llvm.call<"print_memref">([%0, %1])
        |     %3 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : () = llvm.alloca<8>()
        |     %3 : index = 0
        |     %4 : () = llvm.br<"loop_header0">()
        |     %5 : () = llvm.label<"loop_header0">()
        |     %6 : () = llvm.phi<["entry", "loop_exit1"]>([%3, %61])
        |     %7 : index = 2
        |     %8 : () = llvm.icmp<"slt">(%6, %7)
        |     %9 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : () = llvm.label<"loop_body0">()
        |     %11 : index = 0
        |     %12 : () = llvm.br<"loop_header1">()
        |     %13 : () = llvm.label<"loop_header1">()
        |     %14 : () = llvm.phi<["loop_body0", "loop_exit2"]>([%11, %57])
        |     %15 : index = 2
        |     %16 : () = llvm.icmp<"slt">(%14, %15)
        |     %17 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : () = llvm.label<"loop_body1">()
        |     %19 : index = 0
        |     %20 : () = llvm.br<"loop_header2">()
        |     %21 : () = llvm.label<"loop_header2">()
        |     %22 : () = llvm.phi<["loop_body1", "loop_body2"]>([%19, %53])
        |     %23 : index = 2
        |     %24 : () = llvm.icmp<"slt">(%22, %23)
        |     %25 : () = llvm.cond_br<"loop_body2", "loop_exit2">(%24)
        |     %26 : () = llvm.label<"loop_body2">()
        |     %27 : index = 4
        |     %28 : () = llvm.mul(%6, %27)
        |     %29 : index = 2
        |     %30 : () = llvm.mul(%14, %29)
        |     %31 : () = llvm.add(%28, %30)
        |     %32 : () = llvm.add(%31, %22)
        |     %33 : () = llvm.gep(%0, %32)
        |     %34 : () = llvm.load(%33)
        |     %35 : index = 4
        |     %36 : () = llvm.mul(%6, %35)
        |     %37 : index = 2
        |     %38 : () = llvm.mul(%14, %37)
        |     %39 : () = llvm.add(%36, %38)
        |     %40 : () = llvm.add(%39, %22)
        |     %41 : () = llvm.gep(%1, %40)
        |     %42 : () = llvm.load(%41)
        |     %43 : () = llvm.fadd(%34, %42)
        |     %44 : index = 4
        |     %45 : () = llvm.mul(%6, %44)
        |     %46 : index = 2
        |     %47 : () = llvm.mul(%14, %46)
        |     %48 : () = llvm.add(%45, %47)
        |     %49 : () = llvm.add(%48, %22)
        |     %50 : () = llvm.gep(%2, %49)
        |     %51 : () = llvm.store(%43, %50)
        |     %52 : index = 1
        |     %53 : () = llvm.add(%22, %52)
        |     %54 : () = llvm.br<"loop_header2">()
        |     %55 : () = llvm.label<"loop_exit2">()
        |     %56 : index = 1
        |     %57 : () = llvm.add(%14, %56)
        |     %58 : () = llvm.br<"loop_header1">()
        |     %59 : () = llvm.label<"loop_exit1">()
        |     %60 : index = 1
        |     %61 : () = llvm.add(%6, %60)
        |     %62 : () = llvm.br<"loop_header0">()
        |     %63 : () = llvm.label<"loop_exit0">()
        |     %64 : index = 8
        |     %65 : () = llvm.call<"print_memref">([%2, %64])
        |     %66 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : () = llvm.alloca<8>()
        |     %3 : index = 0
        |     %4 : () = llvm.br<"loop_header0">()
        |     %5 : () = llvm.label<"loop_header0">()
        |     %6 : () = llvm.phi<["entry", "loop_exit1"]>([%3, %61])
        |     %7 : index = 2
        |     %8 : () = llvm.icmp<"slt">(%6, %7)
        |     %9 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : () = llvm.label<"loop_body0">()
        |     %11 : index = 0
        |     %12 : () = llvm.br<"loop_header1">()
        |     %13 : () = llvm.label<"loop_header1">()
        |     %14 : () = llvm.phi<["loop_body0", "loop_exit2"]>([%11, %57])
        |     %15 : index = 2
        |     %16 : () = llvm.icmp<"slt">(%14, %15)
        |     %17 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : () = llvm.label<"loop_body1">()
        |     %19 : index = 0
        |     %20 : () = llvm.br<"loop_header2">()
        |     %21 : () = llvm.label<"loop_header2">()
        |     %22 : () = llvm.phi<["loop_body1", "loop_body2"]>([%19, %53])
        |     %23 : index = 2
        |     %24 : () = llvm.icmp<"slt">(%22, %23)
        |     %25 : () = llvm.cond_br<"loop_body2", "loop_exit2">(%24)
        |     %26 : () = llvm.label<"loop_body2">()
        |     %27 : index = 4
        |     %28 : () = llvm.mul(%6, %27)
        |     %29 : index = 2
        |     %30 : () = llvm.mul(%14, %29)
        |     %31 : () = llvm.add(%28, %30)
        |     %32 : () = llvm.add(%31, %22)
        |     %33 : () = llvm.gep(%0, %32)
        |     %34 : () = llvm.load(%33)
        |     %35 : index = 4
        |     %36 : () = llvm.mul(%6, %35)
        |     %37 : index = 2
        |     %38 : () = llvm.mul(%14, %37)
        |     %39 : () = llvm.add(%36, %38)
        |     %40 : () = llvm.add(%39, %22)
        |     %41 : () = llvm.gep(%1, %40)
        |     %42 : () = llvm.load(%41)
        |     %43 : () = llvm.fmul(%34, %42)
        |     %44 : index = 4
        |     %45 : () = llvm.mul(%6, %44)
        |     %46 : index = 2
        |     %47 : () = llvm.mul(%14, %46)
        |     %48 : () = llvm.add(%45, %47)
        |     %49 : () = llvm.add(%48, %22)
        |     %50 : () = llvm.gep(%2, %49)
        |     %51 : () = llvm.store(%43, %50)
        |     %52 : index = 1
        |     %53 : () = llvm.add(%22, %52)
        |     %54 : () = llvm.br<"loop_header2">()
        |     %55 : () = llvm.label<"loop_exit2">()
        |     %56 : index = 1
        |     %57 : () = llvm.add(%14, %56)
        |     %58 : () = llvm.br<"loop_header1">()
        |     %59 : () = llvm.label<"loop_exit1">()
        |     %60 : index = 1
        |     %61 : () = llvm.add(%6, %60)
        |     %62 : () = llvm.br<"loop_header0">()
        |     %63 : () = llvm.label<"loop_exit0">()
        |     %64 : index = 8
        |     %65 : () = llvm.call<"print_memref">([%2, %64])
        |     %66 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : index = 6
        |     %2 : () = llvm.call<"print_memref">([%0, %1])
        |     %3 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : index = 6
        |     %2 : () = llvm.call<"print_memref">([%0, %1])
        |     %3 : () = return(())
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
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : () = llvm.alloca<6>()
        |     %3 : index = 0
        |     %4 : () = llvm.br<"loop_header0">()
        |     %5 : () = llvm.label<"loop_header0">()
        |     %6 : () = llvm.phi<["entry", "loop_exit1"]>([%3, %34])
        |     %7 : index = 2
        |     %8 : () = llvm.icmp<"slt">(%6, %7)
        |     %9 : () = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : () = llvm.label<"loop_body0">()
        |     %11 : index = 0
        |     %12 : () = llvm.br<"loop_header1">()
        |     %13 : () = llvm.label<"loop_header1">()
        |     %14 : () = llvm.phi<["loop_body0", "loop_body1"]>([%11, %30])
        |     %15 : index = 3
        |     %16 : () = llvm.icmp<"slt">(%14, %15)
        |     %17 : () = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : () = llvm.label<"loop_body1">()
        |     %19 : index = 3
        |     %20 : () = llvm.mul(%6, %19)
        |     %21 : () = llvm.add(%20, %14)
        |     %22 : () = llvm.gep(%0, %21)
        |     %23 : () = llvm.load(%22)
        |     %24 : index = 2
        |     %25 : () = llvm.mul(%14, %24)
        |     %26 : () = llvm.add(%25, %6)
        |     %27 : () = llvm.gep(%2, %26)
        |     %28 : () = llvm.store(%23, %27)
        |     %29 : index = 1
        |     %30 : () = llvm.add(%14, %29)
        |     %31 : () = llvm.br<"loop_header1">()
        |     %32 : () = llvm.label<"loop_exit1">()
        |     %33 : index = 1
        |     %34 : () = llvm.add(%6, %33)
        |     %35 : () = llvm.br<"loop_header0">()
        |     %36 : () = llvm.label<"loop_exit0">()
        |     %37 : () = llvm.alloca<6>()
        |     %38 : index = 0
        |     %39 : () = llvm.br<"loop_header2">()
        |     %40 : () = llvm.label<"loop_header2">()
        |     %41 : () = llvm.phi<["loop_exit0", "loop_exit3"]>([%38, %69])
        |     %42 : index = 2
        |     %43 : () = llvm.icmp<"slt">(%41, %42)
        |     %44 : () = llvm.cond_br<"loop_body2", "loop_exit2">(%43)
        |     %45 : () = llvm.label<"loop_body2">()
        |     %46 : index = 0
        |     %47 : () = llvm.br<"loop_header3">()
        |     %48 : () = llvm.label<"loop_header3">()
        |     %49 : () = llvm.phi<["loop_body2", "loop_body3"]>([%46, %65])
        |     %50 : index = 3
        |     %51 : () = llvm.icmp<"slt">(%49, %50)
        |     %52 : () = llvm.cond_br<"loop_body3", "loop_exit3">(%51)
        |     %53 : () = llvm.label<"loop_body3">()
        |     %54 : index = 3
        |     %55 : () = llvm.mul(%41, %54)
        |     %56 : () = llvm.add(%55, %49)
        |     %57 : () = llvm.gep(%1, %56)
        |     %58 : () = llvm.load(%57)
        |     %59 : index = 2
        |     %60 : () = llvm.mul(%49, %59)
        |     %61 : () = llvm.add(%60, %41)
        |     %62 : () = llvm.gep(%37, %61)
        |     %63 : () = llvm.store(%58, %62)
        |     %64 : index = 1
        |     %65 : () = llvm.add(%49, %64)
        |     %66 : () = llvm.br<"loop_header3">()
        |     %67 : () = llvm.label<"loop_exit3">()
        |     %68 : index = 1
        |     %69 : () = llvm.add(%41, %68)
        |     %70 : () = llvm.br<"loop_header2">()
        |     %71 : () = llvm.label<"loop_exit2">()
        |     %72 : () = llvm.alloca<6>()
        |     %73 : index = 0
        |     %74 : () = llvm.br<"loop_header4">()
        |     %75 : () = llvm.label<"loop_header4">()
        |     %76 : () = llvm.phi<["loop_exit2", "loop_exit5"]>([%73, %110])
        |     %77 : index = 3
        |     %78 : () = llvm.icmp<"slt">(%76, %77)
        |     %79 : () = llvm.cond_br<"loop_body4", "loop_exit4">(%78)
        |     %80 : () = llvm.label<"loop_body4">()
        |     %81 : index = 0
        |     %82 : () = llvm.br<"loop_header5">()
        |     %83 : () = llvm.label<"loop_header5">()
        |     %84 : () = llvm.phi<["loop_body4", "loop_body5"]>([%81, %106])
        |     %85 : index = 2
        |     %86 : () = llvm.icmp<"slt">(%84, %85)
        |     %87 : () = llvm.cond_br<"loop_body5", "loop_exit5">(%86)
        |     %88 : () = llvm.label<"loop_body5">()
        |     %89 : index = 2
        |     %90 : () = llvm.mul(%76, %89)
        |     %91 : () = llvm.add(%90, %84)
        |     %92 : () = llvm.gep(%2, %91)
        |     %93 : () = llvm.load(%92)
        |     %94 : index = 2
        |     %95 : () = llvm.mul(%76, %94)
        |     %96 : () = llvm.add(%95, %84)
        |     %97 : () = llvm.gep(%37, %96)
        |     %98 : () = llvm.load(%97)
        |     %99 : () = llvm.fmul(%93, %98)
        |     %100 : index = 2
        |     %101 : () = llvm.mul(%76, %100)
        |     %102 : () = llvm.add(%101, %84)
        |     %103 : () = llvm.gep(%72, %102)
        |     %104 : () = llvm.store(%99, %103)
        |     %105 : index = 1
        |     %106 : () = llvm.add(%84, %105)
        |     %107 : () = llvm.br<"loop_header5">()
        |     %108 : () = llvm.label<"loop_exit5">()
        |     %109 : index = 1
        |     %110 : () = llvm.add(%76, %109)
        |     %111 : () = llvm.br<"loop_header4">()
        |     %112 : () = llvm.label<"loop_exit4">()
        |     %113 : index = 6
        |     %114 : () = llvm.call<"print_memref">([%72, %113])
        |     %115 : () = return(())
    """)
    assert result == expected
