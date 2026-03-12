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
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : llvm.Ptr = llvm.alloca<6>()
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %34)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Index = 0
        |     %12 : Nil = llvm.br<"loop_header1">()
        |     %13 : Nil = llvm.label<"loop_header1">()
        |     %14 : Nil = llvm.phi<"loop_body0", "loop_body1">(%11, %30)
        |     %15 : Index = 3
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%14, %15)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 3
        |     %20 : llvm.Int<64> = llvm.mul(%6, %19)
        |     %21 : llvm.Int<64> = llvm.add(%20, %14)
        |     %22 : llvm.Ptr = llvm.gep(%1, %21)
        |     %23 : llvm.Float = llvm.load(%22)
        |     %24 : Index = 2
        |     %25 : llvm.Int<64> = llvm.mul(%14, %24)
        |     %26 : llvm.Int<64> = llvm.add(%25, %6)
        |     %27 : llvm.Ptr = llvm.gep(%2, %26)
        |     %28 : Nil = llvm.store(%23, %27)
        |     %29 : Index = 1
        |     %30 : llvm.Int<64> = llvm.add(%14, %29)
        |     %31 : Nil = llvm.br<"loop_header1">()
        |     %32 : Nil = llvm.label<"loop_exit1">()
        |     %33 : Index = 1
        |     %34 : llvm.Int<64> = llvm.add(%6, %33)
        |     %35 : Nil = llvm.br<"loop_header0">()
        |     %36 : Nil = llvm.label<"loop_exit0">()
        |     %37 : Index = 6
        |     %38 : Nil = llvm.call<"print_memref">([%2, %37])
        |     %39 : Nil = return(%38)
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
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %3 : llvm.Ptr = llvm.load(%2)
        |     %4 : llvm.Ptr = llvm.alloca<4>()
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %42)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Index = 0
        |     %14 : Nil = llvm.br<"loop_header1">()
        |     %15 : Nil = llvm.label<"loop_header1">()
        |     %16 : Nil = llvm.phi<"loop_body0", "loop_body1">(%13, %38)
        |     %17 : Index = 2
        |     %18 : llvm.Int<1> = llvm.icmp<"slt">(%16, %17)
        |     %19 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%18)
        |     %20 : Nil = llvm.label<"loop_body1">()
        |     %21 : Index = 2
        |     %22 : llvm.Int<64> = llvm.mul(%8, %21)
        |     %23 : llvm.Int<64> = llvm.add(%22, %16)
        |     %24 : llvm.Ptr = llvm.gep(%1, %23)
        |     %25 : llvm.Float = llvm.load(%24)
        |     %26 : Index = 2
        |     %27 : llvm.Int<64> = llvm.mul(%8, %26)
        |     %28 : llvm.Int<64> = llvm.add(%27, %16)
        |     %29 : llvm.Ptr = llvm.gep(%3, %28)
        |     %30 : llvm.Float = llvm.load(%29)
        |     %31 : llvm.Float = llvm.fmul(%25, %30)
        |     %32 : Index = 2
        |     %33 : llvm.Int<64> = llvm.mul(%8, %32)
        |     %34 : llvm.Int<64> = llvm.add(%33, %16)
        |     %35 : llvm.Ptr = llvm.gep(%4, %34)
        |     %36 : Nil = llvm.store(%31, %35)
        |     %37 : Index = 1
        |     %38 : llvm.Int<64> = llvm.add(%16, %37)
        |     %39 : Nil = llvm.br<"loop_header1">()
        |     %40 : Nil = llvm.label<"loop_exit1">()
        |     %41 : Index = 1
        |     %42 : llvm.Int<64> = llvm.add(%8, %41)
        |     %43 : Nil = llvm.br<"loop_header0">()
        |     %44 : Nil = llvm.label<"loop_exit0">()
        |     %45 : Index = 4
        |     %46 : Nil = llvm.call<"print_memref">([%4, %45])
        |     %47 : Nil = return(%46)
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
        |     %0 : toy.Tensor<[2, 2], F64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : toy.Tensor<[2, 2], F64> = [5.0, 6.0, 7.0, 8.0]
        |     %3 : llvm.Ptr = llvm.load(%2)
        |     %4 : llvm.Ptr = llvm.alloca<4>()
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %42)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Index = 0
        |     %14 : Nil = llvm.br<"loop_header1">()
        |     %15 : Nil = llvm.label<"loop_header1">()
        |     %16 : Nil = llvm.phi<"loop_body0", "loop_body1">(%13, %38)
        |     %17 : Index = 2
        |     %18 : llvm.Int<1> = llvm.icmp<"slt">(%16, %17)
        |     %19 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%18)
        |     %20 : Nil = llvm.label<"loop_body1">()
        |     %21 : Index = 2
        |     %22 : llvm.Int<64> = llvm.mul(%8, %21)
        |     %23 : llvm.Int<64> = llvm.add(%22, %16)
        |     %24 : llvm.Ptr = llvm.gep(%1, %23)
        |     %25 : llvm.Float = llvm.load(%24)
        |     %26 : Index = 2
        |     %27 : llvm.Int<64> = llvm.mul(%8, %26)
        |     %28 : llvm.Int<64> = llvm.add(%27, %16)
        |     %29 : llvm.Ptr = llvm.gep(%3, %28)
        |     %30 : llvm.Float = llvm.load(%29)
        |     %31 : llvm.Float = llvm.fadd(%25, %30)
        |     %32 : Index = 2
        |     %33 : llvm.Int<64> = llvm.mul(%8, %32)
        |     %34 : llvm.Int<64> = llvm.add(%33, %16)
        |     %35 : llvm.Ptr = llvm.gep(%4, %34)
        |     %36 : Nil = llvm.store(%31, %35)
        |     %37 : Index = 1
        |     %38 : llvm.Int<64> = llvm.add(%16, %37)
        |     %39 : Nil = llvm.br<"loop_header1">()
        |     %40 : Nil = llvm.label<"loop_exit1">()
        |     %41 : Index = 1
        |     %42 : llvm.Int<64> = llvm.add(%8, %41)
        |     %43 : Nil = llvm.br<"loop_header0">()
        |     %44 : Nil = llvm.label<"loop_exit0">()
        |     %45 : Index = 4
        |     %46 : Nil = llvm.call<"print_memref">([%4, %45])
        |     %47 : Nil = return(%46)
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
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %3 : llvm.Ptr = llvm.load(%2)
        |     %4 : llvm.Ptr = llvm.alloca<8>()
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %63)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Index = 0
        |     %14 : Nil = llvm.br<"loop_header1">()
        |     %15 : Nil = llvm.label<"loop_header1">()
        |     %16 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%13, %59)
        |     %17 : Index = 2
        |     %18 : llvm.Int<1> = llvm.icmp<"slt">(%16, %17)
        |     %19 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%18)
        |     %20 : Nil = llvm.label<"loop_body1">()
        |     %21 : Index = 0
        |     %22 : Nil = llvm.br<"loop_header2">()
        |     %23 : Nil = llvm.label<"loop_header2">()
        |     %24 : Nil = llvm.phi<"loop_body1", "loop_body2">(%21, %55)
        |     %25 : Index = 2
        |     %26 : llvm.Int<1> = llvm.icmp<"slt">(%24, %25)
        |     %27 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%26)
        |     %28 : Nil = llvm.label<"loop_body2">()
        |     %29 : Index = 4
        |     %30 : llvm.Int<64> = llvm.mul(%8, %29)
        |     %31 : Index = 2
        |     %32 : llvm.Int<64> = llvm.mul(%16, %31)
        |     %33 : llvm.Int<64> = llvm.add(%30, %32)
        |     %34 : llvm.Int<64> = llvm.add(%33, %24)
        |     %35 : llvm.Ptr = llvm.gep(%1, %34)
        |     %36 : llvm.Float = llvm.load(%35)
        |     %37 : Index = 4
        |     %38 : llvm.Int<64> = llvm.mul(%8, %37)
        |     %39 : Index = 2
        |     %40 : llvm.Int<64> = llvm.mul(%16, %39)
        |     %41 : llvm.Int<64> = llvm.add(%38, %40)
        |     %42 : llvm.Int<64> = llvm.add(%41, %24)
        |     %43 : llvm.Ptr = llvm.gep(%3, %42)
        |     %44 : llvm.Float = llvm.load(%43)
        |     %45 : llvm.Float = llvm.fadd(%36, %44)
        |     %46 : Index = 4
        |     %47 : llvm.Int<64> = llvm.mul(%8, %46)
        |     %48 : Index = 2
        |     %49 : llvm.Int<64> = llvm.mul(%16, %48)
        |     %50 : llvm.Int<64> = llvm.add(%47, %49)
        |     %51 : llvm.Int<64> = llvm.add(%50, %24)
        |     %52 : llvm.Ptr = llvm.gep(%4, %51)
        |     %53 : Nil = llvm.store(%45, %52)
        |     %54 : Index = 1
        |     %55 : llvm.Int<64> = llvm.add(%24, %54)
        |     %56 : Nil = llvm.br<"loop_header2">()
        |     %57 : Nil = llvm.label<"loop_exit2">()
        |     %58 : Index = 1
        |     %59 : llvm.Int<64> = llvm.add(%16, %58)
        |     %60 : Nil = llvm.br<"loop_header1">()
        |     %61 : Nil = llvm.label<"loop_exit1">()
        |     %62 : Index = 1
        |     %63 : llvm.Int<64> = llvm.add(%8, %62)
        |     %64 : Nil = llvm.br<"loop_header0">()
        |     %65 : Nil = llvm.label<"loop_exit0">()
        |     %66 : Index = 8
        |     %67 : Nil = llvm.call<"print_memref">([%4, %66])
        |     %68 : Nil = return(%67)
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
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %3 : llvm.Ptr = llvm.load(%2)
        |     %4 : llvm.Ptr = llvm.alloca<8>()
        |     %5 : Index = 0
        |     %6 : Nil = llvm.br<"loop_header0">()
        |     %7 : Nil = llvm.label<"loop_header0">()
        |     %8 : Nil = llvm.phi<"entry", "loop_exit1">(%5, %63)
        |     %9 : Index = 2
        |     %10 : llvm.Int<1> = llvm.icmp<"slt">(%8, %9)
        |     %11 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%10)
        |     %12 : Nil = llvm.label<"loop_body0">()
        |     %13 : Index = 0
        |     %14 : Nil = llvm.br<"loop_header1">()
        |     %15 : Nil = llvm.label<"loop_header1">()
        |     %16 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%13, %59)
        |     %17 : Index = 2
        |     %18 : llvm.Int<1> = llvm.icmp<"slt">(%16, %17)
        |     %19 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%18)
        |     %20 : Nil = llvm.label<"loop_body1">()
        |     %21 : Index = 0
        |     %22 : Nil = llvm.br<"loop_header2">()
        |     %23 : Nil = llvm.label<"loop_header2">()
        |     %24 : Nil = llvm.phi<"loop_body1", "loop_body2">(%21, %55)
        |     %25 : Index = 2
        |     %26 : llvm.Int<1> = llvm.icmp<"slt">(%24, %25)
        |     %27 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%26)
        |     %28 : Nil = llvm.label<"loop_body2">()
        |     %29 : Index = 4
        |     %30 : llvm.Int<64> = llvm.mul(%8, %29)
        |     %31 : Index = 2
        |     %32 : llvm.Int<64> = llvm.mul(%16, %31)
        |     %33 : llvm.Int<64> = llvm.add(%30, %32)
        |     %34 : llvm.Int<64> = llvm.add(%33, %24)
        |     %35 : llvm.Ptr = llvm.gep(%1, %34)
        |     %36 : llvm.Float = llvm.load(%35)
        |     %37 : Index = 4
        |     %38 : llvm.Int<64> = llvm.mul(%8, %37)
        |     %39 : Index = 2
        |     %40 : llvm.Int<64> = llvm.mul(%16, %39)
        |     %41 : llvm.Int<64> = llvm.add(%38, %40)
        |     %42 : llvm.Int<64> = llvm.add(%41, %24)
        |     %43 : llvm.Ptr = llvm.gep(%3, %42)
        |     %44 : llvm.Float = llvm.load(%43)
        |     %45 : llvm.Float = llvm.fmul(%36, %44)
        |     %46 : Index = 4
        |     %47 : llvm.Int<64> = llvm.mul(%8, %46)
        |     %48 : Index = 2
        |     %49 : llvm.Int<64> = llvm.mul(%16, %48)
        |     %50 : llvm.Int<64> = llvm.add(%47, %49)
        |     %51 : llvm.Int<64> = llvm.add(%50, %24)
        |     %52 : llvm.Ptr = llvm.gep(%4, %51)
        |     %53 : Nil = llvm.store(%45, %52)
        |     %54 : Index = 1
        |     %55 : llvm.Int<64> = llvm.add(%24, %54)
        |     %56 : Nil = llvm.br<"loop_header2">()
        |     %57 : Nil = llvm.label<"loop_exit2">()
        |     %58 : Index = 1
        |     %59 : llvm.Int<64> = llvm.add(%16, %58)
        |     %60 : Nil = llvm.br<"loop_header1">()
        |     %61 : Nil = llvm.label<"loop_exit1">()
        |     %62 : Index = 1
        |     %63 : llvm.Int<64> = llvm.add(%8, %62)
        |     %64 : Nil = llvm.br<"loop_header0">()
        |     %65 : Nil = llvm.label<"loop_exit0">()
        |     %66 : Index = 8
        |     %67 : Nil = llvm.call<"print_memref">([%4, %66])
        |     %68 : Nil = return(%67)
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
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.load(%0)
        |     %2 : llvm.Ptr = llvm.alloca<6>()
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %34)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Index = 0
        |     %12 : Nil = llvm.br<"loop_header1">()
        |     %13 : Nil = llvm.label<"loop_header1">()
        |     %14 : Nil = llvm.phi<"loop_body0", "loop_body1">(%11, %30)
        |     %15 : Index = 3
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%14, %15)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 3
        |     %20 : llvm.Int<64> = llvm.mul(%6, %19)
        |     %21 : llvm.Int<64> = llvm.add(%20, %14)
        |     %22 : llvm.Ptr = llvm.gep(%1, %21)
        |     %23 : llvm.Float = llvm.load(%22)
        |     %24 : Index = 2
        |     %25 : llvm.Int<64> = llvm.mul(%14, %24)
        |     %26 : llvm.Int<64> = llvm.add(%25, %6)
        |     %27 : llvm.Ptr = llvm.gep(%2, %26)
        |     %28 : Nil = llvm.store(%23, %27)
        |     %29 : Index = 1
        |     %30 : llvm.Int<64> = llvm.add(%14, %29)
        |     %31 : Nil = llvm.br<"loop_header1">()
        |     %32 : Nil = llvm.label<"loop_exit1">()
        |     %33 : Index = 1
        |     %34 : llvm.Int<64> = llvm.add(%6, %33)
        |     %35 : Nil = llvm.br<"loop_header0">()
        |     %36 : Nil = llvm.label<"loop_exit0">()
        |     %37 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %38 : llvm.Ptr = llvm.load(%37)
        |     %39 : llvm.Ptr = llvm.alloca<6>()
        |     %40 : Index = 0
        |     %41 : Nil = llvm.br<"loop_header2">()
        |     %42 : Nil = llvm.label<"loop_header2">()
        |     %43 : Nil = llvm.phi<"loop_exit0", "loop_exit3">(%40, %71)
        |     %44 : Index = 2
        |     %45 : llvm.Int<1> = llvm.icmp<"slt">(%43, %44)
        |     %46 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%45)
        |     %47 : Nil = llvm.label<"loop_body2">()
        |     %48 : Index = 0
        |     %49 : Nil = llvm.br<"loop_header3">()
        |     %50 : Nil = llvm.label<"loop_header3">()
        |     %51 : Nil = llvm.phi<"loop_body2", "loop_body3">(%48, %67)
        |     %52 : Index = 3
        |     %53 : llvm.Int<1> = llvm.icmp<"slt">(%51, %52)
        |     %54 : Nil = llvm.cond_br<"loop_body3", "loop_exit3">(%53)
        |     %55 : Nil = llvm.label<"loop_body3">()
        |     %56 : Index = 3
        |     %57 : llvm.Int<64> = llvm.mul(%43, %56)
        |     %58 : llvm.Int<64> = llvm.add(%57, %51)
        |     %59 : llvm.Ptr = llvm.gep(%38, %58)
        |     %60 : llvm.Float = llvm.load(%59)
        |     %61 : Index = 2
        |     %62 : llvm.Int<64> = llvm.mul(%51, %61)
        |     %63 : llvm.Int<64> = llvm.add(%62, %43)
        |     %64 : llvm.Ptr = llvm.gep(%39, %63)
        |     %65 : Nil = llvm.store(%60, %64)
        |     %66 : Index = 1
        |     %67 : llvm.Int<64> = llvm.add(%51, %66)
        |     %68 : Nil = llvm.br<"loop_header3">()
        |     %69 : Nil = llvm.label<"loop_exit3">()
        |     %70 : Index = 1
        |     %71 : llvm.Int<64> = llvm.add(%43, %70)
        |     %72 : Nil = llvm.br<"loop_header2">()
        |     %73 : Nil = llvm.label<"loop_exit2">()
        |     %74 : llvm.Ptr = llvm.alloca<6>()
        |     %75 : Index = 0
        |     %76 : Nil = llvm.br<"loop_header4">()
        |     %77 : Nil = llvm.label<"loop_header4">()
        |     %78 : Nil = llvm.phi<"loop_exit2", "loop_exit5">(%75, %112)
        |     %79 : Index = 3
        |     %80 : llvm.Int<1> = llvm.icmp<"slt">(%78, %79)
        |     %81 : Nil = llvm.cond_br<"loop_body4", "loop_exit4">(%80)
        |     %82 : Nil = llvm.label<"loop_body4">()
        |     %83 : Index = 0
        |     %84 : Nil = llvm.br<"loop_header5">()
        |     %85 : Nil = llvm.label<"loop_header5">()
        |     %86 : Nil = llvm.phi<"loop_body4", "loop_body5">(%83, %108)
        |     %87 : Index = 2
        |     %88 : llvm.Int<1> = llvm.icmp<"slt">(%86, %87)
        |     %89 : Nil = llvm.cond_br<"loop_body5", "loop_exit5">(%88)
        |     %90 : Nil = llvm.label<"loop_body5">()
        |     %91 : Index = 2
        |     %92 : llvm.Int<64> = llvm.mul(%78, %91)
        |     %93 : llvm.Int<64> = llvm.add(%92, %86)
        |     %94 : llvm.Ptr = llvm.gep(%2, %93)
        |     %95 : llvm.Float = llvm.load(%94)
        |     %96 : Index = 2
        |     %97 : llvm.Int<64> = llvm.mul(%78, %96)
        |     %98 : llvm.Int<64> = llvm.add(%97, %86)
        |     %99 : llvm.Ptr = llvm.gep(%39, %98)
        |     %100 : llvm.Float = llvm.load(%99)
        |     %101 : llvm.Float = llvm.fmul(%95, %100)
        |     %102 : Index = 2
        |     %103 : llvm.Int<64> = llvm.mul(%78, %102)
        |     %104 : llvm.Int<64> = llvm.add(%103, %86)
        |     %105 : llvm.Ptr = llvm.gep(%74, %104)
        |     %106 : Nil = llvm.store(%101, %105)
        |     %107 : Index = 1
        |     %108 : llvm.Int<64> = llvm.add(%86, %107)
        |     %109 : Nil = llvm.br<"loop_header5">()
        |     %110 : Nil = llvm.label<"loop_exit5">()
        |     %111 : Index = 1
        |     %112 : llvm.Int<64> = llvm.add(%78, %111)
        |     %113 : Nil = llvm.br<"loop_header4">()
        |     %114 : Nil = llvm.label<"loop_exit4">()
        |     %115 : Index = 6
        |     %116 : Nil = llvm.call<"print_memref">([%74, %115])
        |     %117 : Nil = return(%116)
    """)
    assert result == expected
