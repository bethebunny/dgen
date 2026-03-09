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
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[3], F64> = [1.0, 2.0, 3.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[3], F64> = [1.0, 2.0, 3.0]
        |     %1 : Index = 3
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(())
    """)
    assert result == expected


def test_constant_preserved():
    """Constants are preserved as tensor constants (not expanded to scalar stores)."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[3], F64> = [1.0, 2.0, 3.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[3], F64> = [1.0, 2.0, 3.0]
        |     %1 : Index = 3
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(())
    """)
    assert result == expected


def test_2d_constant_preserved():
    """2D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Index = 6
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(())
    """)
    assert result == expected


def test_load_store_linearization():
    """Load/store with multi-dim indices are linearized."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %_ = toy.print(%1)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.alloca<6>()
        |     %2 : Index = 0
        |     %3 : Nil = llvm.br<"loop_header0">()
        |     %4 : Nil = llvm.label<"loop_header0">()
        |     %5 : Nil = llvm.phi<"entry", "loop_exit1">(%2, %33)
        |     %6 : Index = 2
        |     %7 : llvm.Int<1> = llvm.icmp<"slt">(%5, %6)
        |     %8 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%7)
        |     %9 : Nil = llvm.label<"loop_body0">()
        |     %10 : Index = 0
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_body1">(%10, %29)
        |     %14 : Index = 3
        |     %15 : llvm.Int<1> = llvm.icmp<"slt">(%13, %14)
        |     %16 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%15)
        |     %17 : Nil = llvm.label<"loop_body1">()
        |     %18 : Index = 3
        |     %19 : llvm.Int<64> = llvm.mul(%5, %18)
        |     %20 : llvm.Int<64> = llvm.add(%19, %13)
        |     %21 : llvm.Ptr = llvm.gep(%0, %20)
        |     %22 : llvm.Float = llvm.load(%21)
        |     %23 : Index = 2
        |     %24 : llvm.Int<64> = llvm.mul(%13, %23)
        |     %25 : llvm.Int<64> = llvm.add(%24, %5)
        |     %26 : llvm.Ptr = llvm.gep(%1, %25)
        |     %27 : Nil = llvm.store(%22, %26)
        |     %28 : Index = 1
        |     %29 : llvm.Int<64> = llvm.add(%13, %28)
        |     %30 : Nil = llvm.br<"loop_header1">()
        |     %31 : Nil = llvm.label<"loop_exit1">()
        |     %32 : Index = 1
        |     %33 : llvm.Int<64> = llvm.add(%5, %32)
        |     %34 : Nil = llvm.br<"loop_header0">()
        |     %35 : Nil = llvm.label<"loop_exit0">()
        |     %36 : Index = 6
        |     %37 : Nil = llvm.call<"print_memref">([%1, %36])
        |     %38 : Nil = return(())
    """)
    assert result == expected


def test_3d_constant_preserved():
    """3D constants are preserved as tensor constants."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %_ = toy.print(%0)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : Index = 8
        |     %2 : Nil = llvm.call<"print_memref">([%0, %1])
        |     %3 : Nil = return(())
    """)
    assert result == expected


def test_3d_load_store_linearization():
    """3D load/store indices are linearized with stride multiplication."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], F64> = toy.add(%0, %1)
        |     %_ = toy.print(%2)
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 2, 2], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], F64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : llvm.Ptr = llvm.alloca<8>()
        |     %3 : Index = 0
        |     %4 : Nil = llvm.br<"loop_header0">()
        |     %5 : Nil = llvm.label<"loop_header0">()
        |     %6 : Nil = llvm.phi<"entry", "loop_exit1">(%3, %61)
        |     %7 : Index = 2
        |     %8 : llvm.Int<1> = llvm.icmp<"slt">(%6, %7)
        |     %9 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%8)
        |     %10 : Nil = llvm.label<"loop_body0">()
        |     %11 : Index = 0
        |     %12 : Nil = llvm.br<"loop_header1">()
        |     %13 : Nil = llvm.label<"loop_header1">()
        |     %14 : Nil = llvm.phi<"loop_body0", "loop_exit2">(%11, %57)
        |     %15 : Index = 2
        |     %16 : llvm.Int<1> = llvm.icmp<"slt">(%14, %15)
        |     %17 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%16)
        |     %18 : Nil = llvm.label<"loop_body1">()
        |     %19 : Index = 0
        |     %20 : Nil = llvm.br<"loop_header2">()
        |     %21 : Nil = llvm.label<"loop_header2">()
        |     %22 : Nil = llvm.phi<"loop_body1", "loop_body2">(%19, %53)
        |     %23 : Index = 2
        |     %24 : llvm.Int<1> = llvm.icmp<"slt">(%22, %23)
        |     %25 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%24)
        |     %26 : Nil = llvm.label<"loop_body2">()
        |     %27 : Index = 4
        |     %28 : llvm.Int<64> = llvm.mul(%6, %27)
        |     %29 : Index = 2
        |     %30 : llvm.Int<64> = llvm.mul(%14, %29)
        |     %31 : llvm.Int<64> = llvm.add(%28, %30)
        |     %32 : llvm.Int<64> = llvm.add(%31, %22)
        |     %33 : llvm.Ptr = llvm.gep(%0, %32)
        |     %34 : llvm.Float = llvm.load(%33)
        |     %35 : Index = 4
        |     %36 : llvm.Int<64> = llvm.mul(%6, %35)
        |     %37 : Index = 2
        |     %38 : llvm.Int<64> = llvm.mul(%14, %37)
        |     %39 : llvm.Int<64> = llvm.add(%36, %38)
        |     %40 : llvm.Int<64> = llvm.add(%39, %22)
        |     %41 : llvm.Ptr = llvm.gep(%1, %40)
        |     %42 : llvm.Float = llvm.load(%41)
        |     %43 : llvm.Float = llvm.fadd(%34, %42)
        |     %44 : Index = 4
        |     %45 : llvm.Int<64> = llvm.mul(%6, %44)
        |     %46 : Index = 2
        |     %47 : llvm.Int<64> = llvm.mul(%14, %46)
        |     %48 : llvm.Int<64> = llvm.add(%45, %47)
        |     %49 : llvm.Int<64> = llvm.add(%48, %22)
        |     %50 : llvm.Ptr = llvm.gep(%2, %49)
        |     %51 : Nil = llvm.store(%43, %50)
        |     %52 : Index = 1
        |     %53 : llvm.Int<64> = llvm.add(%22, %52)
        |     %54 : Nil = llvm.br<"loop_header2">()
        |     %55 : Nil = llvm.label<"loop_exit2">()
        |     %56 : Index = 1
        |     %57 : llvm.Int<64> = llvm.add(%14, %56)
        |     %58 : Nil = llvm.br<"loop_header1">()
        |     %59 : Nil = llvm.label<"loop_exit1">()
        |     %60 : Index = 1
        |     %61 : llvm.Int<64> = llvm.add(%6, %60)
        |     %62 : Nil = llvm.br<"loop_header0">()
        |     %63 : Nil = llvm.label<"loop_exit0">()
        |     %64 : Index = 8
        |     %65 : Nil = llvm.call<"print_memref">([%2, %64])
        |     %66 : Nil = return(())
    """)
    assert result == expected


def test_full_example():
    """Full pipeline: constant + transpose + mul + print -> LLVM IR."""
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
        |     %_ = return(())
    """)
    result = compile_to_llvm(ir_text)
    expected = strip_prefix("""
        | import llvm
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : llvm.Ptr = llvm.alloca<6>()
        |     %2 : Index = 0
        |     %3 : Nil = llvm.br<"loop_header0">()
        |     %4 : Nil = llvm.label<"loop_header0">()
        |     %5 : Nil = llvm.phi<"entry", "loop_exit1">(%2, %33)
        |     %6 : Index = 2
        |     %7 : llvm.Int<1> = llvm.icmp<"slt">(%5, %6)
        |     %8 : Nil = llvm.cond_br<"loop_body0", "loop_exit0">(%7)
        |     %9 : Nil = llvm.label<"loop_body0">()
        |     %10 : Index = 0
        |     %11 : Nil = llvm.br<"loop_header1">()
        |     %12 : Nil = llvm.label<"loop_header1">()
        |     %13 : Nil = llvm.phi<"loop_body0", "loop_body1">(%10, %29)
        |     %14 : Index = 3
        |     %15 : llvm.Int<1> = llvm.icmp<"slt">(%13, %14)
        |     %16 : Nil = llvm.cond_br<"loop_body1", "loop_exit1">(%15)
        |     %17 : Nil = llvm.label<"loop_body1">()
        |     %18 : Index = 3
        |     %19 : llvm.Int<64> = llvm.mul(%5, %18)
        |     %20 : llvm.Int<64> = llvm.add(%19, %13)
        |     %21 : llvm.Ptr = llvm.gep(%0, %20)
        |     %22 : llvm.Float = llvm.load(%21)
        |     %23 : Index = 2
        |     %24 : llvm.Int<64> = llvm.mul(%13, %23)
        |     %25 : llvm.Int<64> = llvm.add(%24, %5)
        |     %26 : llvm.Ptr = llvm.gep(%1, %25)
        |     %27 : Nil = llvm.store(%22, %26)
        |     %28 : Index = 1
        |     %29 : llvm.Int<64> = llvm.add(%13, %28)
        |     %30 : Nil = llvm.br<"loop_header1">()
        |     %31 : Nil = llvm.label<"loop_exit1">()
        |     %32 : Index = 1
        |     %33 : llvm.Int<64> = llvm.add(%5, %32)
        |     %34 : Nil = llvm.br<"loop_header0">()
        |     %35 : Nil = llvm.label<"loop_exit0">()
        |     %36 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %37 : llvm.Ptr = llvm.alloca<6>()
        |     %38 : Index = 0
        |     %39 : Nil = llvm.br<"loop_header2">()
        |     %40 : Nil = llvm.label<"loop_header2">()
        |     %41 : Nil = llvm.phi<"loop_exit0", "loop_exit3">(%38, %69)
        |     %42 : Index = 2
        |     %43 : llvm.Int<1> = llvm.icmp<"slt">(%41, %42)
        |     %44 : Nil = llvm.cond_br<"loop_body2", "loop_exit2">(%43)
        |     %45 : Nil = llvm.label<"loop_body2">()
        |     %46 : Index = 0
        |     %47 : Nil = llvm.br<"loop_header3">()
        |     %48 : Nil = llvm.label<"loop_header3">()
        |     %49 : Nil = llvm.phi<"loop_body2", "loop_body3">(%46, %65)
        |     %50 : Index = 3
        |     %51 : llvm.Int<1> = llvm.icmp<"slt">(%49, %50)
        |     %52 : Nil = llvm.cond_br<"loop_body3", "loop_exit3">(%51)
        |     %53 : Nil = llvm.label<"loop_body3">()
        |     %54 : Index = 3
        |     %55 : llvm.Int<64> = llvm.mul(%41, %54)
        |     %56 : llvm.Int<64> = llvm.add(%55, %49)
        |     %57 : llvm.Ptr = llvm.gep(%36, %56)
        |     %58 : llvm.Float = llvm.load(%57)
        |     %59 : Index = 2
        |     %60 : llvm.Int<64> = llvm.mul(%49, %59)
        |     %61 : llvm.Int<64> = llvm.add(%60, %41)
        |     %62 : llvm.Ptr = llvm.gep(%37, %61)
        |     %63 : Nil = llvm.store(%58, %62)
        |     %64 : Index = 1
        |     %65 : llvm.Int<64> = llvm.add(%49, %64)
        |     %66 : Nil = llvm.br<"loop_header3">()
        |     %67 : Nil = llvm.label<"loop_exit3">()
        |     %68 : Index = 1
        |     %69 : llvm.Int<64> = llvm.add(%41, %68)
        |     %70 : Nil = llvm.br<"loop_header2">()
        |     %71 : Nil = llvm.label<"loop_exit2">()
        |     %72 : llvm.Ptr = llvm.alloca<6>()
        |     %73 : Index = 0
        |     %74 : Nil = llvm.br<"loop_header4">()
        |     %75 : Nil = llvm.label<"loop_header4">()
        |     %76 : Nil = llvm.phi<"loop_exit2", "loop_exit5">(%73, %110)
        |     %77 : Index = 3
        |     %78 : llvm.Int<1> = llvm.icmp<"slt">(%76, %77)
        |     %79 : Nil = llvm.cond_br<"loop_body4", "loop_exit4">(%78)
        |     %80 : Nil = llvm.label<"loop_body4">()
        |     %81 : Index = 0
        |     %82 : Nil = llvm.br<"loop_header5">()
        |     %83 : Nil = llvm.label<"loop_header5">()
        |     %84 : Nil = llvm.phi<"loop_body4", "loop_body5">(%81, %106)
        |     %85 : Index = 2
        |     %86 : llvm.Int<1> = llvm.icmp<"slt">(%84, %85)
        |     %87 : Nil = llvm.cond_br<"loop_body5", "loop_exit5">(%86)
        |     %88 : Nil = llvm.label<"loop_body5">()
        |     %89 : Index = 2
        |     %90 : llvm.Int<64> = llvm.mul(%76, %89)
        |     %91 : llvm.Int<64> = llvm.add(%90, %84)
        |     %92 : llvm.Ptr = llvm.gep(%1, %91)
        |     %93 : llvm.Float = llvm.load(%92)
        |     %94 : Index = 2
        |     %95 : llvm.Int<64> = llvm.mul(%76, %94)
        |     %96 : llvm.Int<64> = llvm.add(%95, %84)
        |     %97 : llvm.Ptr = llvm.gep(%37, %96)
        |     %98 : llvm.Float = llvm.load(%97)
        |     %99 : llvm.Float = llvm.fmul(%93, %98)
        |     %100 : Index = 2
        |     %101 : llvm.Int<64> = llvm.mul(%76, %100)
        |     %102 : llvm.Int<64> = llvm.add(%101, %84)
        |     %103 : llvm.Ptr = llvm.gep(%72, %102)
        |     %104 : Nil = llvm.store(%99, %103)
        |     %105 : Index = 1
        |     %106 : llvm.Int<64> = llvm.add(%84, %105)
        |     %107 : Nil = llvm.br<"loop_header5">()
        |     %108 : Nil = llvm.label<"loop_exit5">()
        |     %109 : Index = 1
        |     %110 : llvm.Int<64> = llvm.add(%76, %109)
        |     %111 : Nil = llvm.br<"loop_header4">()
        |     %112 : Nil = llvm.label<"loop_exit4">()
        |     %113 : Index = 6
        |     %114 : Nil = llvm.call<"print_memref">([%72, %113])
        |     %115 : Nil = return(())
    """)
    assert result == expected
