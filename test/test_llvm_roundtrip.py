"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_alloca():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<3>()
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_gep_load_store():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : Index = 0
        |     %2 : Nil = llvm.gep(%0, %1)
        |     %3 : F64 = 1.0
        |     %_ : Nil = llvm.store(%3, %2)
        |     %4 : Nil = llvm.load(%2)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_fadd_fmul():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : F64 = 1.0
        |     %1 : F64 = 2.0
        |     %2 : Nil = llvm.fadd(%0, %1)
        |     %3 : Nil = llvm.fmul(%0, %1)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_add_mul_int():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Index = 3
        |     %1 : Index = 4
        |     %2 : Nil = llvm.add(%0, %1)
        |     %3 : Nil = llvm.mul(%0, %1)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_icmp_condbr():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Index = 0
        |     %1 : Index = 10
        |     %cmp : Nil = llvm.icmp<"slt">(%0, %1)
        |     %loop_body : llvm.Label = llvm.label() ():
        |         %_ : Nil = return(())
        |     %loop_exit : llvm.Label = llvm.label() ():
        |         %_ : Nil = return(())
        |     %_ : Nil = llvm.cond_br(%cmp, %loop_body, %loop_exit, [], [])
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_label_br():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %loop_header : llvm.Label = llvm.label() ():
        |         %_ : Nil = return(())
        |     %_ : Nil = llvm.br(%loop_header, [])
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_call_with_result():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.call<"foo">([%a, %b])
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_call_void():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %_ : Nil = llvm.call<"print_memref">([%ptr, %size])
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() ():
        |     %0 : F64 = 42.0
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_loop_pattern():
    """Full loop pattern: br with args, label with block args, cond_br with args."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %loop_header0 : llvm.Label = llvm.label() (%i0: Index):
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        |         %loop_body0 : llvm.Label = llvm.label() (%j: Index):
        |             %val : F64 = 1.0
        |             %one : Index = 1
        |             %next0 : Nil = llvm.add(%j, %one)
        |             %_ : Nil = llvm.br(%loop_header0, [%next0])
        |         %loop_exit0 : llvm.Label = llvm.label() ():
        |             %_ : Nil = return(())
        |         %_ : Nil = llvm.cond_br(%cmp, %loop_body0, %loop_exit0, [%i0], [])
        |     %_ : Nil = llvm.br(%loop_header0, [%init])
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
