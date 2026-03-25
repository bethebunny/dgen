"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_alloca():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %0 : Nil = llvm.alloca<3>()
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_gep_load_store():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : Index = 0
        |     %2 : Nil = llvm.gep(%0, %1)
        |     %3 : F64 = 1.0
        |     %4 : Nil = llvm.store(%3, %2)
        |     %5 : Nil = llvm.load(%2)
        |     %_ : Nil = chain(%5, %4)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_fadd_fmul():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %0 : F64 = 1.0
        |     %1 : F64 = 2.0
        |     %2 : Nil = llvm.fadd(%0, %1)
        |     %3 : Nil = llvm.fmul(%0, %1)
        |     %_ : Nil = chain(%2, %3)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_add_mul_int():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %0 : Index = 3
        |     %1 : Index = 4
        |     %2 : Nil = llvm.add(%0, %1)
        |     %3 : Nil = llvm.mul(%0, %1)
        |     %_ : Nil = chain(%2, %3)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_icmp_condbr():
    ir = strip_prefix("""
        | import goto
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %0 : Index = 0
        |     %1 : Index = 10
        |     %cmp : Nil = llvm.icmp<"slt">(%0, %1)
        |     %loop_body : goto.Label = goto.label() body():
        |         %_ : Nil = ()
        |     %loop_exit : goto.Label = goto.label() body():
        |         %_ : Nil = ()
        |     %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [], [])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_label_br():
    ir = strip_prefix("""
        | import goto
        |
        | %f : Nil = function<Nil>() body():
        |     %loop_header : goto.Label = goto.label() body():
        |     %_ : Nil = goto.branch<%loop_header>([])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_call_with_result():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body(%a: Index, %b: Index):
        |     %0 : Nil = llvm.call<"foo">([%a, %b])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_call_void():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() body(%ptr: Index, %size: Index):
        |     %0 : Nil = llvm.call<"print_memref">([%ptr, %size])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() body():
        |     %0 : F64 = 42.0
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_loop_pattern():
    """Full loop pattern: branch with args, label with block args, conditional_branch with args."""
    ir = strip_prefix("""
        | import goto
        | import llvm
        |
        | %f : Nil = function<Nil>() body():
        |     %alloc : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %loop_header : goto.Label = goto.label() body(%i: Index, %p: llvm.Ptr):
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i, %hi)
        |         %loop_body : goto.Label = goto.label() body(%j: Index, %q: llvm.Ptr):
        |             %one : Index = 1
        |             %next : Nil = llvm.add(%j, %one)
        |             %_ : Nil = goto.branch<%loop_header>([%next, %q])
        |         %loop_exit : goto.Label = goto.label() body():
        |             %_ : Nil = ()
        |         %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [%i, %p], [])
        |     %_ : Nil = goto.branch<%loop_header>([%init, %alloc])
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
