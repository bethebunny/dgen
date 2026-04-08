"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import index as _index  # noqa: F401 — register index dialect
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_alloca():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<index.Index(3)>()
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_gep_load_store():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        | import memory
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<index.Index(6)>()
        |     %1 : index.Index = 0
        |     %2 : Nil = llvm.gep(%0, %1)
        |     %3 : number.Float64 = 1.0
        |     %4 : Nil = memory.store(%2, %3, %2)
        |     %5 : Nil = memory.load(%4, %2)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_fadd_fmul():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : number.Float64 = 1.0
        |     %1 : number.Float64 = 2.0
        |     %2 : Nil = llvm.fadd(%0, %1)
        |     %3 : Nil = llvm.fmul(%0, %1)
        |     %_ : Nil = chain(%2, %3)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_add_mul_int():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : index.Index = 3
        |     %1 : index.Index = 4
        |     %2 : Nil = llvm.add(%0, %1)
        |     %3 : Nil = llvm.mul(%0, %1)
        |     %_ : Nil = chain(%2, %3)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_icmp_condbr():
    ir = strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        | import index
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : index.Index = 0
        |     %1 : index.Index = 10
        |     %cmp : llvm.Int<index.Index(1)> = llvm.icmp<String("slt")>(%0, %1)
        |     %loop_body : goto.Label = goto.label([]) body():
        |         %_ : Nil = ()
        |     %loop_exit : goto.Label = goto.label([]) body():
        |         %_ : Nil = ()
        |     %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [], [])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_label_br():
    ir = strip_prefix("""
        | import function
        | import goto
        | import index
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %loop_header : goto.Label = goto.label([]) body():
        |         %_0 : Nil = ()
        |     %_ : Nil = goto.branch<%loop_header>([])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_call_with_result():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        |
        | %f : function.Function<[index.Index, index.Index], Nil> = function.function<Nil>() body(%a: index.Index, %b: index.Index):
        |     %0 : Nil = llvm.call<String("foo")>([%a, %b])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_call_void():
    ir = strip_prefix("""
        | import function
        | import index
        | import llvm
        |
        | %f : function.Function<[index.Index, index.Index], Nil> = function.function<Nil>() body(%ptr: index.Index, %size: index.Index):
        |     %0 : Nil = llvm.call<String("print_memref")>([%ptr, %size])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | import function
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : number.Float64 = 42.0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_loop_pattern():
    """Full loop pattern: branch with args, label with block args, conditional_branch with args."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        | import index
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %alloc : Nil = llvm.alloca<index.Index(3)>()
        |     %init : index.Index = 0
        |     %loop_header : goto.Label = goto.label([]) body<%self: goto.Label, %exit: goto.Label>(%i: index.Index, %p: llvm.Ptr):
        |         %hi : index.Index = 3
        |         %cmp : llvm.Int<index.Index(1)> = llvm.icmp<String("slt")>(%i, %hi)
        |         %loop_body : goto.Label = goto.label([]) body(%j: index.Index, %q: llvm.Ptr) captures(%self):
        |             %one : index.Index = 1
        |             %next : llvm.Int<index.Index(64)> = llvm.add(%j, %one)
        |             %_ : Nil = goto.branch<%self>([%next, %q])
        |         %loop_exit : goto.Label = goto.label([]) body():
        |             %_0 : Nil = ()
        |         %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [%i, %p], [])
        |     %_ : Nil = goto.branch<%loop_header>([%init, %alloc])
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
