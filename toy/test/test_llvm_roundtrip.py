"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_roundtrip_alloca():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.alloca(3)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_gep_load_store():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.alloca(6)
        |     %1 : index = 0
        |     %2 : () = llvm.gep(%0, %1)
        |     %3 : f64 = 1.0
        |     %_ : () = llvm.store(%3, %2)
        |     %4 : () = llvm.load(%2)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_fadd_fmul():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : f64 = 1.0
        |     %1 : f64 = 2.0
        |     %2 : () = llvm.fadd(%0, %1)
        |     %3 : () = llvm.fmul(%0, %1)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add_mul_int():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : index = 3
        |     %1 : index = 4
        |     %2 : () = llvm.add(%0, %1)
        |     %3 : () = llvm.mul(%0, %1)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_icmp_condbr():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : index = 0
        |     %1 : index = 10
        |     %cmp : () = llvm.icmp("slt", %0, %1)
        |     %_ : () = llvm.cond_br(%cmp, "loop_body", "loop_exit")
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_label_br():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %_ : () = llvm.br("loop_header")
        |     %_ : () = llvm.label("loop_header")
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_phi():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %i0 : () = llvm.phi([%init, %next], ["entry", "loop_body"])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_with_result():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.call("foo", [%a, %b])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_void():
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %_ : () = llvm.call("print_memref", [%ptr, %size])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 : f64 = 42.0
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_loop_pattern():
    """Full loop pattern: br, label, phi, icmp, condbr, body, increment."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.alloca(3)
        |     %init : index = 0
        |     %_ : () = llvm.br("loop_header0")
        |     %_ : () = llvm.label("loop_header0")
        |     %i0 : () = llvm.phi([%init, %next0], ["entry", "loop_body0"])
        |     %hi : index = 3
        |     %cmp : () = llvm.icmp("slt", %i0, %hi)
        |     %_ : () = llvm.cond_br(%cmp, "loop_body0", "loop_exit0")
        |     %_ : () = llvm.label("loop_body0")
        |     %val : f64 = 1.0
        |     %ptr : () = llvm.gep(%0, %i0)
        |     %_ : () = llvm.store(%val, %ptr)
        |     %one : index = 1
        |     %next0 : () = llvm.add(%i0, %one)
        |     %_ : () = llvm.br("loop_header0")
        |     %_ : () = llvm.label("loop_exit0")
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
