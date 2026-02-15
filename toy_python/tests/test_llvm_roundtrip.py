"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from toy_python.asm.parser import parse_module
from toy_python.dialects import llvm
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_roundtrip_alloca():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(3)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_gep_load_store():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(6)
        |     %1 = llvm.iconst(0)
        |     %2 = llvm.gep(%0, %1)
        |     %3 = llvm.fconst(1.0)
        |     llvm.store(%3, %2)
        |     %4 = llvm.load(%2)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_fadd_fmul():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.fconst(1.0)
        |     %1 = llvm.fconst(2.0)
        |     %2 = llvm.fadd(%0, %1)
        |     %3 = llvm.fmul(%0, %1)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add_mul_int():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.iconst(3)
        |     %1 = llvm.iconst(4)
        |     %2 = llvm.add(%0, %1)
        |     %3 = llvm.mul(%0, %1)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_icmp_condbr():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.iconst(0)
        |     %1 = llvm.iconst(10)
        |     %cmp = llvm.icmp(slt, %0, %1)
        |     llvm.cond_br(%cmp, loop_body, loop_exit)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_label_br():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     llvm.br(loop_header)
        |     llvm.label(loop_header)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_phi():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %i0 = llvm.phi([%init, %next], [entry, loop_body])
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_with_result():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.call(@foo, [%a, %b])
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_void():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     llvm.call(@print_memref, [%ptr, %size])
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.fconst(42.0)
        |     return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_loop_pattern():
    """Full loop pattern: br, label, phi, icmp, condbr, body, increment."""
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(3)
        |     %init = llvm.iconst(0)
        |     llvm.br(loop_header0)
        |     llvm.label(loop_header0)
        |     %i0 = llvm.phi([%init, %next0], [entry, loop_body0])
        |     %hi = llvm.iconst(3)
        |     %cmp = llvm.icmp(slt, %i0, %hi)
        |     llvm.cond_br(%cmp, loop_body0, loop_exit0)
        |     llvm.label(loop_body0)
        |     %val = llvm.fconst(1.0)
        |     %ptr = llvm.gep(%0, %i0)
        |     llvm.store(%val, %ptr)
        |     %one = llvm.iconst(1)
        |     %next0 = llvm.add(%i0, %one)
        |     llvm.br(loop_header0)
        |     llvm.label(loop_exit0)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
