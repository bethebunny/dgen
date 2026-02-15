"""Round-trip tests for mixed-dialect parsing using import headers."""

from toy_python.asm.parser import parse_module
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_llvm_via_imports():
    """Parse a function using llvm ops via import headers."""
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(6)
        |     %1 = llvm.fconst(1.0)
        |     llvm.store(%1, %0)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers."""
    ir = strip_prefix("""
        | from builtin import function, return
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(3)
        |     %init = llvm.iconst(0)
        |     llvm.br(loop_header)
        |     llvm.label(loop_header)
        |     %i0 = llvm.phi([%init, %next], [entry, loop_body])
        |     %hi = llvm.iconst(3)
        |     %cmp = llvm.icmp(slt, %i0, %hi)
        |     llvm.cond_br(%cmp, loop_body, loop_exit)
        |     llvm.label(loop_body)
        |     %one = llvm.iconst(1)
        |     %next = llvm.add(%i0, %one)
        |     llvm.br(loop_header)
        |     llvm.label(loop_exit)
        |     return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
