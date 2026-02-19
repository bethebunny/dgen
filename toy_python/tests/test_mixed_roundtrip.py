"""Round-trip tests for mixed-dialect parsing using import headers."""

from toy_python.asm.parser import parse_module
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_llvm_via_imports():
    """Parse a function using llvm ops via import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(6)
        |     %1 = constant(1.0) : f64
        |     %_ = llvm.store(%1, %0)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 = llvm.alloca(3)
        |     %init = constant(0) : index
        |     %_ = llvm.br("loop_header")
        |     %_ = llvm.label("loop_header")
        |     %i0 = llvm.phi([%init, %next], ["entry", "loop_body"])
        |     %hi = constant(3) : index
        |     %cmp = llvm.icmp("slt", %i0, %hi)
        |     %_ = llvm.cond_br(%cmp, "loop_body", "loop_exit")
        |     %_ = llvm.label("loop_body")
        |     %one = constant(1) : index
        |     %next = llvm.add(%i0, %one)
        |     %_ = llvm.br("loop_header")
        |     %_ = llvm.label("loop_exit")
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
