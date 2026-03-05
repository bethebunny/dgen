"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_llvm_via_imports():
    """Parse a function using llvm ops via import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.alloca<6>()
        |     %1 : F64 = 1.0
        |     %_ : () = llvm.store(%1, %0)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f = function () -> ():
        |     %0 : () = llvm.alloca<3>()
        |     %init : Index = 0
        |     %_ : () = llvm.br<"loop_header">()
        |     %_ : () = llvm.label<"loop_header">()
        |     %i0 : () = llvm.phi<["entry", "loop_body"]>([%init, %next])
        |     %hi : Index = 3
        |     %cmp : () = llvm.icmp<"slt">(%i0, %hi)
        |     %_ : () = llvm.cond_br<"loop_body", "loop_exit">(%cmp)
        |     %_ : () = llvm.label<"loop_body">()
        |     %one : Index = 1
        |     %next : () = llvm.add(%i0, %one)
        |     %_ : () = llvm.br<"loop_header">()
        |     %_ : () = llvm.label<"loop_exit">()
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
