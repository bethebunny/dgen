"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_llvm_via_imports():
    """Parse a function using llvm ops via import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : F64 = 1.0
        |     %_ : Nil = llvm.store(%1, %0)
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %alloca : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %_ : Nil = llvm.br<"loop_header">([%init])
        |     %_ : Nil = llvm.label<"loop_header">() (%i: Index):
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i, %hi)
        |         %_ : Nil = llvm.cond_br<"loop_body", "loop_exit">(%cmp, [], [])
        |     %_ : Nil = llvm.label<"loop_body">() ():
        |         %one : Index = 1
        |         %next : Nil = llvm.add(%i, %one)
        |         %_ : Nil = llvm.br<"loop_header">([%next])
        |     %_ : Nil = llvm.label<"loop_exit">() ():
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
