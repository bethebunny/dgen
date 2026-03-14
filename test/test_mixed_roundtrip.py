"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent, strip_prefix


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
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %_ : Nil = llvm.br(%loop_header)
        |     %loop_header : llvm.Label = llvm.label() ():
        |         %i0 : Nil = llvm.phi<%entry, %loop_body>(%init, %next)
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        |         %_ : Nil = llvm.cond_br(%cmp, %loop_body, %loop_exit)
        |     %loop_body : llvm.Label = llvm.label() ():
        |         %one : Index = 1
        |         %next : Nil = llvm.add(%i0, %one)
        |         %_ : Nil = llvm.br(%loop_header)
        |     %loop_exit : llvm.Label = llvm.label() ():
        |         %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
