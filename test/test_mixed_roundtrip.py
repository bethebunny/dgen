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
        |     %store : Nil = llvm.store(%1, %0)
        |     %_ : Nil = return(%store)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers — block args, no phi."""
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %loop_header : llvm.Label = llvm.label() (%i0: Index):
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        |         %loop_body : llvm.Label = llvm.label() (%j: Index):
        |             %one : Index = 1
        |             %next : Nil = llvm.add(%j, %one)
        |             %_ : Nil = llvm.br(%loop_header, [%next])
        |         %loop_exit : llvm.Label = llvm.label() ():
        |             %_ : Nil = return(())
        |         %_ : Nil = llvm.cond_br(%cmp, %loop_body, %loop_exit, [%i0], [])
        |     %br : Nil = llvm.br(%loop_header, [%init])
        |     %ret : Nil = return(())
        |     %c0 : Nil = chain(%0, %ret)
        |     %_ : Nil = chain(%br, %c0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
