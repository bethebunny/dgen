"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_llvm_via_imports():
    """Parse a function using llvm ops via import headers."""
    ir = strip_prefix("""
        | import function
        | import llvm
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : F64 = 1.0
        |     %store : Nil = llvm.store(%1, %0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers — block args, no phi."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : Index = 0
        |     %loop_header : goto.Label = goto.label() body(%i0: Index):
        |         %hi : Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        |         %loop_body : goto.Label = goto.label() body(%j: Index):
        |             %one : Index = 1
        |             %next : Nil = llvm.add(%j, %one)
        |             %_ : Nil = goto.branch<%loop_header>([%next])
        |         %loop_exit : goto.Label = goto.label() body():
        |             %_ : Nil = ()
        |         %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [%i0], [])
        |     %br : Nil = goto.branch<%loop_header>([%init])
        |     %c0 : Nil = chain(%0, %ret)
        |     %_ : Nil = chain(%br, %c0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
