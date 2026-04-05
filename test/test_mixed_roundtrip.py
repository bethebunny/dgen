"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import index as _index  # noqa: F401 — register index dialect
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_llvm_via_imports():
    """Parse a function using memory ops via import headers."""
    ir = strip_prefix("""
        | import function
        | import llvm
        | import memory
        | import number
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : number.Float64 = 1.0
        |     %store : Nil = memory.store(%0, %1, %0)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers — block args, no phi."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : index.Index = 0
        |     %loop_header : goto.Label = goto.label([]) body<%self: goto.Label, %exit: goto.Label>(%i0: index.Index):
        |         %hi : index.Index = 3
        |         %cmp : llvm.Int<1> = llvm.icmp<"slt">(%i0, %hi)
        |         %loop_body : goto.Label = goto.label([]) body(%j: index.Index) captures(%self):
        |             %one : index.Index = 1
        |             %next : llvm.Int<64> = llvm.add(%j, %one)
        |             %_ : Nil = goto.branch<%self>([%next])
        |         %loop_exit : goto.Label = goto.label([]) body():
        |             %_0 : Nil = ()
        |         %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [%i0], [])
        |     %br : Nil = goto.branch<%loop_header>([%init])
        |     %_ : Nil = chain(%0, %br)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
