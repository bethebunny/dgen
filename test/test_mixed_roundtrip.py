"""Round-trip tests for mixed-dialect parsing using import headers."""

from dgen import asm
from dgen.asm.parser import parse_module
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
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<6>()
        |     %1 : number.Float64 = 1.0
        |     %store : Nil = memory.store(%0, %1, %0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_llvm_full_loop():
    """Full LLVM loop pattern parsed with import headers — block args, no phi."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : Nil = llvm.alloca<3>()
        |     %init : index.Index = 0
        |     %loop_header : goto.Label = goto.label([]) body(%i0: index.Index):
        |         %hi : index.Index = 3
        |         %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        |         %loop_body : goto.Label = goto.label([]) body(%j: index.Index):
        |             %one : index.Index = 1
        |             %next : Nil = llvm.add(%j, %one)
        |             %_ : Nil = goto.branch<%loop_header>([%next])
        |         %loop_exit : goto.Label = goto.label([]) body():
        |             %_ : Nil = ()
        |         %_ : Nil = goto.conditional_branch<%loop_body, %loop_exit>(%cmp, [%i0], [])
        |     %br : Nil = goto.branch<%loop_header>([%init])
        |     %c0 : Nil = chain(%0, %ret)
        |     %_ : Nil = chain(%br, %c0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
