"""Minimal reproducer: nested loop codegen places outer continuation wrong.

The outer body's back-edge (after the inner loop) must be emitted AFTER the
inner loop's exit label, not inside the outer body block before the inner
loop's entry branch.
"""

from dgen.asm.parser import parse_module
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler, IdentityPass
from dgen.testing import strip_prefix
from toy.passes.control_flow_to_goto import ControlFlowToGoto


NESTED_FOR = strip_prefix("""
    | import control_flow
    | import function
    |
    | %test : function.Function<()> = function.function<Nil>() body():
    |     %outer : Nil = control_flow.for<0, 2>([]) body(%i: index.Index):
    |         %inner : Nil = control_flow.for<0, 2>([%i]) body(%j: index.Index, %i: index.Index):
    |             %0 : index.Index = 0
    |             %1 : Nil = chain(%0, %0)
""")


def test_nested_loop_after_control_flow_lowering(ir_snapshot):
    """Nested ForOps lowered to goto labels."""
    m = parse_module(NESTED_FOR)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)
    assert lowered == ir_snapshot


def test_nested_loop_llvm_ir(snapshot):
    """Nested loop all the way to LLVM IR — shows the codegen issue."""
    m = parse_module(NESTED_FOR)
    exe = Compiler([ControlFlowToGoto()], LLVMCodegen()).compile(m)
    assert exe.ir == snapshot
