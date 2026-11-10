"""Stage boundaries inside nested blocks, eg. an if body."""

import pytest

from dgen.asm.parser import parse
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.staging import StagingError
from dgen.testing import strip_prefix

COMPILER = Compiler(
    [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()], LLVMCodegen()
)


def test_stage0_boundary_in_if_body():
    ir = strip_prefix("""
        | import algebra
        | import control_flow
        | import function
        | import index
        | import number
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %c : number.Boolean = algebra.equal(%x, 0)
        |     %r : index.Index = control_flow.if(%c, [], []) then_body():
        |         %a : index.Index = 21
        |         %t : Type = type(%a)
        |         %y : %t = algebra.add(%a, %a)
        |     else_body():
        |         %b : index.Index = 1
    """)
    exe = COMPILER.compile(parse(ir))
    assert [exe.run(0).to_json(), exe.run(1).to_json()] == [42, 1]


def test_stage1_boundary_in_if_body_specializes_on_call():
    ir = strip_prefix("""
        | import algebra
        | import control_flow
        | import function
        | import index
        | import number
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %c : number.Boolean = algebra.equal(%x, 0)
        |     %r : index.Index = control_flow.if(%c, [], []) then_body():
        |         %b : index.Index = 1
        |     else_body() captures(%x):
        |         %t : Type = type(%x)
        |         %y : %t = algebra.add(%x, %x)
    """)
    exe = COMPILER.compile(parse(ir))
    assert [exe.run(n).to_json() for n in [0, 3, 5]] == [1, 6, 10]


def test_boundary_on_nested_block_argument_raises():
    ir = strip_prefix("""
        | import algebra
        | import control_flow
        | import function
        | import index
        | import number
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %c : number.Boolean = algebra.equal(%x, 0)
        |     %r : index.Index = control_flow.if(%c, [], [%x]) then_body():
        |         %b : index.Index = 1
        |     else_body(%v: index.Index):
        |         %t : Type = type(%v)
        |         %y : %t = algebra.add(%v, %v)
    """)
    with pytest.raises(StagingError, match="%y.type in function %main depends on %v"):
        COMPILER.compile(parse(ir))
