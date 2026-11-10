"""Recursive functions: a function body's %self parameter is the function itself."""

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects.function import FunctionOp
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import assert_ir_equivalent, llvm_compile, strip_prefix
from dgen.verify import (
    VerificationError,
    verify_closed_blocks,
    verify_dag,
    verify_function_parameters,
)

FACTORIAL = strip_prefix("""
    | import algebra
    | import control_flow
    | import function
    | import index
    | import number
    |
    | %factorial : function.Function<[index.Index], index.Index> = function.function<index.Index>() body<%self: function.Function<[index.Index], index.Index>>(%n: index.Index):
    |     %base : number.Boolean = algebra.equal(%n, 0)
    |     %result : index.Index = control_flow.if(%base, [], [%n]) then_body():
    |         %one : index.Index = 1
    |     else_body(%m: index.Index) captures(%self):
    |         %m1 : index.Index = algebra.subtract(%m, 1)
    |         %rest : index.Index = function.call<%self>([%m1])
    |         %product : index.Index = algebra.multiply(%m, %rest)
""")


def test_factorial_roundtrip():
    value = parse(FACTORIAL)
    assert isinstance(value, FunctionOp)
    assert "body<%self: function.Function" in asm.format(value)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_factorial_is_a_closed_dag():
    value = parse(FACTORIAL)
    verify_dag(value)
    verify_closed_blocks(value)
    verify_function_parameters(value)


def test_factorial_jit():
    exe = llvm_compile(parse(FACTORIAL))
    assert [exe.run(n).to_json() for n in [0, 1, 5, 10]] == [1, 1, 120, 3628800]


def test_factorial_staged():
    compiler = Compiler(
        [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()], LLVMCodegen()
    )
    exe = compiler.compile(parse(FACTORIAL))
    assert exe.run(6).to_json() == 720


def test_self_parameter_must_have_the_function_type():
    wrong = FACTORIAL.replace(
        "body<%self: function.Function<[index.Index], index.Index>>",
        "body<%self: function.Function<[], index.Index>>",
    )
    with pytest.raises(VerificationError, match="only parameter may be %self"):
        verify_function_parameters(parse(wrong))


IS_EVEN = strip_prefix("""
    | import algebra
    | import control_flow
    | import function
    | import index
    | import number
    |
    | %is_even : function.Function<[index.Index], index.Index> = function.function<index.Index>() body<%self: function.Function<[index.Index], index.Index>>(%n: index.Index):
    |     %is_odd : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%m: index.Index) captures(%self):
    |         %m_zero : number.Boolean = algebra.equal(%m, 0)
    |         %odd : index.Index = control_flow.if(%m_zero, [], []) then_body():
    |             %no : index.Index = 0
    |         else_body() captures(%m, %self):
    |             %m1 : index.Index = algebra.subtract(%m, 1)
    |             %even : index.Index = function.call<%self>([%m1])
    |     %n_zero : number.Boolean = algebra.equal(%n, 0)
    |     %result : index.Index = control_flow.if(%n_zero, [], []) then_body():
    |         %yes : index.Index = 1
    |     else_body() captures(%n, %is_odd):
    |         %n1 : index.Index = algebra.subtract(%n, 1)
    |         %r : index.Index = function.call<%is_odd>([%n1])
""")


def test_mutual_recursion_roundtrip():
    value = parse(IS_EVEN)
    verify_dag(value)
    verify_closed_blocks(value)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_mutual_recursion_jit():
    exe = llvm_compile(parse(IS_EVEN))
    assert [exe.run(n).to_json() for n in range(5)] == [1, 0, 1, 0, 1]
