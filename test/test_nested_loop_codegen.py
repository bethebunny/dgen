"""Minimal reproducer: nested loop codegen places outer continuation wrong.

The outer body's back-edge (after the inner loop) must be emitted AFTER the
inner loop's exit label, not inside the outer body block before the inner
loop's entry branch.
"""

import pytest

import dgen
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import algebra, control_flow, record
from dgen.dialects.builtin import Array
from dgen.dialects.index import Index
from dgen.llvm.codegen import LLVMCodegen
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.testing import assert_valid_llvm, strip_prefix
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto


NESTED_FOR = strip_prefix("""
    | import control_flow
    | import index
    |
    | %outer : Nil = control_flow.for<index.Index(0), index.Index(2)>([]) body(%i: index.Index):
    |     %inner : Nil = control_flow.for<index.Index(0), index.Index(2)>([]) body(%j: index.Index):
    |         %0 : index.Index = 0
    |         %1 : Nil = chain(%0, %0)
""")


def test_nested_loop_after_control_flow_lowering(ir_snapshot):
    """Nested ForOps lowered to goto labels."""
    m = parse(NESTED_FOR)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)
    assert lowered == ir_snapshot


def test_for_carry_type_mismatch_rejected():
    """A ForOp carry whose body result type doesn't match the carry type is
    rejected at lowering — it would otherwise produce an invalid back-edge phi
    (a Nil next value for an i64 carry). Guards the carry-threading machinery."""
    ir = strip_prefix("""
        | import control_flow
        | import index
        |
        | %loop : Nil = control_flow.for<index.Index(0), index.Index(2)>([index.Index(0)]) body(%j: index.Index, %acc: index.Index):
        |     %0 : index.Index = 0
        |     %1 : Nil = chain(%0, %0)
    """)
    m = parse(ir)
    with pytest.raises(TypeError, match="carry"):
        Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)


def _two_carry_for(body_result_of) -> control_flow.ForOp:
    """A ForOp with two Index carries; ``body_result_of(next_a, next_b)``
    builds the body result from the next carry values."""
    iv = BlockArgument(name="i", type=Index())
    a = BlockArgument(name="a", type=Index())
    b = BlockArgument(name="b", type=Index())
    total = algebra.AddOp(left=a, right=b, type=Index())
    return control_flow.ForOp(
        lower_bound=Index().constant(0),
        upper_bound=Index().constant(3),
        initial_arguments=pack([Index().constant(1), Index().constant(2)]),
        body=dgen.Block(result=body_result_of(b, total), args=[iv, a, b]),
    )


def test_for_multi_carry_llvm_ir(snapshot):
    """Two Index carries exercise lower_for's multi-carry branch: the header
    gets one phi per carry and the back-edge threads both next values."""
    loop = _two_carry_for(lambda next_a, next_b: pack([next_a, next_b]))
    exe = Compiler(
        [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()], LLVMCodegen()
    ).compile(loop)
    assert_valid_llvm(exe.ir)
    assert exe.ir == snapshot


def test_for_multi_carry_undecomposable_result_rejected():
    """A multi-carry body result that isn't a builtin.pack (here record.pack)
    passes the type-level carry verification but can't be spliced with the
    incremented IV — lowering must reject it rather than emit a
    wrong-arity back-edge."""
    loop = _two_carry_for(
        lambda next_a, next_b: record.PackOp(
            values=pack([next_a, next_b]),
            type=Array(element_type=Index(), n=Index().constant(2)),
        )
    )
    with pytest.raises(TypeError, match="decompose"):
        Compiler([ControlFlowToGoto()], IdentityPass()).compile(loop)


def test_nested_loop_llvm_ir(snapshot):
    """Nested loop all the way to LLVM IR — shows the codegen issue."""
    m = parse(NESTED_FOR)
    exe = Compiler(
        [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()], LLVMCodegen()
    ).compile(m)
    assert_valid_llvm(exe.ir)
    assert exe.ir == snapshot
