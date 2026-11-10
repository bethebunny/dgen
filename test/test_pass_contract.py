"""Pass declarations: what a pass lowers and emits, checked by the Compiler."""

import pytest

import dgen
from dgen import asm
from dgen.asm.parser import parse
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler, IdentityPass, PipelineError
from dgen.dialects import algebra, control_flow, llvm
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.pass_ import LoweringError, Pass, lowering_for, rewrite_for
from dgen.testing import strip_prefix

ADD_IR = strip_prefix("""
    | import algebra
    | import function
    | import index
    |
    | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
    |     %y : index.Index = algebra.add(%x, %x)
""")

WHILE_IR = strip_prefix("""
    | import algebra
    | import control_flow
    | import function
    | import index
    | import number
    |
    | %main : function.Function<[], ()> = function.function<Nil>() body():
    |     %zero : index.Index = 0
    |     %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
    |         %ten : index.Index = 10
    |         %cmp : number.Boolean = algebra.less_than(%i, %ten)
    |     body(%i: index.Index):
    |         %one : index.Index = 1
    |         %next : index.Index = algebra.add(%i, %one)
""")


class DecliningLowering(Pass):
    @lowering_for(algebra.AddOp)
    def lower_add(self, op: algebra.AddOp) -> dgen.Value | None:
        return None


class UndeclaredLowering(Pass):
    @lowering_for(algebra.AddOp)
    def lower_add(self, op: algebra.AddOp) -> dgen.Value | None:
        return llvm.AddOp(lhs=op.left, rhs=op.right, type=op.type)


class DecliningRewrite(Pass):
    @rewrite_for(algebra.AddOp)
    def fold_add(self, op: algebra.AddOp) -> dgen.Value | None:
        return None


def test_pipeline_rejects_misordered_passes():
    with pytest.raises(PipelineError, match="ControlFlowToGoto emits algebra.add, "):
        Compiler([AlgebraToLLVM(), ControlFlowToGoto()], LLVMCodegen())


def test_pipeline_error_lists_every_op():
    with pytest.raises(PipelineError) as info:
        Compiler([AlgebraToLLVM(), ControlFlowToGoto()], LLVMCodegen())
    assert str(info.value).splitlines() == [
        "ControlFlowToGoto emits algebra.add, which no later pass lowers and LLVMCodegen does not accept",
        "ControlFlowToGoto emits algebra.less_than, which no later pass lowers and LLVMCodegen does not accept",
    ]


def test_pipeline_accepts_ordered_passes():
    Compiler([ControlFlowToGoto(), AlgebraToLLVM()], LLVMCodegen())


def test_identity_exit_accepts_anything():
    Compiler([AlgebraToLLVM(), ControlFlowToGoto()], IdentityPass())


def test_pipeline_rejects_unlowered_input():
    with pytest.raises(PipelineError, match="the frontend emits control_flow.for, "):
        Compiler([AlgebraToLLVM()], LLVMCodegen(), inputs=(control_flow.ForOp,))


def test_pipeline_accepts_lowered_input():
    Compiler(
        [ControlFlowToGoto(), AlgebraToLLVM()],
        LLVMCodegen(),
        inputs=(control_flow.control_flow,),
    )


def test_codegen_accepts_what_it_emits():
    assert LLVMCodegen().accepts(llvm.AddOp)
    assert not LLVMCodegen().accepts(algebra.AddOp)


def test_eliminated_comes_from_lowering_handlers():
    assert UndeclaredLowering.eliminated == {algebra.AddOp}
    assert DecliningRewrite.eliminated == frozenset()


def test_eliminated_expands_dialects():
    assert AlgebraToLLVM.eliminated == set(algebra.algebra.ops.values())


def test_emitted_expands_dialects():
    assert set(llvm.llvm.ops.values()) <= AlgebraToLLVM.emitted


def test_incomplete_dialect_lowering_fails_at_definition():
    with pytest.raises(TypeError, match="algebra.subtract"):

        class Partial(Pass):
            lowers = (algebra.algebra,)

            @lowering_for(algebra.AddOp)
            def lower_add(self, op: algebra.AddOp) -> dgen.Value | None:
                return None


def test_pass_that_leaves_an_op_fails():
    with pytest.raises(LoweringError, match="DecliningLowering left algebra.add"):
        Compiler([DecliningLowering()], IdentityPass()).run(parse(ADD_IR))


def test_pass_with_undeclared_op_fails():
    with pytest.raises(LoweringError, match="UndeclaredLowering built llvm.add"):
        Compiler([UndeclaredLowering()], IdentityPass()).run(parse(ADD_IR))


def test_declared_op_passes():
    class DeclaredLowering(UndeclaredLowering):
        emits = (llvm.AddOp,)

    Compiler([DeclaredLowering()], IdentityPass()).run(parse(ADD_IR))


def test_rewrite_may_decline():
    Compiler([DecliningRewrite()], IdentityPass()).run(parse(ADD_IR))


def test_input_outside_inputs_fails():
    compiler = Compiler([], IdentityPass(), inputs=(algebra.SubtractOp,))
    with pytest.raises(LoweringError, match="algebra.add"):
        compiler.run(parse(ADD_IR))


def test_pass_state_lasts_one_run():
    lowering = ControlFlowToGoto()
    compiler = Compiler([lowering], IdentityPass())
    first = asm.format(compiler.run(parse(WHILE_IR)))
    second = asm.format(compiler.run(parse(WHILE_IR)))
    assert "while_header0" in first
    assert first == second
    assert vars(lowering) == {}
