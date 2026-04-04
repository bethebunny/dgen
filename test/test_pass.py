"""Tests for the Pass base class, Block.replace_uses_of, and Pass.run."""

import pytest

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.block import BlockArgument
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler, IdentityPass
from dgen.dialects.builtin import Nil
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp, Module
import dgen
from dgen.passes.pass_ import Pass, lowering_for
from dgen.staging import ConstantFold
from dgen.verify import ClosedBlockError
from toy.dialects import toy
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.testing import strip_prefix


# ---------------------------------------------------------------------------
# Task 5: @lowering_for handler registration
# ---------------------------------------------------------------------------


def test_lowering_for_registers_handler():
    class MyPass(Pass):
        @lowering_for(ConstantOp)
        def handle_constant(self, op: ConstantOp) -> dgen.Value | None:
            return None

    assert ConstantOp in MyPass._handlers
    assert len(MyPass._handlers[ConstantOp]) == 1


def test_multiple_handlers_per_op_type():
    class MyPass(Pass):
        @lowering_for(ConstantOp)
        def handler_a(self, op: ConstantOp) -> dgen.Value | None:
            return None

        @lowering_for(ConstantOp)
        def handler_b(self, op: ConstantOp) -> dgen.Value | None:
            return op

    assert len(MyPass._handlers[ConstantOp]) == 2
    names = [h.__name__ for h in MyPass._handlers[ConstantOp]]
    assert names == ["handler_a", "handler_b"]


# ---------------------------------------------------------------------------
# Task 6: Block.replace_uses_of
# ---------------------------------------------------------------------------


def test_block_replace_uses_of(ir_snapshot):
    """replace_uses_of eagerly updates all referencing ops."""
    ir_text = strip_prefix("""
        | import function
        | import index
        | import llvm
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 1
        |     %1 : index.Index = 2
        |     %2 : index.Index = llvm.add(%0, %0)
        |     %3 : index.Index = chain(%2, %1)
    """)
    m = parse_module(ir_text)
    func = m.functions[0]
    ops_by_name = {op.name: op for op in func.body.ops}
    old = ops_by_name["0"]  # %0 = 1
    new = ops_by_name["1"]  # %1 = 2

    func.body.replace_uses_of(old, new)

    assert m == ir_snapshot


def test_block_replace_uses_of_op_type():
    """replace_uses_of updates op.type when it references the replaced value."""
    old_type = ConstantOp(
        name="old_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    new_type = ConstantOp(
        name="new_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    x = ConstantOp(name="x", value=42, type=old_type)
    block = dgen.Block(result=x)

    assert x.type is old_type
    block.replace_uses_of(old_type, new_type)
    assert x.type is new_type


def test_block_replace_uses_of_block_arg_type():
    """replace_uses_of updates block argument types."""
    from dgen.dialects.index import Index

    old_type = ConstantOp(
        name="old_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    new_type = ConstantOp(
        name="new_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    arg = BlockArgument(name="x", type=old_type)
    result = ConstantOp(name="r", value=0, type=Index())
    block = dgen.Block(result=result, args=[arg])

    assert arg.type is old_type
    block.replace_uses_of(old_type, new_type)
    assert arg.type is new_type


def test_block_replace_uses_of_captures():
    """replace_uses_of updates captures and ops referencing the captured value."""
    from dgen.dialects.index import Index
    from dgen.dialects.builtin import ChainOp

    old_val = ConstantOp(name="old", value=1, type=Index())
    new_val = ConstantOp(name="new", value=2, type=Index())
    inner_op = ChainOp(name="use", lhs=old_val, rhs=old_val, type=Index())
    block = dgen.Block(result=inner_op, captures=[old_val])

    assert old_val in block.captures
    assert inner_op.lhs is old_val
    block.replace_uses_of(old_val, new_val)
    assert new_val in block.captures
    assert old_val not in block.captures
    assert inner_op.lhs is new_val
    assert inner_op.rhs is new_val


def test_block_replace_uses_of_block_parameter_type():
    """replace_uses_of updates block parameter types."""
    from dgen.block import BlockParameter
    from dgen.dialects.index import Index

    old_type = ConstantOp(
        name="old_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    new_type = ConstantOp(
        name="new_t", value={"tag": "index.Index"}, type=dgen.TypeType()
    )
    param = BlockParameter(name="p", type=old_type)
    result = ConstantOp(name="r", value=0, type=Index())
    block = dgen.Block(result=result, parameters=[param])

    assert param.type is old_type
    block.replace_uses_of(old_type, new_type)
    assert param.type is new_type


# ---------------------------------------------------------------------------
# Task 7: Pass.run — walk graph, dispatch handlers
# ---------------------------------------------------------------------------


def test_pass_run_eliminates_double_transpose(ir_snapshot):
    """A pass that eliminates transpose(transpose(x)) -> x."""
    ir_text = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<ndbuffer.Shape<2>([3, 2]), number.Float64> = toy.transpose(%0)
        |     %2 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = toy.transpose(%1)
        |     %3 : Nil = toy.print(%2)
    """)

    class ElimTranspose(Pass):
        allow_unregistered_ops = True

        @lowering_for(toy.TransposeOp)
        def eliminate(self, op: toy.TransposeOp) -> dgen.Value | None:
            if not isinstance(op.input, toy.TransposeOp):
                return None
            return op.input.input

    m = parse_module(ir_text)
    compiler = Compiler(passes=[ElimTranspose()], exit=IdentityPass())
    result = compiler.run(m)
    assert result == ir_snapshot


def test_pass_unregistered_ops_error():
    """allow_unregistered_ops=False raises on unhandled ops."""
    ir_text = strip_prefix("""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)

    class StrictPass(Pass):
        allow_unregistered_ops = False

    m = parse_module(ir_text)
    compiler = Compiler(passes=[StrictPass()], exit=IdentityPass())
    with pytest.raises(TypeError, match="No handler for"):
        compiler.run(m)


def test_pass_multiple_handlers_first_wins():
    """Multiple handlers: first one that returns True wins."""
    call_log: list[str] = []

    class MultiPass(Pass):
        allow_unregistered_ops = True

        @lowering_for(ConstantOp)
        def first(self, op: ConstantOp) -> dgen.Value | None:
            call_log.append("first")
            return op

        @lowering_for(ConstantOp)
        def second(self, op: ConstantOp) -> dgen.Value | None:
            call_log.append("second")
            return op

    ir_text = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    m = parse_module(ir_text)
    compiler = Compiler(passes=[MultiPass()], exit=IdentityPass())
    compiler.run(m)
    assert call_log == ["first"]


# ---------------------------------------------------------------------------
# Task 8: Compiler.run with verification
# ---------------------------------------------------------------------------


def test_compiler_run_verification_catches_closed_block_violation():
    """Post-condition check detects a closed-block violation introduced by a pass."""
    ir_text = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : Nil = {}
    """)

    class CorruptPass(Pass):
        """Introduces a closed-block violation by replacing the block result
        with a BlockArgument not in the block's args list."""

        allow_unregistered_ops = True

        def run(self, value: dgen.Value, compiler: Compiler) -> dgen.Value:
            if isinstance(value, FunctionOp):
                # Replace result with a foreign BlockArgument — a clear
                # closed-block violation since it is not in block.args.
                value.body.result = BlockArgument(type=Nil())
            return value

    m = parse_module(ir_text)
    compiler_inst: Compiler = Compiler(passes=[CorruptPass()], exit=IdentityPass())
    with pytest.raises(ClosedBlockError):
        compiler_inst.run(m)


# ---------------------------------------------------------------------------
# ConstantFold pass
# ---------------------------------------------------------------------------


def test_constant_fold_resolves_stage0_boundary():
    """ConstantFold pass resolves stage-0 type boundaries in the pipeline.

    %t is a ConstantOp producing a type value. The inner function uses %t
    as its result type — a stage-0 boundary. ConstantFold should resolve it
    so that subsequent passes see a concrete type.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %f : function.Function<[Type, index.Index], %t> = function.function<%t>() body(%rt: Type, %x: index.Index):
        |         %y : %rt = algebra.add(%x, %x)
    """)
    module = parse_module(ir)
    inner_func = next(
        op for op in module.functions[0].body.ops if isinstance(op, FunctionOp)
    )

    # Before constant folding: result is a ConstantOp SSA ref, not a Type
    assert isinstance(inner_func.result_type, ConstantOp)

    class LowerToLLVMPass(Pass):
        allow_unregistered_ops = True

        def run(self, value: dgen.Value, compiler: Compiler) -> dgen.Value:
            value = ControlFlowToGoto().run(value, compiler)
            value = NDBufferToMemory().run(value, compiler)
            return MemoryToLLVM().run(value, compiler)

    compiler: Compiler[Executable] = Compiler(
        passes=[ConstantFold(), LowerToLLVMPass()],
        exit=LLVMCodegen(),
    )
    exe = compiler.compile(Module(ops=[inner_func]))
    assert exe.run({"tag": "index.Index"}, 21).to_json() == 42


def test_constant_fold_is_noop_without_boundaries():
    """ConstantFold does nothing when there are no stage-0 boundaries."""
    ir_text = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    m = parse_module(ir_text)
    before = asm.format(m)
    compiler = Compiler(passes=[ConstantFold()], exit=IdentityPass())
    result = compiler.run(m)
    after = asm.format(result)
    assert after == before


def test_pass_run_receives_continuation_compiler():
    """Pass.run receives a Compiler representing the remaining pipeline."""
    seen_pass_count: list[int] = []

    class SpyPass(Pass):
        allow_unregistered_ops = True

        def run(self, value: dgen.Value, compiler: Compiler) -> dgen.Value:
            seen_pass_count.append(len(compiler.passes))
            return value

    ir_text = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    m = parse_module(ir_text)
    compiler = Compiler(passes=[SpyPass(), SpyPass(), SpyPass()], exit=IdentityPass())
    compiler.run(m)
    # First pass sees 2 remaining, second sees 1, third sees 0
    assert seen_pass_count == [2, 1, 0]
