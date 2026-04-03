"""Tests for control_flow.if: ASM round-trip, lowering to goto, and JIT.

The if op takes then_arguments: List and else_arguments: List operands.  Values
threaded via these lists become block arguments on the respective branch
blocks, making each branch a closed term.
"""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.compiler import Compiler, IdentityPass
from dgen.dialects import control_flow, function
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_if_no_capture_roundtrip():
    """if with no captured values: then_arguments=[], else_arguments=[] — no block args."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | import function
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %ten : index.Index = 10
        |     else_body():
        |         %twenty : index.Index = 20
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, function.FunctionOp)
    if_op = fn.body.result
    assert isinstance(if_op, control_flow.IfOp)
    assert if_op.then_body.args == []
    assert if_op.else_body.args == []
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_if_with_capture_roundtrip():
    """if with captured value threaded through then_arguments and else_arguments."""
    ir = strip_prefix("""
        | import control_flow
        | import index
        | import function
        | import index
        | import number
        | %main : function.Function<[index.Index, number.Float64], number.Float64> = function.function<number.Float64>() body(%cond: index.Index, %x: number.Float64):
        |     %result : number.Float64 = control_flow.if(%cond, [%x], [%x]) then_body(%x: number.Float64):
        |         %a : number.Float64 = 1.0
        |     else_body(%x: number.Float64):
        |         %b : number.Float64 = 2.0
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, function.FunctionOp)
    if_op = fn.body.result
    assert isinstance(if_op, control_flow.IfOp)
    assert len(if_op.then_body.args) == 1
    assert len(if_op.else_body.args) == 1
    assert if_op.then_body.args[0].name == "x"
    assert if_op.else_body.args[0].name == "x"
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_if_python_api_no_capture():
    """Construct IfOp directly via ASM and verify structure."""
    ir = strip_prefix("""
        | import control_flow
        | import index
        | import function
        | import index
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%cond: index.Index):
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %ten : index.Index = 10
        |     else_body():
        |         %twenty : index.Index = 20
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, function.FunctionOp)
    if_op = fn.body.result
    assert isinstance(if_op, control_flow.IfOp)
    assert if_op.then_body.args == []
    assert if_op.else_body.args == []
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


# -- Lowering to goto --


def test_if_value_lowering(ir_snapshot):
    """Value-producing if lowers to a region with %self merge and result phi."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | import function
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %ten : index.Index = 10
        |     else_body():
        |         %twenty : index.Index = 20
    """)
    module = parse_module(ir)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(module)
    assert lowered == ir_snapshot


def test_if_void_lowering(ir_snapshot):
    """Void if (side-effect only) lowers to the same structure as value ifs."""
    ir = strip_prefix("""
        | import control_flow
        | import function
        | import index
        | import memory
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %cond : index.Index = 1
        |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |         %t : index.Index = 42
        |         %_ : Nil = memory.store(%alloc, %t, %alloc)
        |     else_body() captures(%alloc):
        |         %f : index.Index = 99
        |         %_ : Nil = memory.store(%alloc, %f, %alloc)
        |     %result : index.Index = memory.load(%if, %alloc)
    """)
    module = parse_module(ir)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(module)
    assert lowered == ir_snapshot


def test_if_with_branch_arguments_lowering(ir_snapshot):
    """If with per-branch arguments threads values into branch bodies."""
    ir = strip_prefix("""
        | import control_flow
        | import index
        | import function
        | import number
        | %main : function.Function<[index.Index, number.Float64], number.Float64> = function.function<number.Float64>() body(%cond: index.Index, %x: number.Float64):
        |     %result : number.Float64 = control_flow.if(%cond, [%x], [%x]) then_body(%tx: number.Float64):
        |         %a : number.Float64 = 1.0
        |     else_body(%ex: number.Float64):
        |         %b : number.Float64 = 2.0
    """)
    module = parse_module(ir)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(module)
    assert lowered == ir_snapshot


def test_if_type_mismatch_then():
    """IfOp with then-branch result type mismatch is caught in preconditions."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | import function
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %a : number.Float64 = 1.0
        |     else_body():
        |         %b : index.Index = 20
    """)
    module = parse_module(ir)
    import pytest

    with pytest.raises(TypeError, match="then-branch result type"):
        Compiler([ControlFlowToGoto()], IdentityPass()).compile(module)


def test_if_type_mismatch_else():
    """IfOp with else-branch result type mismatch is caught in preconditions."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | import function
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %a : index.Index = 10
        |     else_body():
        |         %b : number.Float64 = 2.0
    """)
    module = parse_module(ir)
    import pytest

    with pytest.raises(TypeError, match="else-branch result type"):
        Compiler([ControlFlowToGoto()], IdentityPass()).compile(module)
