"""Tests for builtin.if with per-branch explicit capture.

The if op now takes then_args: List and else_args: List operands.  Values
threaded via these lists become block arguments on the respective branch
blocks, making each branch a closed term.
"""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_if_no_capture_roundtrip():
    """if with no captured values: then_args=[], else_args=[] — no block args."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() body(%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %result : Index = if(%cond, [], []) then_body():
        |         %ten : Index = 10
        |     else_body():
        |         %twenty : Index = 20
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, builtin.FunctionOp)
    if_op = fn.body.ops[-1]
    assert isinstance(if_op, builtin.IfOp)
    assert if_op.then_body.args == []
    assert if_op.else_body.args == []
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_if_with_capture_roundtrip():
    """if with captured value threaded through then_args and else_args."""
    ir = strip_prefix("""
        | %main : Nil = function<F64>() body(%cond: Index, %x: F64):
        |     %result : F64 = if(%cond, [%x], [%x]) then_body(%x: F64):
        |         %a : F64 = 1.0
        |     else_body(%x: F64):
        |         %b : F64 = 2.0
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, builtin.FunctionOp)
    if_op = fn.body.ops[-1]
    assert isinstance(if_op, builtin.IfOp)
    assert len(if_op.then_body.args) == 1
    assert len(if_op.else_body.args) == 1
    assert if_op.then_body.args[0].name == "x"
    assert if_op.else_body.args[0].name == "x"
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_if_python_api_no_capture():
    """Construct IfOp directly via ASM and verify structure."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() body(%cond: Index):
        |     %result : Index = if(%cond, [], []) then_body():
        |         %ten : Index = 10
        |     else_body():
        |         %twenty : Index = 20
    """)
    module = parse_module(ir)
    fn = module.ops[0]
    assert isinstance(fn, builtin.FunctionOp)
    if_op = fn.body.ops[-1]
    assert isinstance(if_op, builtin.IfOp)
    assert if_op.then_body.args == []
    assert if_op.else_body.args == []
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
