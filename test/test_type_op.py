"""Tests for the builtin type op: op type(value) -> Type."""

from dgen import Block, asm
from dgen.asm.parser import parse_module
from dgen.dialects.builtin import TypeOp
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.testing import assert_ir_equivalent, strip_prefix
from dgen.type import TypeType


def test_type_op_construction():
    """TypeOp takes a value and defaults to TypeType() result type."""
    x = ConstantOp(name="x", value=42, type=Index())
    t = TypeOp(name="t", value=x)
    assert isinstance(t.type, TypeType)
    assert t.value is x


def test_type_op_in_block_ops():
    """TypeOp participates in use-def: reachable from block.result."""
    x = ConstantOp(name="x", value=42, type=Index())
    t = TypeOp(name="t", value=x)
    block = Block(result=t)
    assert x in block.ops
    assert t in block.ops


def test_type_op_asm_roundtrip():
    """TypeOp round-trips through ASM print/parse."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %x : index.Index = 42
        |     %t : Type = type(%x)
    """)
    module = parse_module(ir)
    func = module.functions[0]
    ops = func.body.ops
    assert isinstance(ops[-1], TypeOp)
    assert isinstance(ops[-1].type, TypeType)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_type_op_result_is_type_dependency():
    """TypeOp's result can be used as a type for another op."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %x : index.Index = 42
        |     %t : Type = type(%x)
        |     %y : %t = 7
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    type_op = ops[1]
    y_op = ops[2]
    assert isinstance(type_op, TypeOp)
    # %y's type is the SSA value %t (the TypeOp)
    assert y_op.type is type_op
