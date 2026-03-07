"""Tests for type values as first-class SSA citizens."""

from dgen import Block, TypeType, Value, asm
from dgen.asm.formatting import format_expr
from dgen.asm.parser import IRParser, parse_expr, parse_module
from dgen.block import BlockArgument
from dgen.codegen import compile as compile_module
from dgen.dialects import builtin
from dgen.dialects.builtin import FunctionOp, Index
from dgen.module import ConstantOp, Function, Module
from dgen.type import Memory
from toy.test.helpers import strip_prefix


def test_parse_dict_literal():
    """parse_expr handles {key: value, ...} and returns a Python dict."""
    parser = IRParser('{"tag": "builtin.Index"}')
    result = parse_expr(parser)
    assert result == {"tag": "builtin.Index"}


def test_format_dict_literal():
    """format_expr handles dicts."""
    result = format_expr({"tag": "builtin.Index"})
    assert result == '{"tag": "builtin.Index"}'


def test_typetype_constant_asm_roundtrip():
    """TypeType constant with dict literal round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_ssa_ref_as_op_type():
    """SSA ref in type position: %x's type is unresolved Value, op not ready."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    t_op = ops[0]  # %t = TypeType constant
    x_op = ops[1]  # %x : %t = 42
    # %x's type is the SSA value %t, not a resolved Type
    assert x_op.type is t_op
    assert not x_op.ready


def test_ssa_ref_as_op_type_roundtrip():
    """SSA ref in type position round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
