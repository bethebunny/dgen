"""IR built in Python survives print and parse, compared with itself rather than a parsed copy."""

import pytest

from dgen import Block, Value, asm
from dgen.asm.parser import ASMParser, value_expression
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.dialects.number import Boolean, Float64
from dgen.module import ConstantOp, pack
from dgen.testing import assert_ir_equivalent
from dgen.type import format_json


def _function(result: Value, *args: BlockArgument) -> FunctionOp:
    return FunctionOp(
        name="f",
        body=Block(result=result, args=list(args)),
        result_type=result.type,
        type=Function(
            arguments=pack([arg.type for arg in args]), result_type=result.type
        ),
    )


def test_constant_operand_keeps_its_type():
    n = BlockArgument(name="n", type=Index())
    cond = algebra.EqualOp(left=n, right=Index().constant(0), type=Boolean())
    value = _function(cond, n)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_constant_matching_its_field_prints_bare():
    x = BlockArgument(name="x", type=Float64())
    total = algebra.AddOp(left=x, right=Float64().constant(1.0), type=Float64())
    text = asm.format(_function(total, x))
    assert "algebra.add(%x, 1.0)" in text


STRINGS = [
    pytest.param('say "hi"', id="quote"),
    pytest.param("back\\slash", id="backslash"),
    pytest.param("line\nbreak", id="newline"),
    pytest.param("tab\tand ünïcode", id="tab and unicode"),
]


@pytest.mark.parametrize("text", STRINGS)
def test_string_constant_round_trips(text: str):
    value = _function(ConstantOp(value=text, type=builtin.String()))
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


@pytest.mark.parametrize("text", STRINGS)
def test_dict_key_round_trips(text: str):
    literal = {text: [text]}
    assert value_expression(ASMParser(format_json(literal))) == literal
