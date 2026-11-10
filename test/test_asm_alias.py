"""%name = %ref names an existing value, so a block can end on an input or a capture."""

import pytest

from dgen import Block, asm
from dgen.asm.parser import ParseError, parse
from dgen.block import BlockArgument
from dgen.dialects import goto
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.module import pack
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_block_ends_on_its_argument():
    f = parse(
        strip_prefix("""
        | import function
        | import index
        | %f : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %_ = %x
        """)
    )
    assert isinstance(f, FunctionOp)
    assert f.body.result is f.body.args[0]
    assert asm.format(asm.parse(asm.format(f))) == asm.format(f)


def test_identity_built_in_python_round_trips():
    x = BlockArgument(name="x", type=Index())
    f = FunctionOp(
        name="f",
        body=Block(result=x, args=[x]),
        result_type=Index(),
        type=Function(arguments=pack([Index()]), result_type=Index()),
    )
    assert "    %_ = %x" in asm.format(f)
    assert_ir_equivalent(f, asm.parse(asm.format(f)))


def test_block_ends_on_a_capture():
    f = parse(
        strip_prefix("""
        | import function
        | import goto
        | import index
        | %f : function.Function<[index.Index], goto.Label> = function.function<goto.Label>() body(%x: index.Index):
        |     %inner : goto.Label = goto.label([]) body() captures(%x):
        |         %_ = %x
        """)
    )
    assert isinstance(f, FunctionOp)
    assert isinstance(inner := f.body.result, goto.LabelOp)
    assert inner.body.result is f.body.args[0]
    assert_ir_equivalent(f, asm.parse(asm.format(f)))


def test_alias_names_the_value_for_later_statements():
    value = parse(
        strip_prefix("""
        | import index
        | %a : index.Index = 3
        | %b = %a
        | %c : index.Index = chain(%b, ())
        """)
    )
    assert asm.format(value) == asm.format(
        parse(
            strip_prefix("""
            | import index
            | %a : index.Index = 3
            | %c : index.Index = chain(%a, ())
            """)
        )
    )


def test_alias_takes_no_type():
    text = strip_prefix("""
        | import index
        | %a : index.Index = 3
        | %b : index.Index = %a
    """)
    with pytest.raises(ParseError, match="%b: an alias takes no type"):
        parse(text)
