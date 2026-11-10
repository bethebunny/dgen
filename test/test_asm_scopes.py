"""A block body sees only its parameters, arguments, captures and own statements."""

import pytest

from dgen import asm
from dgen.asm.parser import ParseError, parse
from dgen.testing import assert_ir_equivalent, strip_prefix

UNCAPTURED = [
    pytest.param(
        """
        | import function
        | import goto
        | import index
        | import llvm
        | %f : function.Function<[index.Index], ()> = function.function<Nil>() body(%x: index.Index):
        |     %inner : goto.Label = goto.label([]) body():
        |         %0 : index.Index = 0
        |         %1 : llvm.Int<64> = llvm.add(%x, %0)
        """,
        "%x is not in scope in the body of %inner: capture it",
        id="block argument",
    ),
    pytest.param(
        """
        | import function
        | import goto
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %header : goto.Label = goto.label([]) body<%self: goto.Label>():
        |         %body : goto.Label = goto.label([]) body():
        |             %0 : Nil = goto.branch<%self>([])
        """,
        "%self is not in scope in the body of %body: capture it",
        id="block parameter",
    ),
    pytest.param(
        """
        | import algebra
        | import function
        | import index
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %result : index.Index = function.call<%add_one>([%x])
        """,
        "%add_one is not in scope in the body of %main: capture it",
        id="function",
    ),
    pytest.param(
        """
        | import function
        | import goto
        | import index
        | import llvm
        | %f : function.Function<[index.Index], ()> = function.function<Nil>() body(%x: index.Index):
        |     %mid : goto.Label = goto.label([]) body():
        |         %inner : goto.Label = goto.label([]) body() captures(%x):
        |             %0 : index.Index = 0
        |             %1 : llvm.Int<64> = llvm.add(%x, %0)
        """,
        "%x is not in scope in the body of %mid: capture it",
        id="unchained capture",
    ),
    pytest.param(
        """
        | import algebra
        | import control_flow
        | import function
        | import index
        | import number
        | %main : function.Function<[index.Index], ()> = function.function<Nil>() body(%limit: index.Index):
        |     %zero : index.Index = 0
        |     %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |         %cmp : number.Boolean = algebra.less_than(%i, %limit)
        |     body(%i: index.Index):
        |         %one : index.Index = 1
        |         %next : index.Index = algebra.add(%i, %one)
        """,
        "%limit is not in scope in the condition of %loop: capture it",
        id="second block",
    ),
    pytest.param(
        """
        | import function
        | import goto
        | import index
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %inner : goto.Label = goto.label([]) body():
        |         %0 : index.Index = 0
        |     %1 : index.Index = chain(%0, %inner)
        """,
        "Undefined reference: %0",
        id="name from a closed block",
    ),
]


@pytest.mark.parametrize(("text", "message"), UNCAPTURED)
def test_uncaptured_reference_fails(text: str, message: str):
    with pytest.raises(ParseError, match=message):
        parse(strip_prefix(text))


def test_undefined_reference_fails():
    text = strip_prefix("""
        | import index
        | %0 : index.Index = chain(%1, ())
    """)
    with pytest.raises(ParseError, match="Undefined reference: %1"):
        parse(text)


def test_header_types_see_enclosing_scope():
    value = parse(
        strip_prefix("""
        | import function
        | import goto
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %t : Type = {"tag": "index.Index"}
        |     %inner : goto.Label = goto.label([]) body(%x: %t) captures(%t):
        |         %0 : %t = chain(%x, ())
        """)
    )
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_captured_op_stays_outside_after_round_trip():
    text = strip_prefix("""
        | import function
        | import goto
        | import index
        | %f : function.Function<[], index.Index> = function.function<Nil>() body():
        |     %outer : index.Index = 7
        |     %inner : goto.Label = goto.label([]) body() captures(%outer):
        |         %0 : index.Index = chain(%outer, ())
        |     %1 : index.Index = chain(%outer, %inner)
    """)
    value = parse(text)
    assert asm.format(asm.parse(asm.format(value))) == asm.format(value)
    assert "    %outer" in asm.format(value)
