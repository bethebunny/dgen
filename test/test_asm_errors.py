"""Malformed asm, with the message, line and column each one fails with."""

import re

import pytest

from dgen.asm.parser import ParseError, parse
from dgen.testing import strip_prefix

HEADER = "import algebra\nimport function\nimport index\n"

MALFORMED = [
    pytest.param(
        "%a : index.Index = 3\n%b : index.Index = algebr.add(%a, %a)\n",
        "Unknown name: algebr.add",
        (5, 20),
        id="unknown op",
    ),
    pytest.param(
        "import algebr\n%a : index.Index = 3\n",
        "Unknown dialect: algebr",
        (4, 8),
        id="unknown dialect",
    ),
    pytest.param(
        "%a : index.Index = algebra.add(%b, %b)\n",
        "Undefined reference: %b",
        (4, 32),
        id="undefined reference",
    ),
    pytest.param(
        "%a : index.Index = 3\n%f : function.Function<[], index.Index> = function.function<index.Index>() body():\n    %b : index.Index = algebra.add(%a, %a)\n",
        "%a is not in scope in the body of %f: capture it",
        (6, 36),
        id="uncaptured",
    ),
    pytest.param(
        "%a : index.Index 3\n",
        "Expected '='",
        (4, 18),
        id="missing equals",
    ),
    pytest.param(
        "a : index.Index = 3\n",
        "Expected SSA name",
        (4, 1),
        id="missing percent",
    ),
    pytest.param(
        "%a : index.Index = 3\n%b : index.Index = algebra.add(%a, %a\n",
        "Expected ')'",
        (5, 38),
        id="unclosed operands",
    ),
    pytest.param(
        "%a : index.Index = index.Index()\n",
        "Expected an op, got type index.Index",
        (4, 20),
        id="type for op",
    ),
    pytest.param(
        "%a : algebra.add = 3\n",
        "Expected a type, got op algebra.add",
        (4, 6),
        id="op for type",
    ),
    pytest.param(
        "%a : index.Index = 3\n%b : index.Index = algebra.add(%a, %a, %a)\n",
        "algebra.add takes 2 operands, got 3",
        (5, 1),
        id="extra operand",
    ),
    pytest.param(
        "%a : index.Index = 3\n%b : index.Index = algebra.add(%a)\n",
        "missing 1 required keyword-only argument: 'right'",
        (5, 1),
        id="missing operand",
    ),
    pytest.param(
        "import ndbuffer\nimport number\nimport toy\n%a : toy.Tensor<[2, 3], number.Float64> = [1.0]\n",
        "toy.Tensor: Cannot use a bare literal for parameterized type Shape",
        (7, 6),
        id="bare literal for parameterized type",
    ),
    pytest.param(
        "%f : function.Function<[], index.Index> = function.function<index.Index>() body():\n",
        "Empty block body",
        (5, 1),
        id="empty block",
    ),
    pytest.param(
        "",
        "No statements in input",
        (4, 1),
        id="no statements",
    ),
]


@pytest.mark.parametrize(("text", "message", "where"), MALFORMED)
def test_malformed(text: str, message: str, where: tuple[int, int]):
    with pytest.raises(ParseError, match=re.escape(message)) as caught:
        parse(HEADER + text)
    assert (caught.value.line, caught.value.column) == where


def test_message_shows_line_and_caret():
    text = strip_prefix("""
        | import index
        |
        | %a : index.Index = 3
        | %b : index.Index = algebr.add(%a, %a)
    """)
    with pytest.raises(ParseError) as caught:
        parse(text)
    assert str(caught.value) == (
        "line 4, column 20: Unknown name: algebr.add\n"
        "    %b : index.Index = algebr.add(%a, %a)\n"
        "                       ^"
    )


def test_error_inside_list_is_not_a_mismatch():
    text = HEADER + "%a : index.Index = 3\n%b : Nil = chain([%a, %c], ())\n"
    with pytest.raises(ParseError, match=re.escape("Undefined reference: %c")):
        parse(text)
