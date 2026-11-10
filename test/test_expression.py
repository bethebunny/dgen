"""Tests for the requires-clause expression language."""

import re

import pytest

from dgen.gen.expression import (
    Attribute,
    Binary,
    Literal,
    Name,
    TypeOf,
    Unary,
    parse_expression,
)


def test_parse_attribute_path() -> None:
    assert parse_expression("axis < lhs.shape.rank") == Binary(
        "<", Name("axis"), Attribute(Attribute(Name("lhs"), "shape"), "rank")
    )


def test_parse_type_of() -> None:
    assert parse_expression("$X.dtype == $Result.dtype") == Binary(
        "==", Attribute(TypeOf("X"), "dtype"), Attribute(TypeOf("Result"), "dtype")
    )


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("a + b * c", "(a + (b * c))"),
        ("(a + b) * c", "((a + b) * c)"),
        ("a - b - c", "((a - b) - c)"),
        ("-a // 2 % 3", "((-a // 2) % 3)"),
        ("a < b + 1 and not c == d or e", "(((a < (b + 1)) and not (c == d)) or e)"),
        ("not not a", "not not a"),
        ("a.b.c >= 10", "(a.b.c >= 10)"),
    ],
)
def test_parse_precedence(text: str, parsed: str) -> None:
    assert str(parse_expression(text)) == parsed


def test_parse_literal() -> None:
    assert parse_expression("-12") == Unary("-", Literal(12))


@pytest.mark.parametrize(
    "text",
    ["$X ~= Tensor", "a < b < c", "a +", "a b", "(a", "a.", "a.1", "and", "'x' == a"],
)
def test_parse_errors(text: str) -> None:
    with pytest.raises(SyntaxError, match=re.escape(repr(text))):
        parse_expression(text)
