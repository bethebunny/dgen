"""Tests for ASMParser core tokenizer with read/try_read protocol."""

from __future__ import annotations

import re

import pytest

import toy.dialects.toy  # noqa: F401 — registers dialect

from dgen.asm.parser import ASMParser, ParseError, parse_module
from dgen.testing import strip_prefix

_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class TestASMParserBasics:
    def test_done_on_empty(self) -> None:
        assert ASMParser("").done

    def test_done_with_whitespace(self) -> None:
        assert ASMParser("   \n\n  ").done

    def test_not_done(self) -> None:
        assert not ASMParser("hello").done

    def test_peek_at_end(self) -> None:
        assert ASMParser("").peek() == ""

    def test_peek_returns_current(self) -> None:
        assert ASMParser("abc").peek() == "a"


class TestReadAndTryRead:
    def test_read_string(self) -> None:
        p = ASMParser("  (")
        p.read("(")
        assert p.pos == 3

    def test_read_raises_on_mismatch(self) -> None:
        with pytest.raises(RuntimeError):
            ASMParser("  )").read("(")

    def test_try_read_success(self) -> None:
        p = ASMParser("  (")
        assert p.try_read("(") == "("
        assert p.pos == 3

    def test_try_read_failure_restores_pos(self) -> None:
        p = ASMParser("  )")
        assert p.try_read("(") is None
        assert p.pos == 0

    def test_read_function(self) -> None:
        def ident(parser: ASMParser) -> str:
            return parser.expect_token(_IDENT, "identifier")

        assert ASMParser("  hello").read(ident) == "hello"

    def test_try_read_function_failure(self) -> None:
        def fail(parser: ASMParser) -> int:
            raise ParseError("no match")

        p = ASMParser("  abc")
        assert p.try_read(fail) is None
        assert p.pos == 0


class TestTokenMethods:
    def test_parse_token(self) -> None:
        p = ASMParser("  hello world")
        assert p.parse_token(_IDENT) == "hello"

    def test_parse_token_at_end(self) -> None:
        assert ASMParser("   ").parse_token(_IDENT) is None

    def test_expect_token(self) -> None:
        assert ASMParser("  hello").expect_token(_IDENT, "identifier") == "hello"

    def test_expect_token_fails(self) -> None:
        with pytest.raises(RuntimeError, match="Expected identifier"):
            ASMParser("   ").expect_token(_IDENT, "identifier")


class TestReadList:
    def test_empty(self) -> None:
        def number(parser: ASMParser) -> int:
            token = parser.parse_token(re.compile(r"\d+"))
            if token is None:
                raise ParseError("expected number")
            return int(token)

        assert ASMParser("").read_list(number) == []

    def test_single(self) -> None:
        def number(parser: ASMParser) -> int:
            token = parser.parse_token(re.compile(r"\d+"))
            if token is None:
                raise ParseError("expected number")
            return int(token)

        assert ASMParser("42").read_list(number) == [42]

    def test_multiple(self) -> None:
        def number(parser: ASMParser) -> int:
            token = parser.parse_token(re.compile(r"\d+"))
            if token is None:
                raise ParseError("expected number")
            return int(token)

        assert ASMParser("1, 2, 3").read_list(number) == [1, 2, 3]


class TestParseErrors:
    """Error reporting for common ASM mistakes."""

    def test_bare_span_for_parameterized_type(self) -> None:
        ir = strip_prefix("""
            | import function
            | import ndbuffer
            | import number
            | import toy
            |
            | %f : function.Function<toy.Tensor<[2, 3], number.Float64>> = function.function<toy.Tensor<[2, 3], number.Float64>>() body():
        """)
        with pytest.raises(RuntimeError, match=r"bare literal.*Shape.*Shape<\.\.\.>"):
            parse_module(ir)

    def test_unimported_dialect_rejected(self) -> None:
        """Referencing a dialect not declared via `import` should fail."""
        ir = strip_prefix("""
            | import function
            | import toy
            |
            | %f : function.Function<()> = function.function<Nil>() body():
            |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        """)
        with pytest.raises(Exception):
            parse_module(ir)
