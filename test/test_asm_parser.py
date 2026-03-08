"""Tests for ASMParser core tokenizer with read/try_read protocol."""

from __future__ import annotations

import re

import pytest

from dgen.asm.parser import ASMParser, Namespace

_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class TestNamespace:
    def test_builtin_ops_registered(self) -> None:
        ns = Namespace()
        assert "function" in ns.ops
        assert "return" in ns.ops

    def test_builtin_types_registered(self) -> None:
        ns = Namespace()
        assert "Nil" in ns.types
        assert "Index" in ns.types

    def test_import_dialect(self) -> None:
        ns = Namespace()
        ns.import_dialect("builtin")
        # After explicit import, qualified names are available
        assert "builtin.function" in ns.ops
        assert "builtin.Nil" in ns.types

    def test_import_toy_dialect(self) -> None:
        ns = Namespace()
        ns.import_dialect("toy")
        assert "toy.transpose" in ns.ops

    def test_import_unknown_dialect_raises(self) -> None:
        ns = Namespace()
        with pytest.raises(RuntimeError, match="Unknown dialect"):
            ns.import_dialect("nonexistent_dialect_xyz")


class TestASMParserBasics:
    def test_done_on_empty(self) -> None:
        p = ASMParser("")
        assert p.done

    def test_done_with_whitespace(self) -> None:
        p = ASMParser("   \n\n  ")
        assert p.done

    def test_not_done(self) -> None:
        p = ASMParser("hello")
        assert not p.done

    def test_peek_at_end(self) -> None:
        p = ASMParser("")
        assert p.peek() == ""

    def test_peek_returns_current(self) -> None:
        p = ASMParser("abc")
        assert p.peek() == "a"


class TestReadPunctuation:
    def test_read_single_char(self) -> None:
        p = ASMParser("  (")
        p.read("(")
        assert p.pos == 3

    def test_read_multi_char(self) -> None:
        p = ASMParser("  ->")
        p.read("->")
        assert p.pos == 4

    def test_read_raises_on_mismatch(self) -> None:
        p = ASMParser("  )")
        with pytest.raises(RuntimeError):
            p.read("(")

    def test_read_skips_whitespace_only(self) -> None:
        """read(str) skips spaces/tabs but not newlines before matching."""
        p = ASMParser("  +")
        p.read("+")
        assert p.pos == 3


class TestTryRead:
    def test_try_read_success(self) -> None:
        p = ASMParser("  (")
        result = p.try_read("(")
        assert result == "("
        assert p.pos == 3

    def test_try_read_failure_restores_pos(self) -> None:
        p = ASMParser("  )")
        result = p.try_read("(")
        assert result is None
        assert p.pos == 0

    def test_try_read_multi_char_failure(self) -> None:
        p = ASMParser("  -x")
        result = p.try_read("->")
        assert result is None
        assert p.pos == 0


class TestTokenMethods:
    def test_parse_token(self) -> None:
        p = ASMParser("  hello world")
        token = p.parse_token(_IDENT)
        assert token == "hello"

    def test_parse_token_at_end(self) -> None:
        p = ASMParser("   ")
        token = p.parse_token(_IDENT)
        assert token is None

    def test_expect_token(self) -> None:
        p = ASMParser("  hello")
        token = p.expect_token(_IDENT, "identifier")
        assert token == "hello"

    def test_expect_token_fails(self) -> None:
        p = ASMParser("   ")
        with pytest.raises(RuntimeError, match="Expected identifier"):
            p.expect_token(_IDENT, "identifier")

    def test_parse_identifier(self) -> None:
        p = ASMParser("  foo_bar")
        ident = p._parse_identifier()
        assert ident == "foo_bar"

    def test_parse_string_literal(self) -> None:
        p = ASMParser('  "hello world"')
        s = p._parse_string_literal()
        assert s == "hello world"

    def test_skip_line(self) -> None:
        p = ASMParser("hello\nworld")
        p._skip_line()
        assert p.text[p.pos :].startswith("world")


class TestReadWithGrammarClass:
    """Test read/try_read with grammar classes that have a read classmethod."""

    def test_read_grammar_class(self) -> None:
        class IdentGrammar:
            @classmethod
            def read(cls, parser: ASMParser) -> str:
                return parser.expect_token(_IDENT, "identifier")

        p = ASMParser("  hello")
        result = p.read(IdentGrammar)
        assert result == "hello"

    def test_try_read_grammar_class_success(self) -> None:
        class IdentGrammar:
            @classmethod
            def read(cls, parser: ASMParser) -> str:
                return parser.expect_token(_IDENT, "identifier")

        p = ASMParser("  hello")
        result = p.try_read(IdentGrammar)
        assert result == "hello"

    def test_try_read_grammar_class_failure(self) -> None:
        class FailGrammar:
            @classmethod
            def read(cls, parser: ASMParser) -> int:
                raise RuntimeError("no match")

        p = ASMParser("  abc")
        result = p.try_read(FailGrammar)
        assert result is None
        assert p.pos == 0


class TestExpect:
    def test_expect_string(self) -> None:
        p = ASMParser("hello")
        p._expect("hello")
        assert p.pos == 5

    def test_expect_fails(self) -> None:
        p = ASMParser("hello")
        with pytest.raises(RuntimeError):
            p._expect("world")
