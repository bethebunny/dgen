"""Tests for ASMParser core tokenizer with read/try_read protocol."""

from __future__ import annotations

import re

import pytest

import toy.dialects.toy  # noqa: F401 — registers dialect

from dgen.asm.parser import ASMParser, ParseError, parse
from dgen.dialects.function import FunctionOp
from dgen.graph import transitive_dependencies
from dgen.module import asm_with_imports
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
            | %f : function.Function<[], toy.Tensor<[2, 3], number.Float64>> = function.function<toy.Tensor<[2, 3], number.Float64>>() body():
        """)
        with pytest.raises(RuntimeError, match=r"bare literal.*Shape.*Shape<\.\.\.>"):
            parse(ir)

    def test_unimported_dialect_rejected(self) -> None:
        """Referencing a dialect not declared via `import` should fail."""
        ir = strip_prefix("""
            | import function
            | import toy
            |
            | %f : function.Function<[], ()> = function.function<Nil>() body():
            |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        """)
        with pytest.raises(Exception):
            parse(ir)


class TestParseValue:
    def test_parses_single_statement(self) -> None:
        value = parse(
            strip_prefix("""
            | import function
            | import index
            |
            | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
            |     %r : index.Index = 42
        """)
        )
        assert isinstance(value, FunctionOp)

    def test_returns_last_statement_with_earlier_as_deps(self) -> None:
        """Multiple statements: later statements can reference earlier ones by name."""
        value = parse(
            strip_prefix("""
            | import index
            | import algebra
            |
            | %a : index.Index = 3
            | %b : index.Index = 4
            | %c : index.Index = algebra.add(%a, %b)
        """)
        )
        assert value.name == "c"
        # %a and %b are reachable as transitive deps of %c.
        dep_names = {v.name for v in transitive_dependencies(value)}
        assert {"a", "b", "c"} <= dep_names

    def test_interleaved_imports_and_statements(self) -> None:
        """Imports may appear between statements, not only at the top."""
        value = parse(
            strip_prefix("""
            | import index
            |
            | %a : index.Index = 3
            |
            | import algebra
            |
            | %b : index.Index = algebra.add(%a, %a)
        """)
        )
        assert value.name == "b"

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ParseError):
            parse("")

    def test_roundtrips_with_asm_with_imports(self) -> None:
        """parse + asm_with_imports → stable formatted text."""
        text = strip_prefix("""
            | import function
            | import index
            | import algebra
            |
            | %f : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
            |     %r : index.Index = algebra.add(%a, %b)
        """)
        first = "\n".join(asm_with_imports(parse(text)))
        second = "\n".join(asm_with_imports(parse(first)))
        assert first == second

    def test_roundtrips_multi_statement(self) -> None:
        """Multi-statement IR survives a parse → asm → parse round-trip."""
        text = strip_prefix("""
            | import index
            | import algebra
            |
            | %a : index.Index = 3
            | %b : index.Index = 4
            | %c : index.Index = algebra.add(%a, %b)
        """)
        first = "\n".join(asm_with_imports(parse(text)))
        second = "\n".join(asm_with_imports(parse(first)))
        assert first == second


class TestAsmWithImports:
    def test_collects_dialects_from_nested_blocks(self) -> None:
        """asm_with_imports walks into block bodies to find every dialect used."""
        # `algebra.add` is only reachable inside the function body.
        func = parse(
            strip_prefix("""
            | import function
            | import index
            | import algebra
            |
            | %f : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
            |     %r : index.Index = algebra.add(%x, %x)
        """)
        )
        imports = [
            line for line in asm_with_imports(func) if line.startswith("import ")
        ]
        assert "import algebra" in imports

    def test_emits_transitive_dep_statements(self) -> None:
        """Root's transitive dep ops show up as their own SSA statements."""
        root = parse(
            strip_prefix("""
            | import index
            | import algebra
            |
            | %a : index.Index = 3
            | %b : index.Index = 4
            | %c : index.Index = algebra.add(%a, %b)
        """)
        )
        lines = list(asm_with_imports(root))
        # Each of a, b, c is emitted as a top-level statement.
        defines = [line for line in lines if line.startswith("%")]
        names_defined = {line.split()[0] for line in defines}
        assert {"%a", "%b", "%c"} <= names_defined
