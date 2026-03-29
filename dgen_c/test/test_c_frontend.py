"""Tests for the C frontend: parsing, lowering, and pass pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from dgen_c.parser.c_parser import parse_c_string
from dgen_c.parser.lowering import lower
from dgen_c.parser.type_resolver import TypeResolver
from dgen_c.dialects import c_int
from dgen_c.dialects.c import (
    CFloat,
    CInt,
    CPtr,
    CStruct,
    CVoid,
)

TESTDATA = Path(__file__).parent / "testdata"


# ---------------------------------------------------------------------------
# Type resolver
# ---------------------------------------------------------------------------


class TestTypeResolver:
    def test_basic_int_types(self) -> None:
        r = TypeResolver()
        ty = r._resolve_identifier_type(["int"])
        assert isinstance(ty, CInt)

    def test_unsigned_long(self) -> None:
        r = TypeResolver()
        ty = r._resolve_identifier_type(["unsigned", "long"])
        assert isinstance(ty, CInt)

    def test_void(self) -> None:
        r = TypeResolver()
        ty = r._resolve_identifier_type(["void"])
        assert isinstance(ty, CVoid)

    def test_double(self) -> None:
        r = TypeResolver()
        ty = r._resolve_identifier_type(["double"])
        assert isinstance(ty, CFloat)

    def test_typedef_resolution(self) -> None:
        r = TypeResolver()
        r.register_typedef("myint", c_int(32))
        ty = r._resolve_identifier_type(["myint"])
        assert isinstance(ty, CInt)

    def test_pointer_type(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.PtrDecl(
            quals=[],
            type=c_ast.TypeDecl(
                declname="x",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(names=["int"]),
            ),
        )
        ty = r.resolve(node)
        assert isinstance(ty, CPtr)

    def test_struct_resolution(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.Struct(
            name="Point",
            decls=[
                c_ast.Decl(
                    name="x",
                    quals=[],
                    align=None,
                    storage=[],
                    funcspec=[],
                    type=c_ast.TypeDecl(
                        declname="x",
                        quals=[],
                        align=None,
                        type=c_ast.IdentifierType(names=["double"]),
                    ),
                    init=None,
                    bitsize=None,
                ),
            ],
        )
        ty = r.resolve(node)
        assert isinstance(ty, CStruct)

    def test_enum_constants(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.Enum(
            name="Color",
            values=c_ast.EnumeratorList(
                enumerators=[
                    c_ast.Enumerator(name="RED", value=None),
                    c_ast.Enumerator(name="GREEN", value=None),
                    c_ast.Enumerator(
                        name="BLUE",
                        value=c_ast.Constant(type="int", value="5"),
                    ),
                ]
            ),
        )
        r.resolve(node)
        assert r.enum_constants["RED"] == 0
        assert r.enum_constants["GREEN"] == 1
        assert r.enum_constants["BLUE"] == 5

    def test_const_expr_eval(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.BinaryOp(
            op="+",
            left=c_ast.Constant(type="int", value="3"),
            right=c_ast.Constant(type="int", value="4"),
        )
        assert r._eval_const_expr(node) == 7

    def test_hex_literal(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.Constant(type="int", value="0xFF")
        assert r._eval_const_expr(node) == 255


# ---------------------------------------------------------------------------
# Parsing and lowering
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parse_simple(self) -> None:
        ast = parse_c_string("int add(int a, int b) { return a + b; }")
        assert ast is not None
        assert len(ast.ext) == 1

    def test_parse_multiple_functions(self) -> None:
        source = """
        int foo(int x) { return x + 1; }
        int bar(int y) { return y * 2; }
        """
        ast = parse_c_string(source)
        assert len(ast.ext) == 2


class TestLowering:
    def test_lower_add(self) -> None:
        source = "int add(int a, int b) { return a + b; }"
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1
        assert module.functions[0].name == "add"
        assert len(module.functions[0].body.args) == 2
        assert stats.functions == 1

    def test_lower_with_locals(self) -> None:
        source = """
        int square(int x) {
            int result = x * x;
            return result;
        }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1
        assert stats.expressions > 0

    def test_lower_if_else(self) -> None:
        source = """
        int abs(int x) {
            if (x < 0)
                return -x;
            else
                return x;
        }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_while_loop(self) -> None:
        source = """
        int sum_to_n(int n) {
            int s = 0;
            int i = 0;
            while (i < n) {
                s = s + i;
                i = i + 1;
            }
            return s;
        }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_for_loop(self) -> None:
        source = """
        int factorial(int n) {
            int r = 1;
            int i;
            for (i = 1; i <= n; i++) {
                r = r * i;
            }
            return r;
        }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_typedef(self) -> None:
        source = """
        typedef unsigned int uint;
        uint double_it(uint x) { return x * 2; }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert stats.typedefs == 1
        assert len(module.functions) == 1

    def test_lower_struct(self) -> None:
        source = """
        struct Point { double x; double y; };
        double get_x(struct Point *p) { return p->x; }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_function_call(self) -> None:
        source = """
        int add(int a, int b) { return a + b; }
        int main() { return add(1, 2); }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 2

    def test_lower_enum(self) -> None:
        source = """
        enum Color { RED, GREEN, BLUE };
        int get_color() { return GREEN; }
        """
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_void_function(self) -> None:
        source = "void noop() { return; }"
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_ternary(self) -> None:
        source = "int max(int a, int b) { return a > b ? a : b; }"
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_lower_cast(self) -> None:
        source = "double to_double(int x) { return (double)x; }"
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert len(module.functions) == 1


# ---------------------------------------------------------------------------
# Benchmark tests (file-based)
# ---------------------------------------------------------------------------


class TestFiles:
    def test_simple_file(self) -> None:
        ast = parse_c_string((TESTDATA / "simple.c").read_text())
        module, stats = lower(ast)
        assert stats.functions == 3
        assert stats.skipped_stmts == 0

    def test_medium_file(self) -> None:
        ast = parse_c_string((TESTDATA / "medium.c").read_text())
        module, stats = lower(ast)
        assert stats.functions == 5
        assert stats.skipped_stmts == 0

    @pytest.mark.skipif(
        not (TESTDATA / "large.c").exists(), reason="large.c not generated"
    )
    def test_large_file(self) -> None:
        ast = parse_c_string((TESTDATA / "large.c").read_text())
        module, stats = lower(ast)
        assert stats.functions == 1500
        assert stats.skipped_stmts == 0
