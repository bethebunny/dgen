"""Tests for the C frontend: parsing, type resolution, lowering, and JIT execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from dgen.codegen import compile as llvm_compile
from dgen.compiler import Compiler, IdentityPass
from dgen_c.dialects import c_int
from dgen_c.dialects.c import CFloat, CInt, CPtr, CStruct, CVoid
from dgen_c.parser.c_parser import parse_c_string
from dgen_c.parser.lowering import lower
from dgen_c.parser.type_resolver import TypeResolver
from dgen_c.passes.c_to_llvm import CToLLVM

TESTDATA = Path(__file__).parent / "testdata"

_c_compiler = Compiler([CToLLVM()], IdentityPass())


def run_c(source: str, *args: int) -> int:
    """Compile a C source string, JIT the first function, return the result."""
    ast = parse_c_string(source)
    module, _stats = lower(ast)
    module = _c_compiler.run(module)
    exe = llvm_compile(module)
    result = exe.run(*args)
    return result.to_json()


# ---------------------------------------------------------------------------
# Type resolver
# ---------------------------------------------------------------------------


class TestTypeResolver:
    def test_basic_int_types(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["int"]), CInt)

    def test_unsigned_long(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["unsigned", "long"]), CInt)

    def test_void(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["void"]), CVoid)

    def test_double(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["double"]), CFloat)

    def test_typedef(self) -> None:
        r = TypeResolver()
        r.register_typedef("myint", c_int(32))
        assert isinstance(r._resolve_identifier_type(["myint"]), CInt)

    def test_pointer(self) -> None:
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
        assert isinstance(r.resolve(node), CPtr)

    def test_struct(self) -> None:
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
        assert isinstance(r.resolve(node), CStruct)

    def test_enum_constants(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        r.resolve(
            c_ast.Enum(
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
        )
        assert r.enum_constants == {"RED": 0, "GREEN": 1, "BLUE": 5}

    def test_const_expr_hex(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        assert r._eval_const_expr(c_ast.Constant(type="int", value="0xFF")) == 255

    def test_const_expr_binop(self) -> None:
        from pycparser import c_ast

        r = TypeResolver()
        node = c_ast.BinaryOp(
            op="+",
            left=c_ast.Constant(type="int", value="3"),
            right=c_ast.Constant(type="int", value="4"),
        )
        assert r._eval_const_expr(node) == 7


# ---------------------------------------------------------------------------
# End-to-end: C source → JIT → verify return value
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_add(self) -> None:
        assert run_c("int f(int a, int b) { return a + b; }", 3, 4) == 7

    def test_subtract(self) -> None:
        assert run_c("int f(int a, int b) { return a - b; }", 10, 3) == 7

    def test_multiply(self) -> None:
        assert run_c("int f(int a, int b) { return a * b; }", 6, 7) == 42

    def test_divide(self) -> None:
        assert run_c("int f(int a, int b) { return a / b; }", 20, 4) == 5

    def test_negate(self) -> None:
        assert run_c("int f(int x) { return -x; }", 5) == -5

    def test_constant(self) -> None:
        assert run_c("int f() { return 42; }") == 42

    def test_nested_arithmetic(self) -> None:
        assert run_c("int f(int a, int b) { return (a + b) * (a - b); }", 7, 3) == 40

    def test_bitwise_and(self) -> None:
        assert run_c("int f(int a, int b) { return a & b; }", 0b1100, 0b1010) == 0b1000

    def test_bitwise_or(self) -> None:
        assert run_c("int f(int a, int b) { return a | b; }", 0b1100, 0b1010) == 0b1110

    def test_bitwise_xor(self) -> None:
        assert run_c("int f(int a, int b) { return a ^ b; }", 0b1100, 0b1010) == 0b0110

    def test_comparison_lt(self) -> None:
        assert run_c("int f(int a, int b) { return a < b; }", 3, 5) == 1
        assert run_c("int f(int a, int b) { return a < b; }", 5, 3) == 0

    def test_comparison_eq(self) -> None:
        assert run_c("int f(int a, int b) { return a == b; }", 5, 5) == 1
        assert run_c("int f(int a, int b) { return a == b; }", 5, 3) == 0


# ---------------------------------------------------------------------------
# Lowering (smoke tests — verify parse+lower doesn't crash)
# ---------------------------------------------------------------------------


class TestLowering:
    def test_if_else(self) -> None:
        ast = parse_c_string("""
            int abs(int x) {
                if (x < 0) return -x;
                else return x;
            }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_while_loop(self) -> None:
        ast = parse_c_string("""
            int sum(int n) {
                int s = 0; int i = 0;
                while (i < n) { s = s + i; i = i + 1; }
                return s;
            }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_for_loop(self) -> None:
        ast = parse_c_string("""
            int factorial(int n) {
                int r = 1; int i;
                for (i = 1; i <= n; i++) { r = r * i; }
                return r;
            }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_typedef(self) -> None:
        ast = parse_c_string("""
            typedef unsigned int uint;
            uint f(uint x) { return x * 2; }
        """)
        module, stats = lower(ast)
        assert stats.typedefs == 1

    def test_struct(self) -> None:
        ast = parse_c_string("""
            struct P { int x; int y; };
            int f(struct P *p) { return p->x; }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_enum(self) -> None:
        ast = parse_c_string("""
            enum Color { RED, GREEN, BLUE };
            int f() { return GREEN; }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_ternary(self) -> None:
        ast = parse_c_string("int f(int a, int b) { return a > b ? a : b; }")
        module, stats = lower(ast)
        assert len(module.functions) == 1

    def test_function_call(self) -> None:
        ast = parse_c_string("""
            int add(int a, int b) { return a + b; }
            int f() { return add(1, 2); }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 2


# ---------------------------------------------------------------------------
# File-based tests
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
