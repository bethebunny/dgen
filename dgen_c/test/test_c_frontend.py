"""Tests for the C frontend: type resolution, end-to-end JIT, and scale."""

from __future__ import annotations

import random
import time
from pathlib import Path


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
    """Compile C functions through the full pipeline and verify JIT output."""

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
        assert run_c("int f(int a, int b) { return a & b; }", 0xCC, 0xAA) == 0x88

    def test_bitwise_or(self) -> None:
        assert run_c("int f(int a, int b) { return a | b; }", 0xCC, 0xAA) == 0xEE

    def test_bitwise_xor(self) -> None:
        assert run_c("int f(int a, int b) { return a ^ b; }", 0xCC, 0xAA) == 0x66

    def test_comparison_lt(self) -> None:
        assert run_c("int f(int a, int b) { return a < b; }", 3, 5) == 1
        assert run_c("int f(int a, int b) { return a < b; }", 5, 3) == 0

    def test_comparison_eq(self) -> None:
        assert run_c("int f(int a, int b) { return a == b; }", 5, 5) == 1
        assert run_c("int f(int a, int b) { return a == b; }", 5, 3) == 0

    def test_comparison_ge(self) -> None:
        assert run_c("int f(int a, int b) { return a >= b; }", 5, 5) == 1
        assert run_c("int f(int a, int b) { return a >= b; }", 3, 5) == 0

    def test_comparison_ne(self) -> None:
        assert run_c("int f(int a, int b) { return a != b; }", 3, 5) == 1
        assert run_c("int f(int a, int b) { return a != b; }", 5, 5) == 0

    def test_multiple_ops(self) -> None:
        # (a * b) + (a / b) with a=20, b=4 => 80 + 5 = 85
        assert run_c("int f(int a, int b) { return a * b + a / b; }", 20, 4) == 85

    def test_chained_comparisons(self) -> None:
        # (a < b) + (a > b) — exactly one is 1
        assert run_c("int f(int a, int b) { return (a < b) + (a > b); }", 3, 5) == 1
        assert run_c("int f(int a, int b) { return (a < b) + (a > b); }", 5, 5) == 0


# ---------------------------------------------------------------------------
# Lowering (verify C constructs lower without crashing)
# ---------------------------------------------------------------------------


class TestLowering:
    def test_if_else(self) -> None:
        ast = parse_c_string("""
            int abs(int x) {
                if (x < 0) return -x;
                else return x;
            }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_while_loop(self) -> None:
        ast = parse_c_string("""
            int sum(int n) {
                int s = 0; int i = 0;
                while (i < n) { s = s + i; i = i + 1; }
                return s;
            }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_for_loop(self) -> None:
        ast = parse_c_string("""
            int factorial(int n) {
                int r = 1; int i;
                for (i = 1; i <= n; i++) { r = r * i; }
                return r;
            }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_typedef(self) -> None:
        ast = parse_c_string("""
            typedef unsigned int uint;
            uint f(uint x) { return x * 2; }
        """)
        _, stats = lower(ast)
        assert stats.typedefs == 1

    def test_struct(self) -> None:
        ast = parse_c_string("""
            struct P { int x; int y; };
            int f(struct P *p) { return p->x; }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_enum(self) -> None:
        ast = parse_c_string("""
            enum Color { RED, GREEN, BLUE };
            int f() { return GREEN; }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_ternary(self) -> None:
        ast = parse_c_string("int f(int a, int b) { return a > b ? a : b; }")
        module, _ = lower(ast)
        assert len(module.functions) == 1

    def test_function_call(self) -> None:
        ast = parse_c_string("""
            int add(int a, int b) { return a + b; }
            int f() { return add(1, 2); }
        """)
        module, _ = lower(ast)
        assert len(module.functions) == 2

    def test_goto_label(self) -> None:
        ast = parse_c_string("""
            int f(int x) {
                goto done;
                x = x + 1;
                done:
                return x;
            }
        """)
        module, stats = lower(ast)
        assert len(module.functions) == 1
        assert stats.skipped_stmts == 0


# ---------------------------------------------------------------------------
# Scale: generate sqlite3-scale C code and verify parse+lower
# ---------------------------------------------------------------------------


def _generate_sqlite_scale_c(n_functions: int, seed: int = 42) -> str:
    """Generate a C source string with realistic function complexity."""
    rng = random.Random(seed)
    lines: list[str] = []
    lines.append("typedef unsigned int u32;")
    lines.append("typedef int i32;")
    lines.append("")

    for i in range(50):
        lines.append(f"struct S{i} {{")
        for j in range(rng.randint(2, 8)):
            lines.append(f"  int f{j};")
        lines.append("};")
        lines.append("")

    for i in range(20):
        lines.append(f"enum E{i} {{")
        for j in range(rng.randint(3, 10)):
            lines.append(f"  E{i}_V{j} = {j},")
        lines.append("};")
        lines.append("")

    for i in range(n_functions):
        ret = rng.choice(["int", "void", "u32"])
        n_params = rng.randint(0, 4)
        params = [f"int a{j}" for j in range(n_params)]
        param_str = ", ".join(params) if params else "void"

        lines.append(f"{ret} func_{i}({param_str}) {{")

        n_locals = rng.randint(1, 5)
        for j in range(n_locals):
            lines.append(f"  int x{j} = 0;")

        for _ in range(rng.randint(3, 20)):
            v = rng.randint(0, n_locals - 1)
            kind = rng.choice(["assign", "if", "while", "for", "call"])
            if kind == "assign":
                op = rng.choice(["+", "-", "*", "&", "|"])
                lines.append(f"  x{v} = x{v} {op} {rng.randint(1, 100)};")
            elif kind == "if":
                lines.append(f"  if (x{v} > {rng.randint(0, 50)}) x{v} = x{v} + 1;")
            elif kind == "while":
                lines.append(f"  while (x{v} < {rng.randint(10, 50)}) x{v} = x{v} + 1;")
            elif kind == "for":
                lines.append(
                    f"  for (x{v} = 0; x{v} < {rng.randint(5, 20)}; x{v}++) x{v} = x{v} + 1;"
                )
            elif kind == "call" and i > 0:
                target = rng.randint(0, i - 1)
                lines.append(f"  func_{target}();")

        if ret == "void":
            lines.append("  return;")
        else:
            lines.append("  return x0;")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


class TestScale:
    """Verify the frontend handles sqlite3-scale input."""

    def test_1500_functions(self) -> None:
        """Parse and lower 1500 functions with no skipped statements."""
        source = _generate_sqlite_scale_c(1500)
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert stats.functions == 1500
        assert stats.skipped_stmts == 0
        assert stats.skipped_exprs == 0
        assert len(module.functions) == 1500

    def test_5000_functions(self) -> None:
        """Parse and lower 5000 functions (sqlite3-scale) under 60s."""
        source = _generate_sqlite_scale_c(5000)
        t0 = time.perf_counter()
        ast = parse_c_string(source)
        module, stats = lower(ast)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Pipeline took {elapsed:.1f}s"
        assert stats.functions == 5000
        assert stats.skipped_stmts == 0
        assert len(module.functions) == 5000
