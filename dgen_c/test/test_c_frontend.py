"""Tests for the C frontend: type resolution, end-to-end JIT, and scale."""

from __future__ import annotations

import random
import shutil
import time
import urllib.request
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

_SQLITE3_URL = (
    "https://raw.githubusercontent.com/mattn/go-sqlite3/master/sqlite3-binding.c"
)

# GCC extensions that survive preprocessing and that pycparser can't handle.
# glibc headers use __attribute__, __asm__, __restrict, __extension__,
# __builtin_va_list, and _FloatN types unconditionally — even with -std=c99.
# Defining these away is the standard pycparser approach.
_GCC_COMPAT_DEFINES = [
    "-D__attribute__(x)=",
    "-D__asm__(x)=",
    "-D__extension__=",
    "-D__restrict=",
    "-D__inline=inline",
    "-D__signed__=signed",
    "-D__volatile=volatile",
    "-D__const=const",
    "-D__builtin_va_list=void*",
    "-D__builtin_offsetof(t,f)=((long)(0))",
    "-D__builtin_va_start(a,b)=",
    "-D__builtin_va_end(a)=",
    "-D__builtin_va_arg(a,b)=0",
    "-D__builtin_va_copy(a,b)=",
    # glibc exposes _FloatN types — map them to standard types
    "-D_Float16=float",
    "-D_Float32=float",
    "-D_Float64=double",
    "-D_Float128=double",
    "-D_Float32x=double",
    "-D_Float64x=double",
    "-D_Float128x=double",
    "-D__CFLOAT32=float _Complex",
    "-D__CFLOAT64=double _Complex",
    "-D__CFLOAT128=double _Complex",
    "-D__CFLOAT32X=double _Complex",
    "-D__CFLOAT64X=double _Complex",
]


def run_c(source: str, *args: int) -> int:
    """Compile a C source string, JIT the first function, return the result."""
    ast = parse_c_string(source)
    module, _stats = lower(ast)
    module = _c_compiler.run(module)
    exe = llvm_compile(module)
    result = exe.run(*args)
    return result.to_json()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sqlite3_ast(tmp_path_factory: pytest.TempPathFactory) -> object:
    """Download sqlite3.c, preprocess with gcc -E, parse with pycparser.

    gcc -E expands all macros and includes using real system headers.
    The _GCC_COMPAT_DEFINES stub out GCC extensions that pycparser
    can't handle (__attribute__, __asm__, _FloatN, etc.).
    """
    from pycparser import parse_file

    if not shutil.which("gcc"):
        pytest.skip("gcc not available")

    tmp = tmp_path_factory.mktemp("sqlite3")
    raw = tmp / "sqlite3.c"

    try:
        resp = urllib.request.urlopen(_SQLITE3_URL, timeout=30)
        raw.write_bytes(resp.read())
    except Exception as exc:
        pytest.skip(f"Cannot download sqlite3.c: {exc}")

    if raw.stat().st_size < 1_000_000:
        pytest.skip("Downloaded file too small — likely truncated")

    try:
        return parse_file(
            str(raw),
            use_cpp=True,
            cpp_path="gcc",
            cpp_args=["-E", *_GCC_COMPAT_DEFINES],
        )
    except Exception as exc:
        pytest.skip(f"Cannot parse sqlite3.c: {exc}")


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
        assert run_c("int f(int a, int b) { return a * b + a / b; }", 20, 4) == 85

    def test_chained_comparisons(self) -> None:
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
# Scale: generated C code
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
                    f"  for (x{v} = 0; x{v} < {rng.randint(5, 20)}; x{v}++)"
                    f" x{v} = x{v} + 1;"
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
    """Verify the frontend handles large input."""

    def test_1500_functions(self) -> None:
        source = _generate_sqlite_scale_c(1500)
        ast = parse_c_string(source)
        module, stats = lower(ast)
        assert stats.functions == 1500
        assert stats.skipped_stmts == 0
        assert len(module.functions) == 1500

    def test_5000_functions(self) -> None:
        source = _generate_sqlite_scale_c(5000)
        t0 = time.perf_counter()
        ast = parse_c_string(source)
        module, stats = lower(ast)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Pipeline took {elapsed:.1f}s"
        assert stats.functions == 5000
        assert stats.skipped_stmts == 0
        assert len(module.functions) == 5000


# ---------------------------------------------------------------------------
# sqlite3.c: download, preprocess with gcc -E, parse, and lower
# ---------------------------------------------------------------------------


class TestSqlite3:
    """Parse and lower the actual sqlite3.c amalgamation."""

    def test_parse_sqlite3(self, sqlite3_ast: object) -> None:
        """pycparser can parse sqlite3.c (265K lines, ~9MB)."""
        assert len(sqlite3_ast.ext) > 1000

    def test_lower_sqlite3(self, sqlite3_ast: object) -> None:
        """Lower sqlite3.c to dgen IR with no skipped statements."""
        t0 = time.perf_counter()
        module, stats = lower(sqlite3_ast)
        elapsed = time.perf_counter() - t0
        assert stats.functions > 500, f"Only {stats.functions} functions"
        assert stats.skipped_stmts == 0
        assert len(module.functions) > 500
        assert elapsed < 120, f"Lowering took {elapsed:.1f}s"
