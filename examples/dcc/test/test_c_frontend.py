"""Tests for the dcc C frontend."""

from __future__ import annotations

from pathlib import Path

import pytest
from pycparser import c_ast

import dgen
from dgen import Dialect
from dgen.block import BlockArgument
from dgen.dialects import function, index
from dgen.dialects.builtin import Nil
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger
from dgen.ir.traversal import transitive_dependencies
from dgen.builtins import pack

from click.testing import CliRunner

from dcc.cli import c_compiler, main, run_file
from dcc.dialects import c, c_double, c_float, c_int, c_ptr, c_void
from dcc.parser.c_parser import parse_c_string
from dcc.parser.lowering import LoweringError, Parser, Scope, lower
from dcc.parser.type_resolver import TypeResolver, TypeResolverError

TESTDATA_DIR = Path(__file__).parent / "testdata"

# Make dcc dialect discoverable.
Dialect.paths.append(Path(__file__).parent.parent / "dialects")


def run_c(source: str, *args: int) -> int:
    """Compile a C source string, JIT the last function, return the result."""
    ir = lower(parse_c_string(source))
    exe = c_compiler.compile(ir)
    result = exe.run(*args)
    return result.to_json()


class TestDialect:
    """Brick 1: Verify the C dialect loads and all ops/types are available."""

    def test_dialect_loads(self) -> None:
        assert c.c.name == "c"

    def test_lvalue_ops_exist(self) -> None:
        assert c.LvalueVarOp is not None
        assert c.LvalueDerefOp is not None
        assert c.LvalueSubscriptOp is not None
        assert c.LvalueMemberOp is not None
        assert c.LvalueArrowOp is not None
        assert c.LvalueToRvalueOp is not None
        assert c.AddressOfOp is not None
        assert c.AssignOp is not None
        assert c.CompoundAssignOp is not None
        assert c.PreIncrementOp is not None
        assert c.PostIncrementOp is not None
        assert c.PreDecrementOp is not None
        assert c.PostDecrementOp is not None

    def test_conversion_ops_exist(self) -> None:
        assert c.IntegerPromoteOp is not None
        assert c.ArithmeticConvertOp is not None
        assert c.ArrayDecayOp is not None
        assert c.FunctionDecayOp is not None
        assert c.NullToPointerOp is not None
        assert c.ScalarToBoolOp is not None

    def test_c_types_exist(self) -> None:
        assert c.Struct is not None
        assert c.StructField is not None
        assert c.Union is not None
        assert c.Enum is not None
        assert c.CFunctionType is not None

    def test_arithmetic_and_control_ops_exist(self) -> None:
        assert c.CReturnOp is not None
        assert c.CSizeofOp is not None
        assert c.ModuloOp is not None
        assert c.ShiftLeftOp is not None
        assert c.ShiftRightOp is not None
        assert c.LogicalNotOp is not None
        assert c.CommaOp is not None

    def test_type_constructors(self) -> None:
        signed = c_int(32)
        assert isinstance(signed, SignedInteger)

        unsigned = c_int(16, signed=False)
        assert isinstance(unsigned, UnsignedInteger)

        assert isinstance(c_float(), Float64)
        assert isinstance(c_double(), Float64)
        assert isinstance(c_void(), Nil)
        assert isinstance(c_ptr(c_int()), Reference)


class TestPipeline:
    """Brick 2: Verify the compiler pipeline can compile and run IR."""

    def test_pipeline_smoke(self) -> None:
        """Manually construct int f() { return 42; } and run through pipeline."""
        ret_type = SignedInteger(bits=index.Index().constant(64))
        body_result = ret_type.constant(42)
        func = function.FunctionOp(
            name="f",
            result_type=ret_type,
            body=dgen.Block(result=body_result, args=[]),
            type=FunctionType(
                arguments=pack([]),
                result_type=ret_type,
            ),
        )
        exe = c_compiler.compile(func)
        assert exe.run().to_json() == 42

    def test_pipeline_with_parameter(self) -> None:
        """int f(int x) { return x; } with x=7."""
        ret_type = SignedInteger(bits=index.Index().constant(64))
        arg = BlockArgument(name="x", type=ret_type)
        func = function.FunctionOp(
            name="f",
            result_type=ret_type,
            body=dgen.Block(result=arg, args=[arg]),
            type=FunctionType(
                arguments=pack([ret_type]),
                result_type=ret_type,
            ),
        )
        exe = c_compiler.compile(func)
        assert exe.run(7).to_json() == 7


class TestTypeResolver:
    """Brick 3: Verify type resolution from pycparser AST nodes."""

    def test_int_types(self) -> None:

        r = TypeResolver()
        signed = r.resolve(
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["int"]))
        )
        assert isinstance(signed, SignedInteger)

        unsigned = r.resolve(
            c_ast.TypeDecl(
                None, None, None, c_ast.IdentifierType(names=["unsigned", "int"])
            )
        )
        assert isinstance(unsigned, UnsignedInteger)

    def test_float_and_double(self) -> None:

        r = TypeResolver()
        f = r.resolve(
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["float"]))
        )
        assert isinstance(f, Float64)

        d = r.resolve(
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["double"]))
        )
        assert isinstance(d, Float64)

    def test_void(self) -> None:

        r = TypeResolver()
        v = r.resolve(
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["void"]))
        )
        assert isinstance(v, Nil)

    def test_pointer(self) -> None:

        r = TypeResolver()
        p = r.resolve(
            c_ast.PtrDecl(
                None,
                c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["int"])),
            )
        )
        assert isinstance(p, Reference)

    def test_struct_with_fields(self) -> None:

        r = TypeResolver()
        decls = [
            c_ast.Decl(
                "x",
                None,
                None,
                None,
                None,
                c_ast.TypeDecl("x", None, None, c_ast.IdentifierType(names=["int"])),
                None,
                None,
            ),
            c_ast.Decl(
                "y",
                None,
                None,
                None,
                None,
                c_ast.TypeDecl("y", None, None, c_ast.IdentifierType(names=["int"])),
                None,
                None,
            ),
        ]
        node = c_ast.Struct("Point", decls)
        t = r.resolve(node)
        assert isinstance(t, c.Struct)
        # Verify struct is cached and forward-referenceable.
        assert r._resolve_struct(c_ast.Struct("Point", None)) is t

    def test_anonymous_structs_are_unique(self) -> None:

        r = TypeResolver()
        decl_x = [
            c_ast.Decl(
                "x",
                None,
                None,
                None,
                None,
                c_ast.TypeDecl("x", None, None, c_ast.IdentifierType(names=["int"])),
                None,
                None,
            )
        ]
        decl_y = [
            c_ast.Decl(
                "y",
                None,
                None,
                None,
                None,
                c_ast.TypeDecl("y", None, None, c_ast.IdentifierType(names=["int"])),
                None,
                None,
            )
        ]
        s1 = r.resolve(c_ast.Struct(None, decl_x))
        s2 = r.resolve(c_ast.Struct(None, decl_y))
        assert isinstance(s1, c.Struct)
        assert isinstance(s2, c.Struct)
        assert s1 is not s2

    def test_enum_constants(self) -> None:

        r = TypeResolver()
        enumerators = c_ast.EnumeratorList(
            [
                c_ast.Enumerator("A", None),
                c_ast.Enumerator("B", None),
                c_ast.Enumerator("C", c_ast.Constant("int", "10")),
                c_ast.Enumerator("D", None),
            ]
        )
        r.resolve(c_ast.Enum("Color", enumerators))
        assert r.enum_constants["A"] == 0
        assert r.enum_constants["B"] == 1
        assert r.enum_constants["C"] == 10
        assert r.enum_constants["D"] == 11

    def test_unknown_type_raises(self) -> None:

        r = TypeResolver()
        with pytest.raises(TypeResolverError, match="unknown type"):
            r.resolve(
                c_ast.TypeDecl(
                    None, None, None, c_ast.IdentifierType(names=["mystery_t"])
                )
            )

    def test_f_void_has_no_params(self) -> None:
        """C11: f(void) should resolve to a function with zero parameters."""

        r = TypeResolver()
        # int f(void)
        func_decl = c_ast.FuncDecl(
            c_ast.ParamList(
                [
                    c_ast.Typename(
                        None,
                        None,
                        None,
                        c_ast.TypeDecl(
                            None, None, None, c_ast.IdentifierType(names=["void"])
                        ),
                    )
                ]
            ),
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["int"])),
        )
        t = r.resolve(func_decl)
        assert isinstance(t, c.CFunctionType)

    def test_parse_c_string(self) -> None:
        ast = parse_c_string("int f(int x) { return x; }")
        assert ast is not None
        assert len(ast.ext) == 1


class TestEndToEnd:
    """Brick 4+: End-to-end tests via run_c()."""

    def test_return_constant(self) -> None:
        assert run_c("int f() { return 42; }") == 42

    def test_return_zero(self) -> None:
        assert run_c("int f() { return 0; }") == 0

    def test_return_negative(self) -> None:
        assert run_c("int f() { return -1; }") == -1

    def test_integer_literal_suffixes(self) -> None:
        """Integer literal suffixes (u, l, ll, ul, ull) now parse."""
        assert run_c("int f() { return 42U; }") == 42
        assert run_c("int f() { return 42L; }") == 42
        assert run_c("int f() { return 42LL; }") == 42
        assert run_c("int f() { return 42UL; }") == 42
        assert run_c("int f() { return 42ULL; }") == 42

    def test_hex_and_octal_literals(self) -> None:
        """Hex and octal bases parse (pre-existing but re-verified)."""
        assert run_c("int f() { return 0xFF; }") == 255
        assert run_c("int f() { return 0x1FU; }") == 31
        assert run_c("int f() { return 0777; }") == 511

    def test_unknown_literal_type_raises(self) -> None:
        """Unknown pycparser literal types raise rather than silently returning 0."""
        bad = c_ast.Constant("string", '"hi"')
        with pytest.raises(LoweringError, match="unsupported literal type"):
            Parser()._constant(bad, Scope())

    def test_return_parameter(self) -> None:
        assert run_c("int f(int x) { return x; }", 7) == 7

    def test_two_parameters(self) -> None:
        assert run_c("int f(int a, int b) { return a; }", 3, 4) == 3

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

    def test_nested_arithmetic(self) -> None:
        assert run_c("int f(int a, int b) { return (a + b) * (a - b); }", 7, 3) == 40

    def test_comparison_less_than(self) -> None:
        assert run_c("int f(int a, int b) { return a < b; }", 3, 5) == 1
        assert run_c("int f(int a, int b) { return a < b; }", 5, 3) == 0

    def test_comparison_equal(self) -> None:
        assert run_c("int f(int a, int b) { return a == b; }", 5, 5) == 1
        assert run_c("int f(int a, int b) { return a == b; }", 5, 3) == 0

    def test_literal_and_param_types_match(self) -> None:
        """Integer literals and parameters should both be i32 (C int = 32-bit)."""
        assert run_c("int f(int x) { return x + 1; }", 5) == 6
        assert run_c("int f(int x) { return 10 - x; }", 3) == 7

    # --- Brick 5: Local variables (lvalue model) ---

    def test_local_variable(self) -> None:
        assert run_c("int f(int x) { int y = x + 1; return y; }", 5) == 6

    def test_local_mutation(self) -> None:
        assert run_c("int f(int x) { int y = x; y = y + 10; return y; }", 5) == 15

    def test_multiple_locals(self) -> None:
        assert (
            run_c(
                "int f(int a, int b) { int s = a + b; int d = a - b; return s * d; }",
                7,
                3,
            )
            == 40
        )

    def test_uninitialized_local(self) -> None:
        assert run_c("int f() { int x; x = 42; return x; }") == 42

    def test_reassign_multiple_times(self) -> None:
        assert run_c("int f() { int x = 1; x = 2; x = 3; return x; }") == 3

    def test_local_from_param_arithmetic(self) -> None:
        assert run_c("int f(int a, int b) { int r = a * b + 1; return r; }", 6, 7) == 43

    def test_nested_scope(self) -> None:
        assert run_c("int f() { int x = 10; { int y = 20; } return x; }") == 10

    def test_shadowed_variable(self) -> None:
        """Inner x doesn't affect outer x."""
        assert run_c("int f() { int x = 10; { int x = 99; } return x; }") == 10

    def test_shadow_then_reassign_outer(self) -> None:
        assert run_c("int f() { int x = 0; { int x = 99; } x = 2; return x; }") == 2

    def test_assign_to_parameter(self) -> None:
        """Parameters are mutable local variables in C."""
        assert run_c("int f(int x) { x = 10; return x; }", 5) == 10

    def test_read_then_reassign_same_variable(self) -> None:
        """Read x, then reassign x — read must see original value."""
        assert run_c("int f(int x) { int y = x; x = 10; return x + y; }", 5) == 15

    def test_multiple_reads_between_writes(self) -> None:
        """Multiple reads of x between writes must see correct values."""
        assert (
            run_c("int f() { int x = 1; int a = x; x = 2; int b = x; return a + b; }")
            == 3
        )

    # --- Brick 6: Control flow ---

    def test_if_true(self) -> None:
        assert run_c("int f(int x) { int r = 0; if (x) r = 1; return r; }", 1) == 1

    def test_if_false(self) -> None:
        assert run_c("int f(int x) { int r = 0; if (x) r = 1; return r; }", 0) == 0

    def test_if_else(self) -> None:
        assert (
            run_c("int f(int x) { int r; if (x) r = 1; else r = 2; return r; }", 1) == 1
        )
        assert (
            run_c("int f(int x) { int r; if (x) r = 1; else r = 2; return r; }", 0) == 2
        )

    def test_if_compound_body(self) -> None:
        assert (
            run_c("int f(int x) { int r = 0; if (x) { r = x + 10; } return r; }", 5)
            == 15
        )

    def test_while_loop(self) -> None:
        assert (
            run_c(
                "int f(int n) { int s = 0; int i = 0;"
                " while (i < n) { s = s + i; i = i + 1; } return s; }",
                5,
            )
            == 10
        )

    def test_while_zero_iterations(self) -> None:
        assert (
            run_c(
                "int f(int n) { int s = 0; int i = 0;"
                " while (i < n) { s = s + i; i = i + 1; } return s; }",
                0,
            )
            == 0
        )

    def test_for_loop(self) -> None:
        assert (
            run_c(
                "int f(int n) { int r = 1; int i;"
                " for (i = 1; i <= n; i = i + 1) r = r * i; return r; }",
                5,
            )
            == 120
        )

    def test_for_with_decl_init(self) -> None:
        assert (
            run_c(
                "int f(int n) { int s = 0;"
                " for (int i = 0; i < n; i = i + 1) s = s + i; return s; }",
                5,
            )
            == 10
        )

    def test_nested_if_while(self) -> None:
        assert (
            run_c(
                "int f(int n) { int s = 0; int i = 0;"
                " while (i < n) { if (i - (i/2)*2 == 0) s = s + i;"
                " i = i + 1; } return s; }",
                6,
            )
            == 6
        )

    def test_short_circuit_and(self) -> None:
        assert run_c("int f(int x, int y) { return x && y; }", 0, 42) == 0
        assert run_c("int f(int x, int y) { return x && y; }", 1, 42) == 1
        assert run_c("int f(int x, int y) { return x && y; }", 1, 0) == 0

    def test_short_circuit_or(self) -> None:
        assert run_c("int f(int x, int y) { return x || y; }", 0, 0) == 0
        assert run_c("int f(int x, int y) { return x || y; }", 0, 1) == 1
        assert run_c("int f(int x, int y) { return x || y; }", 1, 0) == 1

    def test_logical_not(self) -> None:
        """!x is 1 when x is zero, 0 otherwise."""
        assert run_c("int f(int x) { return !x; }", 0) == 1
        assert run_c("int f(int x) { return !x; }", 5) == 0
        assert run_c("int f(int x) { return !x; }", -1) == 0

    def test_logical_not_in_condition(self) -> None:
        """!x inside an if-condition."""
        assert run_c("int f(int x) { int r = 0; if (!x) r = 1; return r; }", 0) == 1
        assert run_c("int f(int x) { int r = 0; if (!x) r = 1; return r; }", 7) == 0

    def test_double_logical_not(self) -> None:
        """!!x normalises any non-zero to 1."""
        assert run_c("int f(int x) { return !!x; }", 0) == 0
        assert run_c("int f(int x) { return !!x; }", 42) == 1

    def test_read_inside_if_then_write(self) -> None:
        """Read inside if-body must fence subsequent write (diamond pattern)."""
        assert (
            run_c(
                "int f(int c) { int x = 1; if (c) { int y = x; } x = 2; return x; }",
                1,
            )
            == 2
        )

    def test_while_read_then_outer_write(self) -> None:
        """Reads inside while body must fence subsequent write."""
        assert (
            run_c(
                "int f() { int x = 10; int s = 0; int i = 0;"
                " while (i < 3) { s = s + x; i = i + 1; } x = 0; return s; }"
            )
            == 30
        )


class TestBreakContinue:
    """Brick 6.5: break and continue in loops."""

    def test_break_exits_loop(self) -> None:
        """while(1) { break; } completes immediately."""
        assert (
            run_c("int f() { int x = 0; while (1) { x = 42; break; } return x; }") == 42
        )

    def test_break_in_if(self) -> None:
        """Sum until i==3, then break."""
        assert (
            run_c(
                "int f() { int s = 0; int i = 0;"
                " while (i < 10) { if (i == 3) { break; } s = s + i; i = i + 1; }"
                " return s; }"
            )
            == 3  # 0+1+2
        )

    def test_nested_break(self) -> None:
        """Inner break doesn't exit outer loop."""
        assert (
            run_c(
                "int f() { int s = 0; int i = 0;"
                " while (i < 3) {"
                "   int j = 0;"
                "   while (j < 10) { if (j == 2) { break; } j = j + 1; }"
                "   s = s + j; i = i + 1;"
                " } return s; }"
            )
            == 6  # 2+2+2
        )

    def test_continue_while(self) -> None:
        """Continue skips the rest of the body."""
        assert (
            run_c(
                "int f() { int s = 0; int i = 0;"
                " while (i < 5) {"
                "   i = i + 1; if (i == 3) { continue; } s = s + i;"
                " } return s; }"
            )
            == 12  # 1+2+4+5
        )


class TestCLI:
    """Brick 2: the dcc CLI compiles and runs .c files."""

    def test_run_file_returns_result(self) -> None:
        """run_file compiles a .c file and returns the last function's output."""
        path = TESTDATA_DIR / "square.c"
        assert run_file(path, ["7"]) == 49

    def test_main_prints_result(self) -> None:
        """`python -m dcc square.c 7` prints 49."""
        path = TESTDATA_DIR / "square.c"
        runner = CliRunner()
        result = runner.invoke(main, [str(path), "7"])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "49"

    def test_main_rejects_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["/no/such/file.c"])
        assert result.exit_code != 0


class TestFunctionCalls:
    """Brick 7: Cross-function calls via ExternOp + function.CallOp."""

    def test_single_call(self) -> None:
        assert (
            run_c("int g(int x) { return x + 1; } int f(int x) { return g(x); }", 5)
            == 6
        )

    def test_multi_arg_call(self) -> None:
        assert (
            run_c(
                "int add(int a, int b) { return a + b; }"
                " int f(int a, int b) { return add(a, b); }",
                3,
                4,
            )
            == 7
        )

    def test_nested_call(self) -> None:
        """Compose two calls into one expression."""
        assert (
            run_c(
                "int sq(int x) { return x * x; } int f(int x) { return sq(sq(x)); }",
                3,
            )
            == 81
        )

    def test_call_in_expression(self) -> None:
        """Call appears as a subexpression of a larger expression."""
        assert (
            run_c(
                "int sq(int x) { return x * x; }"
                " int f(int x) { return sq(x) + sq(x + 1); }",
                3,
            )
            == 25  # 9 + 16
        )

    def test_call_with_mixed_operand(self) -> None:
        """Callee result used alongside a local variable."""
        assert (
            run_c(
                "int inc(int x) { return x + 1; }"
                " int f(int x) { int y = inc(x); return y + inc(y); }",
                5,
            )
            == 13  # (5+1) + (6+1)
        )

    def test_forward_declared_call(self) -> None:
        """Caller appears before the callee's definition in source order."""
        assert (
            run_c(
                "int g(int x);"
                " int f(int x) { return g(x) + 1; }"
                " int g(int x) { return x * 2; }"
                " int h(int x) { return f(x); }",
                4,
            )
            == 9  # (4*2) + 1
        )

    @pytest.mark.skip(
        reason="self-recursion hangs the DAG walker; needs func.recursive (TODO.md)"
    )
    def test_self_recursion(self) -> None:
        """int fact(int n) { ... n * fact(n-1) ... }."""
        assert (
            run_c(
                "int fact(int n)"
                " { int r; if (n <= 1) r = 1; else r = n * fact(n - 1); return r; }",
                5,
            )
            == 120
        )

    @pytest.mark.skip(
        reason="mutual recursion forms a FunctionOp↔FunctionOp capture cycle; hangs"
    )
    def test_mutual_recursion(self) -> None:
        assert (
            run_c(
                "int is_odd(int n);"
                " int is_even(int n)"
                "   { int r; if (n == 0) r = 1; else r = is_odd(n - 1); return r; }"
                " int is_odd(int n)"
                "   { int r; if (n == 0) r = 0; else r = is_even(n - 1); return r; }"
                " int f(int n) { return is_even(n); }",
                4,
            )
            == 1
        )

    @pytest.mark.skip(reason="variadic call lowering not yet implemented; hangs")
    def test_variadic_call(self) -> None:
        """pycparser accepts `...` in prototypes; lowering omits it."""
        run_c('int printf(const char *fmt, ...); int f(void) { return printf("hi"); }')

    def test_discarded_call_is_reachable(self) -> None:
        """A call whose return value is discarded must still be reachable
        from block.result. Without effect threading, the first call here
        would be dropped as dead code."""
        ir = lower(
            parse_c_string(
                "int g(int x) { return x + 1; } int f(int x) { g(x); return g(x) + 2; }"
            )
        )
        f_calls = [op for op in ir.body.values if isinstance(op, function.CallOp)]
        assert len(f_calls) == 2
        reachable = set(transitive_dependencies(ir.body.result))
        for call in f_calls:
            assert call in reachable, f"CallOp {call} not reachable"

    def test_void_call_statement_preserved(self) -> None:
        """A bare void call in statement position (non-final) must appear
        in the lowered function body — it's the only observable evidence
        that the callee ran."""
        ir = lower(parse_c_string("void g(int x); int f(int x) { g(x); return x; }"))
        f_calls = [op for op in ir.body.values if isinstance(op, function.CallOp)]
        assert len(f_calls) == 1
        reachable = set(transitive_dependencies(ir.body.result))
        assert f_calls[0] in reachable

    @pytest.mark.xfail(
        reason="indirect calls through function pointers not implemented"
    )
    def test_indirect_call(self) -> None:
        assert (
            run_c(
                "int g(int x) { return x + 1; }"
                " int f(int x) { int (*p)(int) = g; return p(x); }",
                5,
            )
            == 6
        )


class TestMemoryOrdering:
    """Structural tests for the use-def ordering of memory operations."""

    def test_diamond_read_write_dependencies(self) -> None:
        """W1, R2, R3, W4: reads are independent, write fences both reads.

        The use-def graph should be:
            W1 ← R2  (read depends on write)
            W1 ← R3  (read depends on write, independent of R2)
            R2, R3 ← W4  (write depends on both reads via pack)
            W4 ← R5  (final read depends on second write)

        R2 must NOT be in the transitive dependencies of R3 (independent).
        R2 and R3 must both be in the transitive dependencies of W4.
        """

        ir = lower(
            parse_c_string(
                "int f() { int x = 1; int a = x; int b = x; x = 2; return a + b + x; }"
            )
        )

        # Collect ops by role.
        assigns = [op for op in ir.body.ops if isinstance(op, c.AssignOp)]
        reads = [op for op in ir.body.ops if isinstance(op, c.LvalueToRvalueOp)]

        # Filter to x-variable ops only (skip a, b assignments/reads).
        x_assigns = [
            a
            for a in assigns
            if isinstance(a.lvalue, c.LvalueVarOp)
            and a.lvalue.var_name.__constant__.to_json() == "x"
        ]
        x_reads = [
            r
            for r in reads
            if isinstance(r.lvalue, c.LvalueVarOp)
            and r.lvalue.var_name.__constant__.to_json() == "x"
        ]

        assert len(x_assigns) == 2  # W1 (x=1) and W4 (x=2)
        assert len(x_reads) == 3  # R2 (for a), R3 (for b), R5 (for return x)

        write_1, write_4 = x_assigns
        read_2, read_3, read_5 = x_reads

        def deps(value: dgen.Value) -> set[dgen.Value]:
            return set(transitive_dependencies(value))

        # R2 and R3 are independent: neither is in the other's dependencies.
        assert read_2 not in deps(read_3)
        assert read_3 not in deps(read_2)

        # Both R2 and R3 depend on W1.
        assert write_1 in deps(read_2)
        assert write_1 in deps(read_3)

        # W4 depends on both R2 and R3.
        write_4_deps = deps(write_4)
        assert read_2 in write_4_deps
        assert read_3 in write_4_deps

        # R5 (return x) depends on W4.
        assert write_4 in deps(read_5)

        # End-to-end correctness: a=1, b=1, x=2, total=4.
        assert (
            run_c(
                "int f() { int x = 1; int a = x; int b = x; x = 2; return a + b + x; }"
            )
            == 4
        )
