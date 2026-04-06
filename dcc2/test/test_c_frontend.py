"""Tests for the dcc2 C frontend."""

from __future__ import annotations

from pathlib import Path

import pytest
from pycparser import c_ast

import dgen
from dgen import Dialect
from dgen.block import BlockArgument
from dgen.dialects import function, index
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.number import SignedInteger
from dgen.module import ConstantOp, pack

from dcc2.cli import c_compiler
from dcc2.parser.c_parser import parse_c_string
from dcc2.parser.lowering import lower

# Make dcc2 dialect discoverable.
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
        from dcc2.dialects import c

        assert c.c.name == "c"

    def test_lvalue_ops_exist(self) -> None:
        from dcc2.dialects.c import (
            AddressOfOp,
            AssignOp,
            CompoundAssignOp,
            LvalueArrowOp,
            LvalueDerefOp,
            LvalueMemberOp,
            LvalueSubscriptOp,
            LvalueToRvalueOp,
            LvalueVarOp,
            PostDecrementOp,
            PostIncrementOp,
            PreDecrementOp,
            PreIncrementOp,
        )

        assert LvalueVarOp is not None
        assert LvalueDerefOp is not None
        assert LvalueSubscriptOp is not None
        assert LvalueMemberOp is not None
        assert LvalueArrowOp is not None
        assert LvalueToRvalueOp is not None
        assert AddressOfOp is not None
        assert AssignOp is not None
        assert CompoundAssignOp is not None
        assert PreIncrementOp is not None
        assert PostIncrementOp is not None
        assert PreDecrementOp is not None
        assert PostDecrementOp is not None

    def test_conversion_ops_exist(self) -> None:
        from dcc2.dialects.c import (
            ArithmeticConvertOp,
            ArrayDecayOp,
            FunctionDecayOp,
            IntegerPromoteOp,
            NullToPointerOp,
            ScalarToBoolOp,
        )

        assert IntegerPromoteOp is not None
        assert ArithmeticConvertOp is not None
        assert ArrayDecayOp is not None
        assert FunctionDecayOp is not None
        assert NullToPointerOp is not None
        assert ScalarToBoolOp is not None

    def test_c_types_exist(self) -> None:
        from dcc2.dialects.c import (
            CFunctionType,
            Enum,
            Struct,
            StructField,
            Union,
        )

        assert Struct is not None
        assert StructField is not None
        assert Union is not None
        assert Enum is not None
        assert CFunctionType is not None

    def test_arithmetic_and_control_ops_exist(self) -> None:
        from dcc2.dialects.c import (
            CommaOp,
            CReturnOp,
            CSizeofOp,
            LogicalNotOp,
            ModuloOp,
            ShiftLeftOp,
            ShiftRightOp,
        )

        assert CReturnOp is not None
        assert CSizeofOp is not None
        assert ModuloOp is not None
        assert ShiftLeftOp is not None
        assert ShiftRightOp is not None
        assert LogicalNotOp is not None
        assert CommaOp is not None

    def test_type_constructors(self) -> None:
        from dgen.dialects.builtin import Nil
        from dgen.dialects.memory import Reference
        from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger

        from dcc2.dialects import c_double, c_float, c_int, c_ptr, c_void

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
        body_result = ConstantOp(value=42, type=ret_type)
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
        from dgen.dialects.number import SignedInteger, UnsignedInteger

        from dcc2.parser.type_resolver import TypeResolver

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
        from dgen.dialects.number import Float64

        from dcc2.parser.type_resolver import TypeResolver

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
        from dgen.dialects.builtin import Nil

        from dcc2.parser.type_resolver import TypeResolver

        r = TypeResolver()
        v = r.resolve(
            c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["void"]))
        )
        assert isinstance(v, Nil)

    def test_pointer(self) -> None:
        from dgen.dialects.memory import Reference

        from dcc2.parser.type_resolver import TypeResolver

        r = TypeResolver()
        p = r.resolve(
            c_ast.PtrDecl(
                None,
                c_ast.TypeDecl(None, None, None, c_ast.IdentifierType(names=["int"])),
            )
        )
        assert isinstance(p, Reference)

    def test_struct_with_fields(self) -> None:
        from dcc2.dialects.c import Struct
        from dcc2.parser.type_resolver import TypeResolver

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
        assert isinstance(t, Struct)
        # Verify struct is cached and forward-referenceable.
        assert r._resolve_struct(c_ast.Struct("Point", None)) is t

    def test_anonymous_structs_are_unique(self) -> None:
        from dcc2.dialects.c import Struct
        from dcc2.parser.type_resolver import TypeResolver

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
        assert isinstance(s1, Struct)
        assert isinstance(s2, Struct)
        assert s1 is not s2

    def test_enum_constants(self) -> None:
        from dcc2.parser.type_resolver import TypeResolver

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
        from dcc2.parser.type_resolver import TypeResolver, TypeResolverError

        r = TypeResolver()
        with pytest.raises(TypeResolverError, match="unknown type"):
            r.resolve(
                c_ast.TypeDecl(
                    None, None, None, c_ast.IdentifierType(names=["mystery_t"])
                )
            )

    def test_f_void_has_no_params(self) -> None:
        """C11: f(void) should resolve to a function with zero parameters."""
        from dcc2.dialects.c import CFunctionType
        from dcc2.parser.type_resolver import TypeResolver

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
        assert isinstance(t, CFunctionType)

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
