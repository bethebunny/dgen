"""Tests for the C frontend: type resolution, end-to-end JIT, and scale."""

from __future__ import annotations

import random
import shutil
import time
import urllib.request
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from dgen.module import Module
from dgen.testing import llvm_compile
from dgen.compiler import Compiler, IdentityPass
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.dialects.builtin import Nil
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64, SignedInteger, UnsignedInteger
from dcc.dialects import c_int
from dcc.dialects.c import CStruct
from dcc.parser.c_parser import parse_c_string
from dcc.parser.lowering import lower
from dcc.parser.type_resolver import TypeResolver
from dcc.passes.c_to_llvm import CToLLVM
from dcc.passes.c_to_memory import CToMemory

TESTDATA = Path(__file__).parent / "testdata"

_c_compiler = Compiler(
    [CToMemory(), CToLLVM(), AlgebraToLLVM(), MemoryToLLVM()], IdentityPass()
)

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
        assert isinstance(r._resolve_identifier_type(["int"]), SignedInteger)

    def test_unsigned_long(self) -> None:
        r = TypeResolver()
        assert isinstance(
            r._resolve_identifier_type(["unsigned", "long"]), UnsignedInteger
        )

    def test_void(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["void"]), Nil)

    def test_double(self) -> None:
        r = TypeResolver()
        assert isinstance(r._resolve_identifier_type(["double"]), Float64)

    def test_typedef(self) -> None:
        r = TypeResolver()
        r.register_typedef("myint", c_int(32))
        assert isinstance(r._resolve_identifier_type(["myint"]), SignedInteger)

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
        assert isinstance(r.resolve(node), Reference)

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

    def test_local_variable(self) -> None:
        """Local var: alloca + store + load must be in the use-def graph."""
        assert run_c("int f(int x) { int y = x + 1; return y; }", 5) == 6

    def test_local_mutation(self) -> None:
        """Mutating a local via assignment."""
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

    def test_function_call_e2e(self) -> None:
        """Cross-function calls."""
        assert (
            run_c("int g(int x) { return x + 1; }\nint f(int x) { return g(x); }", 5)
            == 6
        )

    def test_assign_then_return(self) -> None:
        """Assignment followed by return."""
        assert run_c("int f(int x) { int y = 0; y = x + 1; return y; }", 5) == 6


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
        assert stats.skipped_functions == 0


# ---------------------------------------------------------------------------
# Session-fix regression tests: minimal repros for things that used to
# fail at the C frontend / codegen boundary. Each test has:
#   - a structural assertion that pinpoints what changed semantically;
#   - an ir_snapshot that captures the full lowered IR for cosmetic review.
# ---------------------------------------------------------------------------


def _lower_c(source: str) -> Module:
    ast = parse_c_string(source)
    module, stats = lower(ast)
    assert stats.skipped_functions == 0, (
        f"lowering skipped functions: {stats.skip_reasons}"
    )
    return module


def _lower_to_llvm(source: str) -> Module:
    """Run the full frontend pipeline, returning the lowered Module."""
    from dgen.passes.control_flow_to_goto import ControlFlowToGoto
    from dgen.compiler import verify_passes

    ast = parse_c_string(source)
    module, _ = lower(ast)
    pipeline = Compiler(
        [CToMemory(), CToLLVM(), AlgebraToLLVM(), MemoryToLLVM(), ControlFlowToGoto()],
        IdentityPass(),
    )
    token = verify_passes.set(False)
    try:
        return pipeline.run(module)
    finally:
        verify_passes.reset(token)


def _codegen_verifies(source: str) -> None:
    """Compile source through the full pipeline; assert LLVM parses + verifies."""
    from dgen.codegen import emit_llvm_ir
    import llvmlite.binding as llvm_binding

    llvm_binding.initialize_native_target()
    llvm_binding.initialize_native_asmprinter()

    module = _lower_to_llvm(source)
    ir, _ = emit_llvm_ir(module)
    mod = llvm_binding.parse_assembly(ir)
    mod.verify()


def _contains_op(module: Module, op_type: type) -> bool:
    from dgen.graph import all_values

    for top in module.ops:
        for v in all_values(top):
            if isinstance(v, op_type):
                return True
    return False


def _count_ops(module: Module, op_type: type) -> int:
    from dgen.graph import all_values

    return sum(isinstance(v, op_type) for top in module.ops for v in all_values(top))


class TestSessionFrontendFixes:
    """Minimal repros for each frontend lowering fix in this session.

    Every test asserts the specific IR structure that distinguishes a
    correct lowering from the pre-fix behavior, then snapshots the full
    Module. Regressions show up as a structural assertion failure; purely
    cosmetic IR shifts only update the snapshot.
    """

    def test_global_variable_reference(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: ID lookup failed with LoweringError("undefined: g")
        # because globals were never registered in file_scope.
        from dgen.dialects.builtin import ExternOp

        m = _lower_c("int g; int f(void) { return g; }")
        assert _contains_op(m, ExternOp), "global should become an ExternOp in the IR"
        assert m == ir_snapshot

    def test_array_subscript_on_array_type(
        self, ir_snapshot: SnapshotAssertion
    ) -> None:
        # Pre-fix: _pointee on Array raised "cannot dereference non-pointer".
        from dcc.dialects.c import SubscriptOp

        m = _lower_c("int arr[10]; int f(void) { return arr[3]; }")
        assert _contains_op(m, SubscriptOp), "arr[3] should produce SubscriptOp"
        assert m == ir_snapshot

    def test_star_array_decays_to_pointer(self, ir_snapshot: SnapshotAssertion) -> None:
        from dcc.dialects.c import DereferenceOp

        m = _lower_c("int arr[10]; int f(void) { return *arr; }")
        assert _contains_op(m, DereferenceOp)
        assert m == ir_snapshot

    def test_lvalue_star_p(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: _assign rejected UnaryOp lvalue.
        from dcc.dialects.c import StoreIndirectOp

        m = _lower_c("void f(int *p) { *p = 5; }")
        assert _count_ops(m, StoreIndirectOp) == 1, (
            "*p = 5 should emit exactly one StoreIndirectOp"
        )
        assert m == ir_snapshot

    def test_lvalue_arrow_field(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: _assign rejected StructRef lvalue.
        from dcc.dialects.c import FieldAddressOp, StoreIndirectOp

        m = _lower_c("struct S { int x; }; void f(struct S *s) { s->x = 7; }")
        assert _count_ops(m, FieldAddressOp) == 1
        assert _count_ops(m, StoreIndirectOp) == 1
        assert m == ir_snapshot

    def test_lvalue_dot_field(self, ir_snapshot: SnapshotAssertion) -> None:
        # obj.x = v on a local struct: dot → address_of(obj) + field_address.
        from dcc.dialects.c import FieldAddressOp, StoreIndirectOp

        m = _lower_c("struct S { int x; }; void f(void) { struct S s; s.x = 7; }")
        assert _contains_op(m, FieldAddressOp)
        assert _contains_op(m, StoreIndirectOp)
        assert m == ir_snapshot

    def test_lvalue_array_index(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: _assign rejected ArrayRef lvalue.
        from dcc.dialects.c import ElementAddressOp, StoreIndirectOp

        m = _lower_c("void f(int *a) { a[3] = 42; }")
        assert _count_ops(m, ElementAddressOp) == 1
        assert _count_ops(m, StoreIndirectOp) == 1
        assert m == ir_snapshot

    def test_compound_assign_through_pointer(
        self, ir_snapshot: SnapshotAssertion
    ) -> None:
        # *p += 5: load through addr, add, store back.
        from dcc.dialects.c import DereferenceOp, StoreIndirectOp

        m = _lower_c("void f(int *p) { *p += 5; }")
        assert _contains_op(m, DereferenceOp)
        assert _contains_op(m, StoreIndirectOp)
        assert m == ir_snapshot

    def test_increment_through_pointer(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: ++/-- on non-ID raised LoweringError.
        # Desugared to load + add + store — no PostIncrementOp in IR.
        from dcc.dialects.c import PostIncrementOp, StoreIndirectOp

        m = _lower_c("void f(int *p) { (*p)++; }")
        assert not _contains_op(m, PostIncrementOp), (
            "non-ID increment must be desugared via StoreIndirectOp"
        )
        assert _contains_op(m, StoreIndirectOp)
        assert m == ir_snapshot

    def test_assign_as_expression_non_id(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: "assign-as-expression on non-variable".
        # The read-back dereference becomes the expression's value.
        from dcc.dialects.c import DereferenceOp

        m = _lower_c("int f(int *p) { int x = (*p = 5); return x; }")
        assert _contains_op(m, DereferenceOp)
        assert m == ir_snapshot

    def test_unresolved_struct_field_chain(
        self, ir_snapshot: SnapshotAssertion
    ) -> None:
        # Pre-fix: unknown field type fell back to c_int(64); chained ->
        # then died with "cannot dereference non-pointer type: i64".
        m = _lower_c(
            "struct Fwd; struct Outer { struct Fwd *p; };"
            "int f(struct Outer *o) { return o->p->x; }"
        )
        # Ratchet: this used to skip the function; presence of the
        # function is the structural signal.
        assert len(m.functions) == 1
        assert m == ir_snapshot

    def test_long_long_int_literal(self, ir_snapshot: SnapshotAssertion) -> None:
        # Pre-fix: "unsupported constant type: long long int"
        from dgen.dialects.number import SignedInteger
        from dgen.module import ConstantOp

        m = _lower_c("long long f(void) { return 1LL + 2LL; }")
        # Both constants should land as 64-bit signed integers.
        consts = [
            v
            for top in m.ops
            for v in __import__("dgen.graph", fromlist=["all_values"]).all_values(top)
            if isinstance(v, ConstantOp) and isinstance(v.type, SignedInteger)
        ]
        assert consts, "no signed-integer constants found"
        for c in consts:
            assert c.type.bits.__constant__.to_json() == 64
        assert m == ir_snapshot

    def test_function_pointer_call_parameter(
        self, ir_snapshot: SnapshotAssertion
    ) -> None:
        # Pre-fix: "indirect function calls not yet supported"
        from dgen.dialects import function as _function
        from dcc.dialects.c import CallOp as CCallOp

        m = _lower_c("int f(int (*p)(int)) { return p(5); }")
        # Function pointer parameter: expect function.CallOp, not c.CallOp.
        assert _contains_op(m, _function.CallOp), (
            "indirect call should become function.CallOp"
        )
        assert not _contains_op(m, CCallOp), (
            "calling a function-pointer local must not become c.CallOp"
        )
        assert m == ir_snapshot

    def test_function_pointer_field_call(self, ir_snapshot: SnapshotAssertion) -> None:
        # p->fn(x): call through a struct-field function pointer.
        from dgen.dialects import function as _function

        m = _lower_c(
            "struct S { int (*fn)(int); };"
            "int f(struct S *s, int x) { return s->fn(x); }"
        )
        assert _contains_op(m, _function.CallOp)
        assert m == ir_snapshot


class TestSessionCodegenFixes:
    """Codegen-level fixes in this session."""

    def test_null_pointer_constant(self) -> None:
        # Pre-fix: ConstantOp(0, Reference) printed as "0", which LLVM
        # rejects with "integer constant must have integer type". A
        # pointer initialised to 0 and then used as a pointer triggers it.
        _codegen_verifies("int *f(void) { int *p = 0; return p; }")

    def test_function_pointer_call_codegen(self) -> None:
        # Pre-fix: emit_function_call always printed @name, failing for
        # SSA-valued callees.
        _codegen_verifies("int f(int (*p)(int), int x) { return p(x); }")


# ---------------------------------------------------------------------------
# Outstanding bugs — xfailed so they'll turn green automatically when fixed.
# Each one reproduces the smallest failing case I could find.
# ---------------------------------------------------------------------------


class TestKnownFailures:
    @pytest.mark.xfail(
        reason="frontend: 32-bit signed int dereference (3 cases in sqlite3)",
        strict=True,
    )
    def test_int32_deref(self) -> None:
        _lower_c("int f(int x) { return *x; }")

    @pytest.mark.xfail(
        reason="codegen: use of undefined value — top bucket (~850 in sqlite3). "
        "if/else with a written-in-both-branches local var loses its phi.",
        strict=True,
    )
    def test_codegen_undefined_value_if_return(self) -> None:
        _codegen_verifies("int f(int x, int y) { if (x > 0) return y; else return x; }")

    @pytest.mark.xfail(
        reason="codegen: multiple definition of local value (~270 in sqlite3). "
        "A local assigned inside an if gets a second alloca in the branch block.",
        strict=True,
    )
    def test_codegen_slot_collision_local_written_in_branch(self) -> None:
        _codegen_verifies("void f(int X) { long a = X; if (X > 0) { a = 1; } }")

    @pytest.mark.xfail(
        reason="codegen: Array.from_json asserts on zero-init of a local array "
        "(17 in sqlite3). _decl emits ConstantOp(0, Array(...)).",
        strict=True,
    )
    def test_codegen_array_init_assertion(self) -> None:
        _codegen_verifies("int f(void) { int arr[4]; arr[0] = 1; return arr[0]; }")


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
        assert stats.skipped_functions == 0
        assert len(module.functions) == 1500

    @pytest.mark.slow
    def test_5000_functions(self) -> None:
        source = _generate_sqlite_scale_c(5000)
        t0 = time.perf_counter()
        ast = parse_c_string(source)
        module, stats = lower(ast)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Pipeline took {elapsed:.1f}s"
        assert stats.functions == 5000
        assert stats.skipped_functions == 0
        assert len(module.functions) == 5000


# ---------------------------------------------------------------------------
# sqlite3.c: download, preprocess with gcc -E, parse, and lower
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSqlite3:
    """Parse and lower the actual sqlite3.c amalgamation."""

    def test_parse_sqlite3(self, sqlite3_ast: object) -> None:
        """pycparser can parse sqlite3.c (265K lines, ~9MB)."""
        assert len(sqlite3_ast.ext) > 1000

    def test_lower_sqlite3(self, sqlite3_ast: object) -> None:
        """Lower sqlite3.c to dgen IR."""
        t0 = time.perf_counter()
        module, stats = lower(sqlite3_ast)
        elapsed = time.perf_counter() - t0

        lowered = len(module.functions)
        skipped = stats.skipped_functions
        total = stats.functions
        print(
            f"\nsqlite3 lowering: {lowered} lowered, {skipped} skipped, {total} total"
        )
        if stats.skip_reasons:
            top = sorted(stats.skip_reasons.items(), key=lambda x: -x[1])[:10]
            print("top skip reasons:")
            for reason, count in top:
                print(f"  {count:5d}  {reason[:120]}")

        assert lowered >= 2560, f"lowered regressed: {lowered}"
        assert elapsed < 120, f"Lowering took {elapsed:.1f}s"

    def test_codegen_sqlite3(self, sqlite3_ast: object) -> None:
        """Lower sqlite3.c through the full pipeline to LLVM-verified IR.

        Tracks progress across four stages. Each assertion is a ratchet —
        update the threshold as improvements land.
        """
        import sys

        from dgen.codegen import EmitContext, _emit_ctx, emit, prepare_function
        from dgen.passes.control_flow_to_goto import ControlFlowToGoto

        import llvmlite.binding as llvm_binding

        llvm_binding.initialize_native_target()
        llvm_binding.initialize_native_asmprinter()

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 50000))

        module, _ = lower(sqlite3_ast)
        pipeline = Compiler(
            [
                CToMemory(),
                CToLLVM(),
                AlgebraToLLVM(),
                MemoryToLLVM(),
                ControlFlowToGoto(),
            ],
            IdentityPass(),
        )
        # TODO: ChainOps inside if-bodies pull parent-scope ops into block.ops.
        # _closed_block captures direct deps but not transitive — needs fixing.
        from dgen.compiler import verify_passes

        token = verify_passes.set(False)
        module = pipeline.run(module)
        verify_passes.reset(token)

        import re

        total = len(module.functions)
        emitted = 0
        parsed = 0
        verified = 0
        emit_errors: dict[str, int] = {}
        parse_errors: dict[str, int] = {}
        preamble = "declare void @print_memref(ptr, i64)\ndeclare ptr @malloc(i64)\n\n"
        for func in module.functions:
            ctx = EmitContext()
            ctx_token = _emit_ctx.set(ctx)
            try:
                prepare_function(func, ctx)
                lines = list(emit(func))
            except Exception as e:
                import traceback

                first = (str(e).splitlines() or [""])[0][:100]
                tb = traceback.extract_tb(e.__traceback__)
                site = (
                    f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}"
                    if tb
                    else "?"
                )
                key = f"{type(e).__name__}@{site}: {first}"
                emit_errors[key] = emit_errors.get(key, 0) + 1
                continue
            finally:
                _emit_ctx.reset(ctx_token)
            emitted += 1
            ir = preamble + "\n".join(lines)
            try:
                mod = llvm_binding.parse_assembly(ir)
                parsed += 1
                mod.verify()
                verified += 1
            except Exception as e:
                # llvmlite puts the real diagnostic on subsequent lines.
                # Pick the first line that looks like a diagnostic.
                lines_ = str(e).splitlines()
                msg = ""
                for line in lines_:
                    if "error:" in line or "<string>:" in line:
                        msg = line
                        break
                if not msg and lines_:
                    msg = lines_[-1]
                # Strip llvmlite's leading "<string>:L:C: error:" prefix
                msg = re.sub(r"^<string>:\d+:\d+:\s*", "", msg)
                msg = re.sub(r"^error:\s*", "", msg)
                # Normalise away specific %names, @names, numeric ids
                norm = re.sub(r"[%@][\w.]+", "%X", msg)
                norm = re.sub(r"\bi\d+\b", "iN", norm)
                norm = re.sub(r"\b\d+\b", "N", norm)
                norm = norm[:140]
                parse_errors[norm] = parse_errors.get(norm, 0) + 1

        report = (
            f"\nsqlite3 codegen progress:\n"
            f"  functions:     {total}\n"
            f"  IR emitted:    {emitted:5d} ({100 * emitted / total:5.1f}%)\n"
            f"  LLVM parsed:   {parsed:5d} ({100 * parsed / total:5.1f}%)\n"
            f"  LLVM verified: {verified:5d} ({100 * verified / total:5.1f}%)\n"
        )
        print(report)
        if emit_errors:
            print("top emit errors:")
            for key, n in sorted(emit_errors.items(), key=lambda x: -x[1])[:10]:
                print(f"  {n:5d}  {key}")
        if parse_errors:
            print("top LLVM parse/verify errors:")
            for key, n in sorted(parse_errors.items(), key=lambda x: -x[1])[:15]:
                print(f"  {n:5d}  {key}")

        # Ratchets — raise these as we fix things
        sys.setrecursionlimit(old_limit)

        # Ratchets — raise as we fix things
        assert emitted >= total - 20, f"emitted regressed: {emitted}/{total}\n{report}"
        assert verified >= 1180, f"verified regressed: {verified}\n{report}"
