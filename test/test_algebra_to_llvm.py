"""Tests for AlgebraToLLVM: every algebra op lowers and JITs correctly."""

from dgen import codegen
from dgen.asm.parser import parse_module
from dgen.testing import strip_prefix


def _jit(ir: str, *args: object) -> object:
    """Parse IR, compile, run with args, return JSON result."""
    module = parse_module(strip_prefix(ir))
    exe = codegen.compile(module)
    return exe.run(*args).to_json()


# --- Arithmetic (integer) ---


def test_add_index():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.add(%x, 10)
    """,
            5,
        )
        == 15
    )


def test_negate_index():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.negate(%x)
    """,
            7,
        )
        == -7
    )


def test_subtract_index():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.subtract(%x, 3)
    """,
            10,
        )
        == 7
    )


def test_multiply_index():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.multiply(%x, 4)
    """,
            3,
        )
        == 12
    )


def test_divide_index():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.divide(%x, 3)
    """,
            10,
        )
        == 3
    )


# --- Arithmetic (float) ---


def test_add_float():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.add(%x, 1.5)
    """,
            2.5,
        )
        == 4.0
    )


def test_negate_float():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.negate(%x)
    """,
            3.0,
        )
        == -3.0
    )


def test_subtract_float():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.subtract(%x, 1.0)
    """,
            5.0,
        )
        == 4.0
    )


def test_multiply_float():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.multiply(%x, 3.0)
    """,
            2.0,
        )
        == 6.0
    )


def test_divide_float():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.divide(%x, 2.0)
    """,
            7.0,
        )
        == 3.5
    )


def test_reciprocal():
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import number
        | %f : function.Function<number.Float64> = function.function<number.Float64>() body(%x: number.Float64):
        |     %r : number.Float64 = algebra.reciprocal(%x)
    """,
            4.0,
        )
        == 0.25
    )


# --- Lattice / bitwise ---


def test_meet():
    # 5 & 6 = 4
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.meet(%x, 6)
    """,
            5,
        )
        == 4
    )


def test_join():
    # 5 | 6 = 7
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.join(%x, 6)
    """,
            5,
        )
        == 7
    )


def test_complement():
    # ~0 = -1
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.complement(%x)
    """,
            0,
        )
        == -1
    )


def test_symmetric_difference():
    # 5 ^ 3 = 6
    assert (
        _jit(
            """
        | import algebra
        | import function
        | import index
        | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.symmetric_difference(%x, 3)
    """,
            5,
        )
        == 6
    )


# --- Comparison (returns Boolean, cast to Index for JIT return) ---

_CMP_TEMPLATE = """
    | import algebra
    | import function
    | import index
    | import number
    | %f : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
    |     %cmp : number.Boolean = algebra.{op}(%x, 5)
    |     %r : index.Index = algebra.cast(%cmp)
"""


def test_equal_true():
    assert _jit(_CMP_TEMPLATE.format(op="equal"), 5) == 1


def test_equal_false():
    assert _jit(_CMP_TEMPLATE.format(op="equal"), 3) == 0


def test_not_equal():
    assert _jit(_CMP_TEMPLATE.format(op="not_equal"), 5) == 0
    assert _jit(_CMP_TEMPLATE.format(op="not_equal"), 3) == 1


def test_less_than():
    assert _jit(_CMP_TEMPLATE.format(op="less_than"), 3) == 1
    assert _jit(_CMP_TEMPLATE.format(op="less_than"), 5) == 0


def test_less_equal():
    assert _jit(_CMP_TEMPLATE.format(op="less_equal"), 5) == 1
    assert _jit(_CMP_TEMPLATE.format(op="less_equal"), 6) == 0


def test_greater_than():
    assert _jit(_CMP_TEMPLATE.format(op="greater_than"), 6) == 1
    assert _jit(_CMP_TEMPLATE.format(op="greater_than"), 5) == 0


def test_greater_equal():
    assert _jit(_CMP_TEMPLATE.format(op="greater_equal"), 5) == 1
    assert _jit(_CMP_TEMPLATE.format(op="greater_equal"), 4) == 0
