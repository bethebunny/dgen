"""Tests for diff_modules output format."""

from dgen.asm.parser import parse
from dgen.ir_diff import diff_values
from toy.dialects import toy  # noqa: F401 — registers toy dialect
from dgen.testing import strip_prefix


def test_diff_empty_when_identical():
    ir = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert diff_values(parse(ir), parse(ir)) == ""


def test_diff_empty_when_ssa_names_differ():
    """Same computation with different SSA names → no diff."""
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %tensor : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %result : Nil = toy.print(%tensor)
    """)
    assert diff_values(parse(a), parse(b)) == ""


def test_diff_empty_when_function_order_differs():
    """Same functions listed in different module order → no diff."""
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %func_a : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %func_b : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %1 : Nil = toy.print(%0)
    """)
    fa = parse(a)
    fb = parse(b)
    from dgen.dialects.builtin import ChainOp

    root_ab = ChainOp(lhs=fa, rhs=fb, type=fa.type)
    root_ba = ChainOp(lhs=fb, rhs=fa, type=fb.type)
    assert diff_values(root_ab, root_ba) == ""


def test_diff_empty_when_op_order_differs():
    """Independent ops in a different block order → no diff (same graph)."""
    a = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = function.function<toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64>>() body():
        |     %a : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %b : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %c : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = toy.mul(%a, %b)
    """)
    b = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = function.function<toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64>>() body():
        |     %x : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %y : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %z : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = toy.mul(%y, %x)
    """)
    assert diff_values(parse(a), parse(b)) == ""


def test_diff_format_semantic_change():
    """A changed constant produces a proper unified-diff output."""
    expected = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    actual = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    result = diff_values(parse(actual), parse(expected))
    assert result.startswith("--- expected\n+++ actual")
    assert "@@" in result
    assert any(
        line.startswith("-")
        for line in result.splitlines()
        if not line.startswith("---")
    )
    assert any(
        line.startswith("+")
        for line in result.splitlines()
        if not line.startswith("+++")
    )


def test_diff_format_missing_function():
    """A function present only in expected shows as all-deleted lines."""
    expected = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    actual_module = None
    result = diff_values(actual_module, parse(expected))
    assert result.startswith("--- expected\n+++ actual")
    lines = result.splitlines()
    assert any(ln.startswith("-") for ln in lines if not ln.startswith("---"))
    assert not any(ln.startswith("+") for ln in lines if not ln.startswith("+++"))


def test_diff_format_extra_function():
    """A function present only in actual shows as all-added lines."""
    actual = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | import toy
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    expected_module = None
    result = diff_values(parse(actual), expected_module)
    assert result.startswith("--- expected\n+++ actual")
    lines = result.splitlines()
    assert any(ln.startswith("+") for ln in lines if not ln.startswith("+++"))
    assert not any(ln.startswith("-") for ln in lines if not ln.startswith("---"))
