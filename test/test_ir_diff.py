"""Tests for diff_modules output format."""

from dgen.asm.parser import parse_module
from dgen.ir_diff import diff_modules
from dgen.module import Module
from toy.dialects import toy  # noqa: F401 — registers toy dialect
from dgen.testing import strip_prefix


def test_diff_empty_when_identical():
    ir = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    assert diff_modules(parse_module(ir), parse_module(ir)) == ""


def test_diff_empty_when_ssa_names_differ():
    """Same computation with different SSA names → no diff."""
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %tensor : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %result : Nil = toy.print(%tensor)
    """)
    assert diff_modules(parse_module(a), parse_module(b)) == ""


def test_diff_empty_when_function_order_differs():
    """Same functions listed in different module order → no diff."""
    a = strip_prefix("""
        | import toy
        |
        | %func_a : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %func_b : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %1 : Nil = toy.print(%0)
    """)
    fa = parse_module(a).functions[0]
    fb = parse_module(b).functions[0]
    module_ab = Module(functions=[fa, fb])
    module_ba = Module(functions=[fb, fa])
    assert diff_modules(module_ab, module_ba) == ""


def test_diff_empty_when_op_order_differs():
    """Independent ops in a different block order → no diff (same graph)."""
    a = strip_prefix("""
        | import toy
        |
        | %main : toy.Tensor<affine.Shape<2>([2, 3]), F64> = function<toy.Tensor<affine.Shape<2>([2, 3]), F64>>() ():
        |     %a : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %b : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %c : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.mul(%a, %b)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : toy.Tensor<affine.Shape<2>([2, 3]), F64> = function<toy.Tensor<affine.Shape<2>([2, 3]), F64>>() ():
        |     %x : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        |     %y : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %z : toy.Tensor<affine.Shape<2>([2, 3]), F64> = toy.mul(%y, %x)
    """)
    assert diff_modules(parse_module(a), parse_module(b)) == ""


def test_diff_format_semantic_change():
    """A changed constant produces a proper unified-diff output."""
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    actual = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
    """)
    result = diff_modules(parse_module(actual), parse_module(expected))
    assert result.startswith("IR equivalence check failed.")
    assert "function 'main':" in result
    assert "@@" in result
    assert any(line.startswith("-  ") for line in result.splitlines())
    assert any(line.startswith("+  ") for line in result.splitlines())


def test_diff_format_missing_function():
    """A function present only in expected shows as all-deleted lines."""
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    actual_module = Module(functions=[])
    result = diff_modules(actual_module, parse_module(expected))
    assert result.startswith("IR equivalence check failed.")
    assert "function 'main':" in result
    assert all(
        line.startswith("-  ") or not line.startswith((" ", "+"))
        for line in result.splitlines()
        if line.startswith("-") or line.startswith("+")
    )
    assert not any(line.startswith("+  ") for line in result.splitlines())


def test_diff_format_extra_function():
    """A function present only in actual shows as all-added lines."""
    actual = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<affine.Shape<2>([2, 3]), F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
    """)
    expected_module = Module(functions=[])
    result = diff_modules(parse_module(actual), expected_module)
    assert result.startswith("IR equivalence check failed.")
    assert "function 'main':" in result
    assert any(line.startswith("+  ") for line in result.splitlines())
    assert not any(line.startswith("-  ") for line in result.splitlines())
