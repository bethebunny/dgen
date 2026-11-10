"""Tests for use-def graph utilities."""

import sys

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.graph import all_values, interior_values, transitive_dependencies
from dgen.module import ConstantOp
from dgen.op import Op
from dgen.testing import assert_ir_equivalent, llvm_compile, strip_prefix
from dgen.verify import verify_closed_blocks, verify_dag


def test_transitive_dependencies_linear_chain():
    """Walk a simple linear dependency chain."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=2, type=builtin.Index())
    c = llvm.AddOp(lhs=a, rhs=b)
    deps = list(transitive_dependencies(c))
    # Root is always last
    assert deps[-1] is c
    # All three ops are present (plus type values)
    assert {v for v in deps if isinstance(v, Op)} == {a, b, c}


def test_transitive_dependencies_diamond():
    """Diamond dependency: a used by both b and c, both used by d."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = llvm.AddOp(lhs=a, rhs=a)
    c = llvm.MulOp(lhs=a, rhs=a)
    d = llvm.AddOp(lhs=b, rhs=c)
    deps = list(transitive_dependencies(d))
    ops = [v for v in deps if isinstance(v, Op)]
    assert ops[0] is a
    assert ops[-1] is d
    assert len(ops) == 4


def test_transitive_dependencies_visits_block_args():
    """BlockArguments are Values and appear in the traversal."""
    arg = BlockArgument(type=builtin.Index())
    op = llvm.AddOp(lhs=arg, rhs=arg)
    deps = list(transitive_dependencies(op))
    assert arg in deps
    assert op in deps


def test_transitive_dependencies_does_not_descend_into_blocks():
    """Ops nested inside another op's block are not included."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    func = parse(ir)
    inner = next(func.body.ops)
    deps = list(transitive_dependencies(func))
    assert func in deps
    assert inner not in deps


def test_chain_asm_round_trip():
    """chain op parses and formats correctly."""
    ir_text = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 0
        |     %1 : index.Index = 1
        |     %2 : index.Index = chain(%1, %0)
    """)
    m = parse(ir_text)
    assert_ir_equivalent(m, asm.parse(asm.format(m)))


def test_transitive_dependencies_follows_chain_dependencies():
    """chain(lhs, rhs) creates dependency on rhs, transitive_dependencies finds both."""
    a = ConstantOp(value=0, type=builtin.Index())
    b = ConstantOp(value=1, type=builtin.Index())
    c = builtin.ChainOp(lhs=b, rhs=a, type=builtin.Index())
    deps = list(transitive_dependencies(c))
    ops = {v for v in deps if isinstance(v, Op)}
    assert ops == {a, b, c}


def test_transitive_dependencies_includes_types():
    """Type values are included in the traversal."""
    a = ConstantOp(value=1, type=builtin.Index())
    deps = list(transitive_dependencies(a))
    # The Index type and its dependencies should be present
    assert any(isinstance(v, builtin.Index) for v in deps)
    assert deps[-1] is a


# ---------------------------------------------------------------------------
# all_values / interior_values
# ---------------------------------------------------------------------------


def _parse(text: str):
    return parse(strip_prefix(text))


def test_all_values_includes_nested_block_ops():
    """all_values descends into nested blocks, unlike transitive_dependencies."""
    func = _parse("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    inner_const = next(func.body.ops)
    # transitive_dependencies does NOT include inner ops
    assert inner_const not in list(transitive_dependencies(func))
    # all_values DOES include inner ops
    assert inner_const in list(all_values(func))


def test_all_values_includes_deeply_nested():
    """all_values reaches ops inside nested label/region bodies."""
    func = _parse("""
        | import function
        | import goto
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %init : index.Index = 0
        |     %r : goto.Label = goto.region([%init]) body<%self: goto.Label, %exit: goto.Label>(%i: index.Index):
        |         %inner : index.Index = 42
    """)
    vals = list(all_values(func))
    names = [v.name for v in vals if hasattr(v, "name") and v.name is not None]
    assert "inner" in names
    assert "r" in names


def test_interior_values_yields_block_contents():
    """interior_values yields values from a value's blocks, not the value itself."""
    func = _parse("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    inner = list(interior_values(func))
    op_names = [v.name for v in inner if isinstance(v, Op)]
    assert "0" in op_names
    # The function itself is NOT in interior_values
    assert func not in inner


def test_interior_values_empty_for_leaf_op():
    """A leaf op with no blocks has no interior values."""
    a = ConstantOp(value=1, type=builtin.Index())
    assert list(interior_values(a)) == []


# ---------------------------------------------------------------------------
# Deep chains
# ---------------------------------------------------------------------------

# Deeper than Python's default recursion limit, so any walk that recurses once
# per use-def edge fails on it.
DEEP_CHAIN = 2000


def _deep_chain_ir(depth: int) -> str:
    adds = "\n".join(
        f"    %v{i} : index.Index = algebra.add(%v{i - 1}, 1)" for i in range(1, depth)
    )
    return (
        "import algebra\nimport function\nimport index\n\n"
        "%f : function.Function<[index.Index], index.Index> = "
        "function.function<index.Index>() body(%x: index.Index):\n"
        f"    %v0 : index.Index = algebra.add(%x, 1)\n{adds}\n"
    )


def test_deep_chain_walks():
    assert DEEP_CHAIN > sys.getrecursionlimit()
    func = parse(_deep_chain_ir(DEEP_CHAIN))
    assert len(list(func.body.ops)) == DEEP_CHAIN
    verify_dag(func)
    verify_closed_blocks(func)
    assert_ir_equivalent(func, asm.parse(asm.format(func)))


# Slow until replacement stops sweeping the block once per lowered op.
@pytest.mark.slow
def test_deep_chain_runs():
    func = parse(_deep_chain_ir(DEEP_CHAIN))
    assert llvm_compile(func).run(5).to_json() == 5 + DEEP_CHAIN
