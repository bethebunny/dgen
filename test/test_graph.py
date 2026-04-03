"""Tests for use-def graph utilities."""

import dgen
from dgen import asm
from dgen.asm.parser import parse_module
from dgen.block import BlockArgument
from dgen.dialects import builtin, function, llvm
from dgen.dialects.function import Function
from dgen.graph import transitive_dependencies
from dgen.module import ConstantOp, pack
from dgen.op import Op
from dgen.testing import assert_ir_equivalent, strip_prefix


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
    inner = ConstantOp(value=42, type=builtin.Index())
    func = function.FunctionOp(
        name="f",
        body=dgen.Block(result=inner, args=[]),
        result_type=builtin.Nil(),
        type=Function(arguments=pack(), result_type=builtin.Nil()),
    )
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
    m = parse_module(ir_text)
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
