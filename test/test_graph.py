"""Tests for use-def graph utilities."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.graph import all_values, interior_values, transitive_dependencies
from dgen.module import ConstantOp
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
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    module = parse(ir)
    func = module
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
    module = _parse("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    func = module
    inner_const = next(func.body.ops)
    # transitive_dependencies does NOT include inner ops
    assert inner_const not in list(transitive_dependencies(func))
    # all_values DOES include inner ops
    assert inner_const in list(all_values(func))


def test_all_values_includes_deeply_nested():
    """all_values reaches ops inside nested label/region bodies."""
    module = _parse("""
        | import function
        | import goto
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %init : index.Index = 0
        |     %r : goto.Label = goto.region([%init]) body<%self: goto.Label, %exit: goto.Label>(%i: index.Index):
        |         %inner : index.Index = 42
    """)
    func = module
    vals = list(all_values(func))
    names = [v.name for v in vals if hasattr(v, "name") and v.name is not None]
    assert "inner" in names
    assert "r" in names


def test_interior_values_yields_block_contents():
    """interior_values yields values from a value's blocks, not the value itself."""
    module = _parse("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    func = module
    inner = list(interior_values(func))
    op_names = [v.name for v in inner if isinstance(v, Op)]
    assert "0" in op_names
    # The function itself is NOT in interior_values
    assert func not in inner


def test_interior_values_empty_for_leaf_op():
    """A leaf op with no blocks has no interior values."""
    a = ConstantOp(value=1, type=builtin.Index())
    assert list(interior_values(a)) == []
