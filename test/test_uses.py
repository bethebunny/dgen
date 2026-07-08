"""Ephemeral Use handles: capture-guided iteration and edge rebinding."""

import dgen
from dgen.block import BlockArgument
from dgen.builtins import PackOp, pack
from dgen.dialects import algebra, goto, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.ir.uses import rebind_all, uses_of


def _single_cell_load() -> tuple[memory.BufferLoadOp, memory.BufferStackAllocateOp]:
    alloca = memory.BufferStackAllocateOp(
        element_type=Index(),
        count=Index().constant(1),
        type=memory.Buffer(element_type=Index()),
    )
    load = memory.BufferLoadOp(
        mem=alloca, buf=alloca, index=Index().constant(0), type=Index()
    )
    return load, alloca


def test_rebind_single_edge_leaves_sibling_field():
    """buf and mem hold the same alloca; rebinding the mem edge must not
    clobber buf — the exact case value-level replace_operand cannot express."""
    load, alloca = _single_cell_load()
    token = BlockArgument(name="token", type=Nil())
    block = dgen.Block(result=load, args=[token], captures=[alloca])

    mem_edges = [u for u in uses_of(alloca, within=block) if u.field == "mem"]
    assert rebind_all(mem_edges, token) == 1
    assert load.mem is token
    assert load.buf is alloca
    # alloca is still used as buf, so its capture survives the prune.
    assert any(c is alloca for c in block.captures)


def test_rebind_repairs_nested_captures():
    """Rebinding a use inside a nested block adds the new value to that
    block's captures and prunes the old value's stale capture."""
    outer_arg = BlockArgument(name="o", type=Index())
    replacement = BlockArgument(name="r", type=Index())
    doubled = algebra.AddOp(left=outer_arg, right=outer_arg, type=Index())
    inner = dgen.Block(result=doubled, captures=[outer_arg])
    label = goto.LabelOp(name="l", initial_arguments=pack([]), body=inner)
    keep_alive = ChainOp(result=replacement, effect=label, type=Index())
    outer = dgen.Block(result=keep_alive, args=[outer_arg, replacement])

    count = rebind_all(uses_of(outer_arg, within=outer), replacement)
    assert count == 2
    assert doubled.left is replacement and doubled.right is replacement
    assert any(c is replacement for c in inner.captures)
    assert not any(c is outer_arg for c in inner.captures)


def test_prune_keeps_capture_needed_by_child_block():
    """A block whose child still captures the old value keeps its own
    capture — the child's declared dependency must stay satisfiable."""
    outer_arg = BlockArgument(name="o", type=Index())
    inner_use = algebra.AddOp(left=outer_arg, right=outer_arg, type=Index())
    inner = dgen.Block(result=inner_use, captures=[outer_arg])
    label = goto.LabelOp(name="l", initial_arguments=pack([]), body=inner)
    direct_use = ChainOp(result=outer_arg, effect=label, type=Index())
    middle = dgen.Block(result=direct_use, captures=[outer_arg])

    # Rebind only the direct edge in the middle block, not the nested one.
    replacement = BlockArgument(name="r", type=Index())
    middle.args = [replacement]
    direct = [u for u in uses_of(outer_arg, within=middle) if u.owner is direct_use]
    assert rebind_all(direct, replacement) == 1
    # The nested block still captures outer_arg, so middle must keep it.
    assert any(c is outer_arg for c in middle.captures)


def test_pack_elements_rebind_independently():
    a = BlockArgument(name="a", type=Index())
    b = BlockArgument(name="b", type=Index())
    bundle = pack([a, b, a])
    assert isinstance(bundle, PackOp)
    block = dgen.Block(result=bundle, args=[a, b])

    last = [u for u in uses_of(a, within=block) if u.field == "values[2]"]
    assert rebind_all(last, b) == 1
    assert bundle.values[0] is a
    assert bundle.values[2] is b


def test_into_predicate_bounds_descent():
    """A descent predicate keeps the walk at one structural level — e.g.
    a nested label treated as an opaque leaf."""
    outer_arg = BlockArgument(name="o", type=Index())
    inner_use = algebra.AddOp(left=outer_arg, right=outer_arg, type=Index())
    inner = dgen.Block(result=inner_use, captures=[outer_arg])
    label = goto.LabelOp(name="l", initial_arguments=pack([]), body=inner)
    outer = dgen.Block(result=label, args=[outer_arg])

    bounded = list(
        uses_of(
            outer_arg, within=outer, into=lambda op: not isinstance(op, goto.LabelOp)
        )
    )
    assert bounded == []
    unbounded = list(uses_of(outer_arg, within=outer))
    assert len(unbounded) == 2
