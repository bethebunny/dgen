"""Tests for Block.captures: explicit capture support."""

import pytest

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.dialects import goto, llvm
from dgen.passes.pass_ import Rewriter
from dgen.testing import assert_ir_equivalent, strip_prefix
from dgen.verify import ClosedBlockError, verify_closed_blocks


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_captures():
    ir = strip_prefix("""
        | import function
        | import goto
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %loop : goto.Label = goto.label() body<%self: goto.Label>(%i: Index) captures(%x):
        |         %zero : Index = 0
    """)
    module = parse_module(ir)
    (func,) = module.ops
    label = func.body.ops[0]
    assert isinstance(label, goto.LabelOp)
    assert len(label.body.captures) == 1
    assert label.body.captures[0].name == "x"
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_empty_captures():
    """Blocks without captures should not emit 'captures()' in ASM."""
    ir = strip_prefix("""
        | import function
        | import goto
        |
        | %loop : goto.Label = goto.label() body(%i: Index):
        |     %zero : Index = 0
    """)
    module = parse_module(ir)
    formatted = asm.format(module)
    assert "captures" not in formatted


# ---------------------------------------------------------------------------
# Verifier: correct captures pass
# ---------------------------------------------------------------------------


def test_verify_captured_block_arg_in_scope():
    """A block that captures an outer BlockArgument passes verification."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %inner : goto.Label = goto.label() body() captures(%x):
        |         %0 : Index = 0
        |         %1 : llvm.Int<64> = llvm.add(%x, %0)
    """)
    verify_closed_blocks(parse_module(ir))


def test_verify_captured_block_parameter():
    """A nested block can capture a block parameter from an enclosing block."""
    ir = strip_prefix("""
        | import function
        | import goto
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %header : goto.Label = goto.label() body<%self: goto.Label>():
        |         %body : goto.Label = goto.label() body() captures(%self):
        |             %0 : Nil = goto.branch<%self>([])
    """)
    verify_closed_blocks(parse_module(ir))


def test_verify_ambient_op_without_capture_passes():
    """Ambient ops (no block-arg deps) don't need capturing."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %inner : goto.Label = goto.label() body():
        |         %0 : Index = 42
        |         %1 : Index = 0
        |         %2 : llvm.Int<64> = llvm.add(%0, %1)
    """)
    verify_closed_blocks(parse_module(ir))


# ---------------------------------------------------------------------------
# Verifier: missing captures caught
# ---------------------------------------------------------------------------


def test_verify_missing_capture_of_block_arg():
    """A block that uses an outer BlockArgument without capturing it fails."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %inner : goto.Label = goto.label() body():
        |         %0 : Index = 0
        |         %1 : llvm.Int<64> = llvm.add(%x, %0)
    """)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(parse_module(ir))


def test_verify_missing_capture_of_block_parameter():
    """A block that uses an enclosing block parameter without capturing it fails."""
    ir = strip_prefix("""
        | import function
        | import goto
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %header : goto.Label = goto.label() body<%self: goto.Label>():
        |         %body : goto.Label = goto.label() body():
        |             %0 : Nil = goto.branch<%self>([])
    """)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(parse_module(ir))


# ---------------------------------------------------------------------------
# Verifier: captures must chain
# ---------------------------------------------------------------------------


def test_verify_chained_captures():
    """Both middle and inner blocks capture an outer value — passes."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %mid : goto.Label = goto.label() body() captures(%x):
        |         %inner : goto.Label = goto.label() body() captures(%x):
        |             %0 : Index = 0
        |             %1 : llvm.Int<64> = llvm.add(%x, %0)
    """)
    verify_closed_blocks(parse_module(ir))


def test_verify_unchained_capture_fails():
    """Inner block captures a value the middle block doesn't — fails."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %mid : goto.Label = goto.label() body():
        |         %inner : goto.Label = goto.label() body() captures(%x):
        |             %0 : Index = 0
        |             %1 : llvm.Int<64> = llvm.add(%x, %0)
    """)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(parse_module(ir))


# ---------------------------------------------------------------------------
# replace_uses maintains captures
# ---------------------------------------------------------------------------


def test_replace_uses_updates_captures():
    """replace_uses swaps values in the captures list and inner ops."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%old: Index, %new: Index):
        |     %inner : goto.Label = goto.label() body() captures(%old):
        |         %0 : Index = 0
        |         %1 : llvm.Int<64> = llvm.add(%old, %0)
    """)
    module = parse_module(ir)
    func = module.functions[0]
    old_arg, new_arg = func.body.args

    Rewriter(func.body).replace_uses(old_arg, new_arg)

    label = func.body.ops[0]
    assert isinstance(label, goto.LabelOp)
    assert new_arg in label.body.captures
    assert old_arg not in label.body.captures
    add_op = label.body.ops[-1]
    assert isinstance(add_op, llvm.AddOp)
    assert add_op.lhs is new_arg


def test_replace_uses_updates_chained_captures():
    """replace_uses propagates through chained captures and stays well-formed."""
    ir = strip_prefix("""
        | import function
        | import goto
        | import llvm
        |
        | %f : function.Function<()> = function.function<Nil>() body(%old: Index, %new: Index):
        |     %mid : goto.Label = goto.label() body() captures(%old):
        |         %inner : goto.Label = goto.label() body() captures(%old):
        |             %0 : Index = 0
        |             %1 : llvm.Int<64> = llvm.add(%old, %0)
    """)
    module = parse_module(ir)
    func = module.functions[0]
    old_arg, new_arg = func.body.args

    Rewriter(func.body).replace_uses(old_arg, new_arg)

    mid_label = func.body.ops[0]
    assert new_arg in mid_label.body.captures
    inner_label = mid_label.body.ops[0]
    assert new_arg in inner_label.body.captures
    assert inner_label.body.ops[-1].lhs is new_arg

    verify_closed_blocks(module)
