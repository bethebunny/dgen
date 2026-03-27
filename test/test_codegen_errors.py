"""Tests for codegen error handling."""

import pytest

from dgen.asm.parser import parse_module
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.testing import strip_prefix


def test_unhandled_op_raises():
    """Codegen raises ValueError for ops it cannot emit, not silent drop."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %y : index.Index = algebra.negate(%x)
    """)
    module = parse_module(ir)
    # algebra.negate has no lowering in AlgebraToLLVM, so it passes through
    # to codegen which should raise, not silently drop it.
    with pytest.raises(ValueError, match="unhandled op"):
        Compiler([], LLVMCodegen()).compile(module)


def test_empty_non_label_group_in_mixed_block():
    """_make_synth handles empty ops lists without crashing.

    When _separate produces a group with no non-label ops, _make_synth
    must not reference an undefined `Nil` — it should use `builtin.Nil()`.
    """
    # A function with a label whose body has only other labels (no plain ops)
    # triggers _make_synth with potentially empty groups.
    ir = strip_prefix("""
        | import algebra
        | import function
        | import goto
        | import index
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %lbl : goto.Label = goto.label([]) body<%self: goto.Label, %exit: goto.Label>(%iv: index.Index):
        |         %one : index.Index = 1
        |         %next : index.Index = algebra.add(%iv, %one)
        |         %br : Nil = goto.branch<%self>([%next])
    """)
    module = parse_module(ir)
    from toy.passes.control_flow_to_goto import ControlFlowToGoto

    exe = Compiler([ControlFlowToGoto()], LLVMCodegen()).compile(module)
    assert "define" in exe.ir
