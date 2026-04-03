"""Tests for codegen error handling."""

import pytest

from dgen.asm.parser import parse_module
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.testing import strip_prefix


def test_unhandled_op_raises():
    """Codegen raises ValueError for ops it cannot emit, not silent drop."""
    ir = strip_prefix("""
        | import function
        | import ndbuffer
        | import number
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %a : ndbuffer.NDBuffer<ndbuffer.Shape<1>([2]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([2]))
    """)
    module = parse_module(ir)
    # ndbuffer.alloc has no lowering in codegen (needs NDBufferToMemory first),
    # so it should raise, not silently drop it.
    with pytest.raises(KeyError, match="ndbuffer.alloc"):
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
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %lbl : goto.Label = goto.label([]) body<%self: goto.Label, %exit: goto.Label>(%iv: index.Index):
        |         %one : index.Index = 1
        |         %next : index.Index = algebra.add(%iv, %one)
        |         %br : Nil = goto.branch<%self>([%next])
    """)
    module = parse_module(ir)
    from dgen.passes.control_flow_to_goto import ControlFlowToGoto

    exe = Compiler([ControlFlowToGoto()], LLVMCodegen()).compile(module)
    assert "define" in exe.ir
