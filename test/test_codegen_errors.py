"""Tests for codegen error handling."""

import pytest

from dgen.asm.parser import parse
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import strip_prefix


def test_unhandled_op_raises():
    """Codegen raises ValueError for ops it cannot emit, not silent drop."""
    ir = strip_prefix("""
        | import index
        | import ndbuffer
        | import number
        | %a : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(1)>([2]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<index.Index(1)>([2]))
    """)
    value = parse(ir)
    # ndbuffer.alloc has no lowering in codegen (needs NDBufferToMemory first),
    # so it should raise, not silently drop it.
    with pytest.raises(ValueError, match="ndbuffer.*alloc"):
        Compiler([], LLVMCodegen()).compile(value)


def test_empty_non_label_group_in_mixed_block():
    """_make_synth handles empty ops lists without crashing.

    When _separate produces a group with no non-label ops, _make_synth
    must not reference an undefined `Nil` — it should use `builtin.Nil()`.
    """
    # A function with a label whose body has only other labels (no plain ops)
    # triggers _make_synth with potentially empty groups.
    ir = strip_prefix("""
        | import algebra
        | import goto
        | import index
        | %lbl : goto.Label = goto.label([]) body<%self: goto.Label, %exit: goto.Label>(%iv: index.Index):
        |     %one : index.Index = 1
        |     %next : index.Index = algebra.add(%iv, %one)
        |     %br : Nil = goto.branch<%self>([%next])
    """)
    value = parse(ir)
    exe = Compiler(
        [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()], LLVMCodegen()
    ).compile(value)
    assert "define" in exe.ir
