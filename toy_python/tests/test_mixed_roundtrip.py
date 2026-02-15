"""Round-trip tests for mixed-dialect parsing using merged tables."""

from toy_python.dialects import affine, llvm
from toy_python.ir_parser import parse_module
from toy_python import asm


def _merge_tables(*dialects):
    """Merge OP_TABLE, KEYWORD_TABLE, TYPE_TABLE from multiple dialects."""
    ops, keywords, types = {}, {}, {}
    for d in dialects:
        ops.update(d.OP_TABLE)
        keywords.update(d.KEYWORD_TABLE)
        types.update(d.TYPE_TABLE)
    return ops, keywords, types


def test_affine_then_llvm():
    """Parse a function using affine ops, another using llvm ops."""
    ops, keywords, types = _merge_tables(affine, llvm)
    # Note: Return conflicts (both dialects) — last wins (llvm)
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloca(6)\n"
        "    %1 = FConst(1.0)\n"
        "    Store(%1, %0)\n"
        "    Return()\n"
    )
    module = parse_module(ir, ops=ops, keywords=keywords, types=types)
    assert asm.format(module) == ir


def test_merged_llvm_full_loop():
    """Full LLVM loop pattern parsed with merged tables."""
    ops, keywords, types = _merge_tables(affine, llvm)
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloca(3)\n"
        "    %init = IConst(0)\n"
        "    Br(loop_header)\n"
        "    Label(loop_header)\n"
        "    %i0 = Phi([%init, %next], [entry, loop_body])\n"
        "    %hi = IConst(3)\n"
        "    %cmp = Icmp(slt, %i0, %hi)\n"
        "    CondBr(%cmp, loop_body, loop_exit)\n"
        "    Label(loop_body)\n"
        "    %one = IConst(1)\n"
        "    %next = Add(%i0, %one)\n"
        "    Br(loop_header)\n"
        "    Label(loop_exit)\n"
        "    Return()\n"
    )
    module = parse_module(ir, ops=ops, keywords=keywords, types=types)
    assert asm.format(module) == ir
