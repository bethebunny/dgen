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
    # Note: return conflicts (both dialects) — last wins (llvm)
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloca(6)\n"
        "    %1 = fconst(1.0)\n"
        "    store(%1, %0)\n"
        "    return()\n"
    )
    module = parse_module(ir, ops=ops, keywords=keywords, types=types)
    assert asm.format(module) == ir


def test_merged_llvm_full_loop():
    """Full LLVM loop pattern parsed with merged tables."""
    ops, keywords, types = _merge_tables(affine, llvm)
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloca(3)\n"
        "    %init = iconst(0)\n"
        "    br(loop_header)\n"
        "    label(loop_header)\n"
        "    %i0 = phi([%init, %next], [entry, loop_body])\n"
        "    %hi = iconst(3)\n"
        "    %cmp = icmp(slt, %i0, %hi)\n"
        "    cond_br(%cmp, loop_body, loop_exit)\n"
        "    label(loop_body)\n"
        "    %one = iconst(1)\n"
        "    %next = add(%i0, %one)\n"
        "    br(loop_header)\n"
        "    label(loop_exit)\n"
        "    return()\n"
    )
    module = parse_module(ir, ops=ops, keywords=keywords, types=types)
    assert asm.format(module) == ir
