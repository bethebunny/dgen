"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from toy_python.dialects import llvm
from toy_python import asm


def test_roundtrip_alloca():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloca(3)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_gep_load_store():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloca(6)\n"
        "    %1 = iconst(0)\n"
        "    %2 = gep(%0, %1)\n"
        "    %3 = fconst(1.0)\n"
        "    store(%3, %2)\n"
        "    %4 = load(%2)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_fadd_fmul():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = fconst(1.0)\n"
        "    %1 = fconst(2.0)\n"
        "    %2 = fadd(%0, %1)\n"
        "    %3 = fmul(%0, %1)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add_mul_int():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = iconst(3)\n"
        "    %1 = iconst(4)\n"
        "    %2 = add(%0, %1)\n"
        "    %3 = mul(%0, %1)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_icmp_condbr():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = iconst(0)\n"
        "    %1 = iconst(10)\n"
        "    %cmp = icmp(slt, %0, %1)\n"
        "    cond_br(%cmp, loop_body, loop_exit)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_label_br():
    ir = (
        "%f = function () -> ():\n"
        "    br(loop_header)\n"
        "    label(loop_header)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_phi():
    ir = (
        "%f = function () -> ():\n"
        "    %i0 = phi([%init, %next], [entry, loop_body])\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_with_result():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = call(@foo, [%a, %b])\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_void():
    ir = (
        "%f = function () -> ():\n"
        "    call(@print_memref, [%ptr, %size])\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = fconst(42.0)\n"
        "    return(%0)\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_loop_pattern():
    """Full loop pattern: br, label, phi, icmp, condbr, body, increment."""
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloca(3)\n"
        "    %init = iconst(0)\n"
        "    br(loop_header0)\n"
        "    label(loop_header0)\n"
        "    %i0 = phi([%init, %next0], [entry, loop_body0])\n"
        "    %hi = iconst(3)\n"
        "    %cmp = icmp(slt, %i0, %hi)\n"
        "    cond_br(%cmp, loop_body0, loop_exit0)\n"
        "    label(loop_body0)\n"
        "    %val = fconst(1.0)\n"
        "    %ptr = gep(%0, %i0)\n"
        "    store(%val, %ptr)\n"
        "    %one = iconst(1)\n"
        "    %next0 = add(%i0, %one)\n"
        "    br(loop_header0)\n"
        "    label(loop_exit0)\n"
        "    return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir
