"""Round-trip tests for LLVM dialect: construct -> asm -> parse -> asm."""

from toy_python.dialects import llvm
from toy_python import asm


def test_roundtrip_alloca():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloca(3)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_gep_load_store():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloca(6)\n"
        "    %1 = IConst(0)\n"
        "    %2 = Gep(%0, %1)\n"
        "    %3 = FConst(1.0)\n"
        "    Store(%3, %2)\n"
        "    %4 = Load(%2)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_fadd_fmul():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = FConst(1.0)\n"
        "    %1 = FConst(2.0)\n"
        "    %2 = FAdd(%0, %1)\n"
        "    %3 = FMul(%0, %1)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_add_mul_int():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = IConst(3)\n"
        "    %1 = IConst(4)\n"
        "    %2 = Add(%0, %1)\n"
        "    %3 = Mul(%0, %1)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_icmp_condbr():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = IConst(0)\n"
        "    %1 = IConst(10)\n"
        "    %cmp = Icmp(slt, %0, %1)\n"
        "    CondBr(%cmp, loop_body, loop_exit)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_label_br():
    ir = (
        "%f = function () -> ():\n"
        "    Br(loop_header)\n"
        "    Label(loop_header)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_phi():
    ir = (
        "%f = function () -> ():\n"
        "    %i0 = Phi([%init, %next], [entry, loop_body])\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_with_result():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Call(@foo, [%a, %b])\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_call_void():
    ir = (
        "%f = function () -> ():\n"
        "    Call(@print_memref, [%ptr, %size])\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = FConst(42.0)\n"
        "    Return(%0)\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_loop_pattern():
    """Full loop pattern: br, label, phi, icmp, condbr, body, increment."""
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloca(3)\n"
        "    %init = IConst(0)\n"
        "    Br(loop_header0)\n"
        "    Label(loop_header0)\n"
        "    %i0 = Phi([%init, %next0], [entry, loop_body0])\n"
        "    %hi = IConst(3)\n"
        "    %cmp = Icmp(slt, %i0, %hi)\n"
        "    CondBr(%cmp, loop_body0, loop_exit0)\n"
        "    Label(loop_body0)\n"
        "    %val = FConst(1.0)\n"
        "    %ptr = Gep(%0, %i0)\n"
        "    Store(%val, %ptr)\n"
        "    %one = IConst(1)\n"
        "    %next0 = Add(%i0, %one)\n"
        "    Br(loop_header0)\n"
        "    Label(loop_exit0)\n"
        "    Return()\n"
    )
    module = llvm.parse_llvm_module(ir)
    assert asm.format(module) == ir
