"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from toy_python.dialects import affine
from toy_python import asm


def test_roundtrip_alloc():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<2x3>)\n"
        "    dealloc(%0)\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<3>)\n"
        "    %1 = arith_constant(1.0)\n"
        "    affine_store(%1, %0, [i0])\n"
        "    %2 = affine_load(%0, [i0])\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = arith_constant(2.5)\n"
        "    %1 = arith_constant(3.0)\n"
        "    %2 = mul_f(%0, %1)\n"
        "    %3 = add_f(%0, %1)\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = index_constant(42)\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<3>)\n"
        "    print_memref(%0)\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<3>)\n"
        "    affine_for(%i0, 0, 3):\n"
        "        %1 = arith_constant(1.0)\n"
        "        affine_store(%1, %0, [i0])\n"
        "    print_memref(%0)\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<2x3>)\n"
        "    affine_for(%i0, 0, 2):\n"
        "        affine_for(%i1, 0, 3):\n"
        "            %1 = arith_constant(1.0)\n"
        "            affine_store(%1, %0, [i0, i1])\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = arith_constant(1.0)\n"
        "    return(%0)\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = alloc(<2x3>)\n"
        "    %1 = arith_constant(5.0)\n"
        "    affine_store(%1, %0, [i0, i1])\n"
        "    %2 = affine_load(%0, [i0, i1])\n"
        "    return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir
