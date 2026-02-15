"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from toy_python.dialects import affine
from toy_python import asm


def test_roundtrip_alloc():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<2x3>)\n"
        "    Dealloc(%0)\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<3>)\n"
        "    %1 = ArithConstant(1.0)\n"
        "    AffineStore(%1, %0, [i0])\n"
        "    %2 = AffineLoad(%0, [i0])\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = ArithConstant(2.5)\n"
        "    %1 = ArithConstant(3.0)\n"
        "    %2 = MulF(%0, %1)\n"
        "    %3 = AddF(%0, %1)\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = IndexConstant(42)\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<3>)\n"
        "    PrintMemRef(%0)\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<3>)\n"
        "    AffineFor(%i0, 0, 3):\n"
        "        %1 = ArithConstant(1.0)\n"
        "        AffineStore(%1, %0, [i0])\n"
        "    PrintMemRef(%0)\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<2x3>)\n"
        "    AffineFor(%i0, 0, 2):\n"
        "        AffineFor(%i1, 0, 3):\n"
        "            %1 = ArithConstant(1.0)\n"
        "            AffineStore(%1, %0, [i0, i1])\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = ArithConstant(1.0)\n"
        "    Return(%0)\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = (
        "%f = function () -> ():\n"
        "    %0 = Alloc(<2x3>)\n"
        "    %1 = ArithConstant(5.0)\n"
        "    AffineStore(%1, %0, [i0, i1])\n"
        "    %2 = AffineLoad(%0, [i0, i1])\n"
        "    Return()\n"
    )
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir
