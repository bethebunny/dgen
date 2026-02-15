"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from toy_python.asm.parser import parse_module
from toy_python.dialects import affine
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_roundtrip_alloc():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     %_ = affine.dealloc(%0)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     %1 = affine.arith_constant(1.0)
        |     %_ = affine.store(%1, %0, [i0])
        |     %2 = affine.load(%0, [i0])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.arith_constant(2.5)
        |     %1 = affine.arith_constant(3.0)
        |     %2 = affine.mul_f(%0, %1)
        |     %3 = affine.add_f(%0, %1)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.index_constant(42)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     %_ = affine.print_memref(%0)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     %_ = affine.for(%i0, 0, 3):
        |         %1 = affine.arith_constant(1.0)
        |         %_ = affine.store(%1, %0, [i0])
        |     %_ = affine.print_memref(%0)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     %_ = affine.for(%i0, 0, 2):
        |         %_ = affine.for(%i1, 0, 3):
        |             %1 = affine.arith_constant(1.0)
        |             %_ = affine.store(%1, %0, [i0, i1])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.arith_constant(1.0)
        |     %_ = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     %1 = affine.arith_constant(5.0)
        |     %_ = affine.store(%1, %0, [i0, i1])
        |     %2 = affine.load(%0, [i0, i1])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
