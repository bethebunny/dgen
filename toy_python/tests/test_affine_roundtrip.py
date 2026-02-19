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
        |     %1 = constant(1.0) : f64
        |     %2 = constant(0) : index
        |     %_ = affine.store(%1, %0, [%2])
        |     %3 = affine.load(%0, [%2])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 = constant(2.5) : f64
        |     %1 = constant(3.0) : f64
        |     %2 = affine.mul_f(%0, %1)
        |     %3 = affine.add_f(%0, %1)
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 = constant(42) : index
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
        |     %_ = affine.for(0, 3) (%i0: index):
        |         %1 = constant(1.0) : f64
        |         %2 = constant(0) : index
        |         %_ = affine.store(%1, %0, [%2])
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
        |     %_ = affine.for(0, 2) (%i0: index):
        |         %_ = affine.for(0, 3) (%i1: index):
        |             %1 = constant(1.0) : f64
        |             %2 = constant(0) : index
        |             %_ = affine.store(%1, %0, [%2, %2])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 = constant(1.0) : f64
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
        |     %1 = constant(5.0) : f64
        |     %2 = constant(0) : index
        |     %3 = constant(1) : index
        |     %_ = affine.store(%1, %0, [%2, %3])
        |     %4 = affine.load(%0, [%2, %3])
        |     %_ = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
