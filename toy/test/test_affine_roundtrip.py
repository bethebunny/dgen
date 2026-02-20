"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.dialects import affine
from toy.test.helpers import strip_prefix


def test_roundtrip_alloc():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<2x3>)
        |     %_ : () = affine.dealloc(%0)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<3>)
        |     %1 : f64 = constant(1.0)
        |     %2 : index = constant(0)
        |     %_ : () = affine.store(%1, %0, [%2])
        |     %3 : () = affine.load(%0, [%2])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : f64 = constant(2.5)
        |     %1 : f64 = constant(3.0)
        |     %2 : () = affine.mul_f(%0, %1)
        |     %3 : () = affine.add_f(%0, %1)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 : index = constant(42)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<3>)
        |     %_ : () = affine.print_memref(%0)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<3>)
        |     %_ : () = affine.for(0, 3) (%i0: index):
        |         %1 : f64 = constant(1.0)
        |         %2 : index = constant(0)
        |         %_ : () = affine.store(%1, %0, [%2])
        |     %_ : () = affine.print_memref(%0)
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<2x3>)
        |     %_ : () = affine.for(0, 2) (%i0: index):
        |         %_ : () = affine.for(0, 3) (%i1: index):
        |             %1 : f64 = constant(1.0)
        |             %2 : index = constant(0)
        |             %_ : () = affine.store(%1, %0, [%2, %2])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 : f64 = constant(1.0)
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : () = affine.alloc(<2x3>)
        |     %1 : f64 = constant(5.0)
        |     %2 : index = constant(0)
        |     %3 : index = constant(1)
        |     %_ : () = affine.store(%1, %0, [%2, %3])
        |     %4 : () = affine.load(%0, [%2, %3])
        |     %_ : () = return()
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
