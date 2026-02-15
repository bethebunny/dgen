"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from toy_python.dialects import affine
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def test_roundtrip_alloc():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     affine.dealloc(%0)
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     %1 = affine.arith_constant(1.0)
        |     affine.affine_store(%1, %0, [i0])
        |     %2 = affine.affine_load(%0, [i0])
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.arith_constant(2.5)
        |     %1 = affine.arith_constant(3.0)
        |     %2 = affine.mul_f(%0, %1)
        |     %3 = affine.add_f(%0, %1)
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.index_constant(42)
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     affine.print_memref(%0)
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<3>)
        |     affine.affine_for(%i0, 0, 3):
        |         %1 = affine.arith_constant(1.0)
        |         affine.affine_store(%1, %0, [i0])
        |     affine.print_memref(%0)
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     affine.affine_for(%i0, 0, 2):
        |         affine.affine_for(%i1, 0, 3):
        |             %1 = affine.arith_constant(1.0)
        |             affine.affine_store(%1, %0, [i0, i1])
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.arith_constant(1.0)
        |     return(%0)
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        | from builtin import function, return
        | import affine
        |
        | %f = function () -> ():
        |     %0 = affine.alloc(<2x3>)
        |     %1 = affine.arith_constant(5.0)
        |     affine.affine_store(%1, %0, [i0, i1])
        |     %2 = affine.affine_load(%0, [i0, i1])
        |     return()
    """)
    module = affine.parse_affine_module(ir)
    assert asm.format(module) == ir
