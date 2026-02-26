"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_roundtrip_alloc():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([2, 3], f64) = affine.alloc([2, 3])
        |     %_ : () = affine.dealloc(%0)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([3], f64) = affine.alloc([3])
        |     %1 : f64 = 1.0
        |     %2 : index = 0
        |     %_ : () = affine.store(%1, %0, [%2])
        |     %3 : () = affine.load(%0, [%2])
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : f64 = 2.5
        |     %1 : f64 = 3.0
        |     %2 : () = affine.mul_f(%0, %1)
        |     %3 : () = affine.add_f(%0, %1)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 : index = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([3], f64) = affine.alloc([3])
        |     %_ : () = affine.print_memref(%0)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([3], f64) = affine.alloc([3])
        |     %_ : () = affine.for(0, 3) (%i0: index):
        |         %1 : f64 = 1.0
        |         %2 : index = 0
        |         %_ : () = affine.store(%1, %0, [%2])
        |     %_ : () = affine.print_memref(%0)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([2, 3], f64) = affine.alloc([2, 3])
        |     %_ : () = affine.for(0, 2) (%i0: index):
        |         %_ : () = affine.for(0, 3) (%i1: index):
        |             %1 : f64 = 1.0
        |             %2 : index = 0
        |             %_ : () = affine.store(%1, %0, [%2, %2])
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f = function () -> ():
        |     %0 : f64 = 1.0
        |     %_ : () = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %0 : affine.MemRef([2, 3], f64) = affine.alloc([2, 3])
        |     %1 : f64 = 5.0
        |     %2 : index = 0
        |     %3 : index = 1
        |     %_ : () = affine.store(%1, %0, [%2, %3])
        |     %4 : () = affine.load(%0, [%2, %3])
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_ssa_in_op_arg():
    """SSA value used as an op argument where a literal list would normally go."""
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %shape : affine.Shape(2) = [2, 3]
        |     %0 : affine.MemRef([2, 3], f64) = affine.alloc(%shape)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_ssa_in_type_param():
    """SSA value used inside a type parameter position."""
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %shape : affine.Shape(2) = [2, 3]
        |     %0 : affine.MemRef(%shape, f64) = affine.alloc(%shape)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_ssa_shape_through_lowering():
    """SSA shape reference form works through affine-to-LLVM lowering."""
    ir = strip_prefix("""
        | import affine
        |
        | %f = function () -> ():
        |     %shape : affine.Shape(2) = [2, 3]
        |     %0 : affine.MemRef([2, 3], f64) = affine.alloc(%shape)
        |     %1 : f64 = 1.0
        |     %2 : index = 0
        |     %_ : () = affine.store(%1, %0, [%2, %2])
        |     %3 : () = affine.load(%0, [%2, %2])
        |     %_ : () = affine.dealloc(%0)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir

    from toy.passes.affine_to_llvm import lower_to_llvm

    llvm_module = lower_to_llvm(module)
    result = asm.format(llvm_module)
    assert "llvm.alloca(6)" in result
