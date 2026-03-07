"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.test.helpers import strip_prefix


def test_roundtrip_alloc():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[2, 3], F64> = affine.alloc([2, 3])
        |     %_ : Nil = affine.dealloc(%0)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_store_load():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[3], F64> = affine.alloc([3])
        |     %1 : F64 = 1.0
        |     %2 : Index = 0
        |     %_ : Nil = affine.store(%1, %0, [%2])
        |     %3 : Nil = affine.load(%0, [%2])
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_arith():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : F64 = 2.5
        |     %1 : F64 = 3.0
        |     %2 : Nil = affine.mul_f(%0, %1)
        |     %3 : Nil = affine.add_f(%0, %1)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() ():
        |     %0 : Index = 42
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[3], F64> = affine.alloc([3])
        |     %_ : Nil = affine.print_memref(%0)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[3], F64> = affine.alloc([3])
        |     %_ : Nil = affine.for<0, 3>() (%i0: Index):
        |         %1 : F64 = 1.0
        |         %2 : Index = 0
        |         %_ : Nil = affine.store(%1, %0, [%2])
        |     %_ : Nil = affine.print_memref(%0)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[2, 3], F64> = affine.alloc([2, 3])
        |     %_ : Nil = affine.for<0, 2>() (%i0: Index):
        |         %_ : Nil = affine.for<0, 3>() (%i1: Index):
        |             %1 : F64 = 1.0
        |             %2 : Index = 0
        |             %_ : Nil = affine.store(%1, %0, [%2, %2])
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | %f : Nil = function<Nil>() ():
        |     %0 : F64 = 1.0
        |     %_ : Nil = return(%0)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %0 : affine.MemRef<[2, 3], F64> = affine.alloc([2, 3])
        |     %1 : F64 = 5.0
        |     %2 : Index = 0
        |     %3 : Index = 1
        |     %_ : Nil = affine.store(%1, %0, [%2, %3])
        |     %4 : Nil = affine.load(%0, [%2, %3])
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_ssa_in_op_arg():
    """SSA value used as an op argument where a literal list would normally go."""
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %shape : affine.Shape<2> = [2, 3]
        |     %0 : affine.MemRef<[2, 3], F64> = affine.alloc(%shape)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_roundtrip_ssa_in_type_param():
    """SSA value used inside a type parameter position."""
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %shape : affine.Shape<2> = [2, 3]
        |     %0 : affine.MemRef<%shape, F64> = affine.alloc(%shape)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_ssa_shape_through_lowering():
    """SSA shape reference form works through affine-to-LLVM lowering."""
    ir = strip_prefix("""
        | import affine
        |
        | %f : Nil = function<Nil>() ():
        |     %shape : affine.Shape<2> = [2, 3]
        |     %0 : affine.MemRef<[2, 3], F64> = affine.alloc(%shape)
        |     %1 : F64 = 1.0
        |     %2 : Index = 0
        |     %_ : Nil = affine.store(%1, %0, [%2, %2])
        |     %3 : Nil = affine.load(%0, [%2, %2])
        |     %_ : Nil = affine.dealloc(%0)
        |     %_ : Nil = return(Nil)
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir

    llvm_module = lower_to_llvm(module)
    result = asm.format(llvm_module)
    assert "llvm.alloca<6>()" in result
