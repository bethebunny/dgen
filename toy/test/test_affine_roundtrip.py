"""Round-trip tests for affine dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from dgen.testing import assert_ir_equivalent
from toy.passes.affine_to_llvm import AffineToLLVMLowering
from toy.test.helpers import strip_prefix

_compiler = Compiler([], IdentityPass())


def lower_to_llvm(m: Module) -> Module:
    return AffineToLLVMLowering().run(m, _compiler)


def test_roundtrip_alloc():
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<2>([2, 3]), F64> = memory.alloc(memory.Shape<2>([2, 3]))
        |     %dealloc : Nil = memory.dealloc(%0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_store_load():
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<1>([3]), F64> = memory.alloc(memory.Shape<1>([3]))
        |     %1 : F64 = 1.0
        |     %2 : Index = 0
        |     %store : Nil = memory.store(%1, %0, [%2])
        |     %3 : F64 = memory.load(%0, [%2])
        |     %4 : F64 = chain(%3, %store)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_arith():
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : F64 = 2.5
        |     %1 : F64 = 3.0
        |     %2 : F64 = affine.mul_f(%0, %1)
        |     %3 : F64 = affine.add_f(%0, %1)
        |     %4 : F64 = chain(%2, %3)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : Index = 42
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<1>([3]), F64> = memory.alloc(memory.Shape<1>([3]))
        |     %print : Nil = memory.print_memref(%0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import function
        | import control_flow
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<1>([3]), F64> = memory.alloc(memory.Shape<1>([3]))
        |     %loop : Nil = control_flow.for<0, 3>([]) body(%i0: Index):
        |         %1 : F64 = 1.0
        |         %2 : Index = 0
        |         %_ : Nil = memory.store(%1, %0, [%2])
        |     %print : Nil = memory.print_memref(%0)
        |     %3 : Nil = chain(%print, %loop)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import function
        | import control_flow
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<2>([2, 3]), F64> = memory.alloc(memory.Shape<2>([2, 3]))
        |     %loop : Nil = control_flow.for<0, 2>([]) body(%i0: Index):
        |         %_ : Nil = control_flow.for<0, 3>([]) body(%i1: Index):
        |             %1 : F64 = 1.0
        |             %2 : Index = 0
        |             %_ : Nil = memory.store(%1, %0, [%2, %2])
        |     %3 : Nil = chain(%loop, %0)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : F64 = 1.0
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : memory.MemRef<memory.Shape<2>([2, 3]), F64> = memory.alloc(memory.Shape<2>([2, 3]))
        |     %1 : F64 = 5.0
        |     %2 : Index = 0
        |     %3 : Index = 1
        |     %store : Nil = memory.store(%1, %0, [%2, %3])
        |     %4 : F64 = memory.load(%0, [%2, %3])
        |     %5 : F64 = chain(%4, %store)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_ssa_in_op_arg():
    """SSA value used as an op argument where a literal list would normally go."""
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %shape : memory.Shape<2> = [2, 3]
        |     %0 : memory.MemRef<memory.Shape<2>([2, 3]), F64> = memory.alloc(%shape)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_ssa_in_type_param():
    """SSA value used inside a type parameter position."""
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %shape : memory.Shape<2> = [2, 3]
        |     %0 : memory.MemRef<%shape, F64> = memory.alloc(%shape)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_ssa_shape_through_lowering():
    """SSA shape reference form works through affine-to-LLVM lowering."""
    ir = strip_prefix("""
        |
        | import function
        |
        | %f : Nil = function.define<Nil>() body():
        |     %shape : memory.Shape<2> = [2, 3]
        |     %0 : memory.MemRef<memory.Shape<2>([2, 3]), F64> = memory.alloc(%shape)
        |     %1 : F64 = 1.0
        |     %2 : Index = 0
        |     %store : Nil = memory.store(%1, %0, [%2, %2])
        |     %3 : F64 = memory.load(%0, [%2, %2])
        |     %4 : F64 = chain(%3, %store)
        |     %dealloc : Nil = memory.dealloc(%0)
        |     %5 : Nil = chain(%dealloc, %4)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    llvm_module = lower_to_llvm(module)
    result = asm.format(llvm_module)
    assert 'llvm.call<"malloc">' in result
