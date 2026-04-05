"""Round-trip tests for structured dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.compiler import Compiler, IdentityPass
from dgen.testing import assert_ir_equivalent
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.testing import strip_prefix


def lower_to_llvm(m):
    return Compiler(
        [ControlFlowToGoto(), NDBufferToMemory(), MemoryToLLVM()],
        IdentityPass(),
    ).run(m)


def test_roundtrip_alloc():
    ir = strip_prefix("""
        |
        | import function
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<2>([2, 3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<2>([2, 3]))
        |     %dealloc : Nil = ndbuffer.dealloc(%0, %0)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_store_load():
    ir = strip_prefix("""
        |
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([3]))
        |     %1 : number.Float64 = 1.0
        |     %2 : index.Index = 0
        |     %store : Nil = ndbuffer.store(%0, %1, %0, [%2])
        |     %3 : number.Float64 = ndbuffer.load(%store, %0, [%2])
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_arith():
    ir = strip_prefix("""
        |
        | import algebra
        | import function
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : number.Float64 = 2.5
        |     %1 : number.Float64 = 3.0
        |     %2 : number.Float64 = algebra.multiply(%0, %1)
        |     %3 : number.Float64 = algebra.add(%0, %1)
        |     %4 : number.Float64 = chain(%2, %3)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_index_constant():
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : index.Index = 42
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_print_memref():
    ir = strip_prefix("""
        |
        | import function
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([3]))
        |     %print : Nil = ndbuffer.print_memref(%0)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_for_op():
    ir = strip_prefix("""
        | import control_flow
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([3]))
        |     %loop : Nil = control_flow.for<0, 3>([]) body(%i0: index.Index):
        |         %1 : number.Float64 = 1.0
        |         %2 : index.Index = 0
        |         %_ : Nil = ndbuffer.store(%0, %1, %0, [%2])
        |     %3 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([3]), number.Float64> = chain(%0, %loop)
        |     %print : Nil = ndbuffer.print_memref(%3)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_nested_for():
    ir = strip_prefix("""
        | import control_flow
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<2>([2, 3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<2>([2, 3]))
        |     %loop : Nil = control_flow.for<0, 2>([]) body(%i0: index.Index):
        |         %_ : Nil = control_flow.for<0, 3>([]) body(%i1: index.Index):
        |             %1 : number.Float64 = 1.0
        |             %2 : index.Index = 0
        |             %_ : Nil = ndbuffer.store(%0, %1, %0, [%2, %2])
        |     %3 : Nil = chain(%loop, %0)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_return_value():
    ir = strip_prefix("""
        | import function
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : number.Float64 = 1.0
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_multi_index_load_store():
    ir = strip_prefix("""
        |
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<2>([2, 3]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<2>([2, 3]))
        |     %1 : number.Float64 = 5.0
        |     %2 : index.Index = 0
        |     %3 : index.Index = 1
        |     %store : Nil = ndbuffer.store(%0, %1, %0, [%2, %3])
        |     %4 : number.Float64 = ndbuffer.load(%store, %0, [%2, %3])
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_ssa_in_op_arg():
    """SSA value used as an op argument where a literal list would normally go."""
    ir = strip_prefix("""
        |
        | import function
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %shape : ndbuffer.Shape<2> = [2, 3]
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<2>([2, 3]), number.Float64> = ndbuffer.alloc(%shape)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_ssa_in_type_param():
    """SSA value used inside a type parameter position."""
    ir = strip_prefix("""
        |
        | import function
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %shape : ndbuffer.Shape<2> = [2, 3]
        |     %0 : ndbuffer.NDBuffer<%shape, number.Float64> = ndbuffer.alloc(%shape)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_ssa_shape_through_lowering():
    """SSA shape reference form works through structured-to-LLVM lowering."""
    ir = strip_prefix("""
        |
        | import function
        | import index
        | import ndbuffer
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %shape : ndbuffer.Shape<2> = [2, 3]
        |     %0 : ndbuffer.NDBuffer<ndbuffer.Shape<2>([2, 3]), number.Float64> = ndbuffer.alloc(%shape)
        |     %1 : number.Float64 = 1.0
        |     %2 : index.Index = 0
        |     %store : Nil = ndbuffer.store(%0, %1, %0, [%2, %2])
        |     %3 : number.Float64 = ndbuffer.load(%store, %0, [%2, %2])
        |     %dealloc : Nil = ndbuffer.dealloc(%3, %0)
    """)
    module = parse(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))

    llvm_module = lower_to_llvm(module)
    result = asm.format(llvm_module)
    assert 'extern<"malloc">' in result
