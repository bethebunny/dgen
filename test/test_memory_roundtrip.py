"""Round-trip tests for memory dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_stack_allocate():
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_heap_allocate():
    ir = strip_prefix("""
        | import function
        | import index
        | import memory
        | import number
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %0 : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_load_store_with_mem():
    """Load and store ops carry a mem operand for explicit memory ordering."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %val : index.Index = 42
        |     %st : Nil = memory.store(%alloc, %val, %alloc)
        |     %ld : index.Index = memory.load(%st, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_offset():
    ir = strip_prefix("""
        | import function
        | import index
        | import memory
        | import number
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %idx : index.Index = 3
        |     %ptr : memory.Reference<number.Float64> = memory.offset(%alloc, %idx)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_deallocate():
    ir = strip_prefix("""
        | import function
        | import index
        | import memory
        | import number
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %n : index.Index = 1
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %dealloc : Nil = memory.deallocate(%alloc, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_load_store_chain():
    """A sequence of loads and stores chained via mem for ordering."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %st0 : Nil = memory.store(%alloc, %zero, %alloc)
        |     %ld0 : index.Index = memory.load(%st0, %alloc)
        |     %one : index.Index = 1
        |     %st1 : Nil = memory.store(%ld0, %one, %alloc)
        |     %ld1 : index.Index = memory.load(%st1, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_offset_load_store():
    """Offset + load/store: typical array access pattern."""
    ir = strip_prefix("""
        | import function
        | import index
        | import memory
        | import number
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %idx : index.Index = 5
        |     %ptr : memory.Reference<number.Float64> = memory.offset(%alloc, %idx)
        |     %val : number.Float64 = 3.14
        |     %st : Nil = memory.store(%ptr, %val, %ptr)
        |     %ld : number.Float64 = memory.load(%st, %ptr)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_mem_from_for_loop():
    """The mem operand can come from a for loop result."""
    ir = strip_prefix("""
        | import control_flow
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %loop : Nil = control_flow.for<0, 10>([]) body(%iv: index.Index) captures(%alloc):
        |         %cur : index.Index = memory.load(%alloc, %alloc)
        |         %_ : Nil = memory.store(%cur, %iv, %alloc)
        |     %ld : index.Index = memory.load(%loop, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_mem_from_if_else():
    """The mem operand can come from an if/else result."""
    ir = strip_prefix("""
        | import control_flow
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %cond : index.Index = 1
        |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |         %ten : index.Index = 10
        |         %_ : Nil = memory.store(%alloc, %ten, %alloc)
        |     else_body() captures(%alloc):
        |         %twenty : index.Index = 20
        |         %_ : Nil = memory.store(%alloc, %twenty, %alloc)
        |     %ld : index.Index = memory.load(%if, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
