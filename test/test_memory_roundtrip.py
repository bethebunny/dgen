"""Round-trip tests for memory dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_stack_allocate():
    ir = strip_prefix("""
        | import memory
        | import index
        |
        | %0 : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_heap_allocate():
    ir = strip_prefix("""
        | import index
        | import memory
        | import number
        |
        | %n : index.Index = 10
        | %0 : memory.Buffer<number.Float64> = memory.buffer_allocate<number.Float64>(%n)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_load_store_with_mem():
    """Load and store ops carry a mem operand for explicit memory ordering."""
    ir = strip_prefix("""
        | import memory
        | import index
        |
        | %alloc : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
        | %val : index.Index = 42
        | %st : Nil = memory.buffer_store(%alloc, %alloc, index.Index(0), %val)
        | %ld : index.Index = memory.buffer_load(%st, %alloc, index.Index(0))
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_deallocate():
    ir = strip_prefix("""
        | import index
        | import memory
        | import number
        |
        | %n : index.Index = 1
        | %alloc : memory.Buffer<number.Float64> = memory.buffer_allocate<number.Float64>(%n)
        | %dealloc : Nil = memory.buffer_deallocate(%alloc, %alloc)
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_load_store_chain():
    """A sequence of loads and stores chained via mem for ordering."""
    ir = strip_prefix("""
        | import memory
        | import index
        |
        | %alloc : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
        | %zero : index.Index = 0
        | %st0 : Nil = memory.buffer_store(%alloc, %alloc, index.Index(0), %zero)
        | %ld0 : index.Index = memory.buffer_load(%st0, %alloc, index.Index(0))
        | %one : index.Index = 1
        | %st1 : Nil = memory.buffer_store(%ld0, %alloc, index.Index(0), %one)
        | %ld1 : index.Index = memory.buffer_load(%st1, %alloc, index.Index(0))
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_mem_from_for_loop():
    """The mem operand can come from a for loop result."""
    ir = strip_prefix("""
        | import control_flow
        | import memory
        | import index
        |
        | %alloc : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
        | %loop : Nil = control_flow.for<index.Index(0), index.Index(10)>([]) body(%iv: index.Index) captures(%alloc):
        |     %cur : index.Index = memory.buffer_load(%alloc, %alloc, index.Index(0))
        |     %_ : Nil = memory.buffer_store(%cur, %alloc, index.Index(0), %iv)
        | %ld : index.Index = memory.buffer_load(%loop, %alloc, index.Index(0))
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_mem_from_if_else():
    """The mem operand can come from an if/else result."""
    ir = strip_prefix("""
        | import control_flow
        | import memory
        | import index
        |
        | %alloc : memory.Buffer<index.Index> = memory.buffer_stack_allocate<index.Index>(index.Index(1))
        | %cond : index.Index = 1
        | %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |     %ten : index.Index = 10
        |     %_ : Nil = memory.buffer_store(%alloc, %alloc, index.Index(0), %ten)
        | else_body() captures(%alloc):
        |     %twenty : index.Index = 20
        |     %_ : Nil = memory.buffer_store(%alloc, %alloc, index.Index(0), %twenty)
        | %ld : index.Index = memory.buffer_load(%if, %alloc, index.Index(0))
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))
