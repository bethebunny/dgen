"""Round-trip tests for memory dialect: construct -> asm -> parse -> asm."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.dialects import index as _index  # noqa: F401 — register index dialect
from dgen.dialects import memory as _memory  # noqa: F401 — register memory dialect
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_stack_allocate():
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_heap_allocate():
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %0 : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_load_store_with_mem():
    """Load and store ops carry a mem operand for explicit memory ordering."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %val : index.Index = 42
        |     %st : Nil = memory.store(%alloc, %val, %alloc)
        |     %ld : index.Index = memory.load(%st, %alloc)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_offset():
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %idx : index.Index = 3
        |     %ptr : memory.Reference<number.Float64> = memory.offset(%alloc, %idx)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_deallocate():
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %n : index.Index = 1
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %dealloc : Nil = memory.deallocate(%alloc)
        |     %_ : Nil = chain(%alloc, %dealloc)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_load_store_chain():
    """A sequence of loads and stores chained via mem for ordering."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %st0 : Nil = memory.store(%alloc, %zero, %alloc)
        |     %ld0 : index.Index = memory.load(%st0, %alloc)
        |     %one : index.Index = 1
        |     %st1 : Nil = memory.store(%ld0, %one, %alloc)
        |     %ld1 : index.Index = memory.load(%st1, %alloc)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_offset_load_store():
    """Offset + load/store: typical array access pattern."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %n : index.Index = 10
        |     %alloc : memory.Reference<number.Float64> = memory.heap_allocate<number.Float64>(%n)
        |     %idx : index.Index = 5
        |     %ptr : memory.Reference<number.Float64> = memory.offset(%alloc, %idx)
        |     %val : number.Float64 = 3.14
        |     %st : Nil = memory.store(%ptr, %val, %ptr)
        |     %ld : number.Float64 = memory.load(%st, %ptr)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_mem_from_control_flow():
    """The mem operand can come from a non-memory op (e.g. control flow result)."""
    ir = strip_prefix("""
        | import function
        | import memory
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %st : Nil = memory.store(%alloc, %zero, %alloc)
        |     %chained : memory.Reference<index.Index> = chain(%alloc, %st)
        |     %ld : index.Index = memory.load(%chained, %alloc)
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
