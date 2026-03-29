"""Semantic tests for memory load/store with mem ordering.

These tests compile memory operations through the full pipeline and JIT-execute
them, verifying that the mem operand correctly enforces ordering. Each test is
designed so that incorrect ordering would produce the wrong result.
"""

from dgen.asm.parser import parse_module
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects import index as _index  # noqa: F401 — register index dialect
from dgen.dialects import memory as _memory  # noqa: F401 — register memory dialect
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.testing import strip_prefix


def _jit(ir: str, *args: object) -> object:
    """Parse IR, compile through memory pipeline, JIT-run, return JSON result."""
    module = parse_module(strip_prefix(ir))
    compiler: Compiler[Executable] = Compiler(
        [ControlFlowToGoto(), MemoryToLLVM()], LLVMCodegen()
    )
    exe = compiler.compile(module)
    return exe.run(*args).to_json()


def test_store_then_load_sees_stored_value():
    """Store 42, then load — must read back 42.

    Without mem ordering, the load could be scheduled before the store
    and read uninitialized memory.
    """
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %val : index.Index = 42
        |     %st : Nil = memory.store(%alloc, %val, %alloc)
        |     %ld : index.Index = memory.load(%st, %alloc)
    """)
        == 42
    )


def test_two_stores_last_wins():
    """Store 10, then store 20, then load — must read 20.

    The mem chain store1 → store2 → load ensures the second store
    overwrites the first. Without ordering, either value could appear.
    """
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v10 : index.Index = 10
        |     %st1 : Nil = memory.store(%alloc, %v10, %alloc)
        |     %v20 : index.Index = 20
        |     %st2 : Nil = memory.store(%st1, %v20, %alloc)
        |     %ld : index.Index = memory.load(%st2, %alloc)
    """)
        == 20
    )


def test_three_stores_last_wins():
    """Store 1, 2, 3 in sequence — load must read 3.

    A longer chain validates that mem threading works across
    multiple sequential stores.
    """
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v1 : index.Index = 1
        |     %s1 : Nil = memory.store(%alloc, %v1, %alloc)
        |     %v2 : index.Index = 2
        |     %s2 : Nil = memory.store(%s1, %v2, %alloc)
        |     %v3 : index.Index = 3
        |     %s3 : Nil = memory.store(%s2, %v3, %alloc)
        |     %ld : index.Index = memory.load(%s3, %alloc)
    """)
        == 3
    )


def test_read_modify_write():
    """Store 0, load, add 7, store back, load — must read 7.

    The load→add→store→load chain requires each step to see the
    previous step's result. Misordering would yield 0 or garbage.
    """
    assert (
        _jit("""
        | import algebra
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %init : Nil = memory.store(%alloc, %zero, %alloc)
        |     %cur : index.Index = memory.load(%init, %alloc)
        |     %seven : index.Index = 7
        |     %sum : index.Index = algebra.add(%cur, %seven)
        |     %st2 : Nil = memory.store(%cur, %sum, %alloc)
        |     %result : index.Index = memory.load(%st2, %alloc)
    """)
        == 7
    )


def test_two_independent_locations():
    """Store different values to two locations, read both back.

    Verifies mem chains work independently per-location. The stores
    to loc_a and loc_b are independent, but each load depends on
    its own store.
    """
    assert (
        _jit("""
        | import algebra
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %a : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v100 : index.Index = 100
        |     %sta : Nil = memory.store(%a, %v100, %a)
        |     %la : index.Index = memory.load(%sta, %a)
        |     %b : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v200 : index.Index = 200
        |     %stb : Nil = memory.store(%b, %v200, %b)
        |     %lb : index.Index = memory.load(%stb, %b)
        |     %result : index.Index = algebra.add(%la, %lb)
    """)
        == 300
    )


def test_store_float_then_load():
    """Store a float, load it back — verifies non-integer types work."""
    assert (
        _jit("""
        | import function
        | import memory
        | import number
        |
        | %main : function.Function<number.Float64> = function.function<number.Float64>() body():
        |     %alloc : memory.Reference<number.Float64> = memory.stack_allocate<number.Float64>()
        |     %val : number.Float64 = 3.14
        |     %st : Nil = memory.store(%alloc, %val, %alloc)
        |     %ld : number.Float64 = memory.load(%st, %alloc)
    """)
        == 3.14
    )


def test_overwrite_with_input_arg():
    """Store a constant, then overwrite with a function argument, load back.

    The mem chain ensures the argument store happens after the constant
    store. Returns the argument, not the constant.
    """
    ir = """
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %junk : index.Index = 999
        |     %st1 : Nil = memory.store(%alloc, %junk, %alloc)
        |     %st2 : Nil = memory.store(%st1, %x, %alloc)
        |     %ld : index.Index = memory.load(%st2, %alloc)
    """
    assert _jit(ir, 42) == 42
    assert _jit(ir, 0) == 0


def test_double_read_modify_write():
    """Two sequential read-modify-write cycles: 0 → +5 → +3 → must be 8.

    Each RMW cycle: load, add, store. The second cycle's load must see
    the first cycle's store. This would break without mem ordering.
    """
    assert (
        _jit("""
        | import algebra
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %z : index.Index = 0
        |     %init : Nil = memory.store(%alloc, %z, %alloc)
        |     %l1 : index.Index = memory.load(%init, %alloc)
        |     %five : index.Index = 5
        |     %a1 : index.Index = algebra.add(%l1, %five)
        |     %s1 : Nil = memory.store(%l1, %a1, %alloc)
        |     %l2 : index.Index = memory.load(%s1, %alloc)
        |     %three : index.Index = 3
        |     %a2 : index.Index = algebra.add(%l2, %three)
        |     %s2 : Nil = memory.store(%l2, %a2, %alloc)
        |     %final : index.Index = memory.load(%s2, %alloc)
    """)
        == 8
    )


def test_mem_from_chain_op():
    """The mem operand can be a ChainOp result, not just a load/store.

    This tests interop: ChainOp orders a side effect, then its result
    feeds as mem to a subsequent load.
    """
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %val : index.Index = 77
        |     %st : Nil = memory.store(%alloc, %val, %alloc)
        |     %ch : memory.Reference<index.Index> = chain(%alloc, %st)
        |     %ld : index.Index = memory.load(%ch, %alloc)
    """)
        == 77
    )


def test_cross_location_ordering():
    """Store to loc_a, then load loc_a into loc_b, read loc_b.

    The cross-location dependency is: store_a → load_a → store_b → load_b.
    load_a's mem depends on store_a; store_b's mem depends on load_a;
    load_b's mem depends on store_b. Must return 55.
    """
    assert (
        _jit("""
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %a : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v55 : index.Index = 55
        |     %sta : Nil = memory.store(%a, %v55, %a)
        |     %la : index.Index = memory.load(%sta, %a)
        |     %b : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %stb : Nil = memory.store(%la, %la, %b)
        |     %lb : index.Index = memory.load(%stb, %b)
    """)
        == 55
    )


def test_swap_via_mem_ordering():
    """Swap two locations: load both, then cross-store, then read back.

    Start: a=10, b=20. After swap: a=20, b=10. Return a+b*100 = 1020.
    This requires precise ordering: both loads must happen before either
    store, or the swap reads stale/overwritten data.
    """
    assert (
        _jit("""
        | import algebra
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<index.Index> = function.function<index.Index>() body():
        |     %b : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v20 : index.Index = 20
        |     %ib : Nil = memory.store(%b, %v20, %b)
        |     %lb : index.Index = memory.load(%ib, %b)
        |     %a : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v10 : index.Index = 10
        |     %ia : Nil = memory.store(%a, %v10, %a)
        |     %la : index.Index = memory.load(%ia, %a)
        |     %ma : index.Index = chain(%lb, %la)
        |     %sa : Nil = memory.store(%ma, %lb, %a)
        |     %fa : index.Index = memory.load(%sa, %a)
        |     %mb : index.Index = chain(%la, %lb)
        |     %sb : Nil = memory.store(%mb, %la, %b)
        |     %fb : index.Index = memory.load(%sb, %b)
        |     %h : index.Index = 100
        |     %bs : index.Index = algebra.multiply(%fb, %h)
        |     %result : index.Index = algebra.add(%fa, %bs)
    """)
        == 1020
    )
