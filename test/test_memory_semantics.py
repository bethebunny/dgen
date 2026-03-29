"""Semantic tests for memory load/store with mem ordering.

These tests compile memory operations through the full pipeline and JIT-execute
them, verifying that the mem operand correctly enforces ordering. Each test is
designed so that incorrect ordering would produce the wrong result.
"""

from dgen import Block, Value
from dgen.block import BlockArgument
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects import algebra, memory
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM


def _compile(func: FunctionOp) -> Executable:
    """Compile a single function through the memory pipeline."""
    module = Module(ops=[func])
    compiler: Compiler[Executable] = Compiler(
        [ControlFlowToGoto(), MemoryToLLVM()], LLVMCodegen()
    )
    return compiler.compile(module)


def _ref(dtype: Value) -> memory.Reference:
    return memory.Reference(element_type=dtype)


def test_store_then_load_sees_stored_value():
    """Store 42, then load — must read back 42.

    Without mem ordering, the load could be scheduled before the store
    and read uninitialized memory.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    val = ConstantOp(value=42, type=Index())
    store = memory.StoreOp(mem=alloc, value=val, ptr=alloc)
    load = memory.LoadOp(mem=store, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 42


def test_two_stores_last_wins():
    """Store 10, then store 20, then load — must read 20.

    The mem chain store1 → store2 → load ensures the second store
    overwrites the first. Without ordering, either value could appear.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    v10 = ConstantOp(value=10, type=Index())
    v20 = ConstantOp(value=20, type=Index())
    store1 = memory.StoreOp(mem=alloc, value=v10, ptr=alloc)
    store2 = memory.StoreOp(mem=store1, value=v20, ptr=alloc)
    load = memory.LoadOp(mem=store2, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 20


def test_three_stores_last_wins():
    """Store 1, 2, 3 in sequence — load must read 3.

    A longer chain validates that mem threading works across
    multiple sequential stores.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    s1 = memory.StoreOp(mem=alloc, value=ConstantOp(value=1, type=Index()), ptr=alloc)
    s2 = memory.StoreOp(mem=s1, value=ConstantOp(value=2, type=Index()), ptr=alloc)
    s3 = memory.StoreOp(mem=s2, value=ConstantOp(value=3, type=Index()), ptr=alloc)
    load = memory.LoadOp(mem=s3, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 3


def test_read_modify_write():
    """Store 0, load, add 7, store back, load — must read 7.

    The load→add→store→load chain requires each step to see the
    previous step's result. Misordering would yield 0 or garbage.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    init = memory.StoreOp(mem=alloc, value=ConstantOp(value=0, type=Index()), ptr=alloc)
    load1 = memory.LoadOp(mem=init, ptr=alloc, type=Index())
    seven = ConstantOp(value=7, type=Index())
    added = algebra.AddOp(left=load1, right=seven, type=Index())
    store2 = memory.StoreOp(mem=load1, value=added, ptr=alloc)
    load2 = memory.LoadOp(mem=store2, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load2, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 7


def test_two_independent_locations():
    """Store different values to two locations, read both back.

    Verifies mem chains work independently per-location. The stores
    to loc_a and loc_b are independent, but each load depends on
    its own store.
    """
    alloc_a = memory.StackAllocateOp(name="a", element_type=Index(), type=_ref(Index()))
    alloc_b = memory.StackAllocateOp(name="b", element_type=Index(), type=_ref(Index()))
    store_a = memory.StoreOp(
        mem=alloc_a, value=ConstantOp(value=100, type=Index()), ptr=alloc_a
    )
    store_b = memory.StoreOp(
        mem=alloc_b, value=ConstantOp(value=200, type=Index()), ptr=alloc_b
    )
    load_a = memory.LoadOp(mem=store_a, ptr=alloc_a, type=Index())
    load_b = memory.LoadOp(mem=store_b, ptr=alloc_b, type=Index())
    result = algebra.AddOp(left=load_a, right=load_b, type=Index())
    # Both stores and loads must be reachable via result
    func = FunctionOp(
        name="main",
        body=Block(result=result, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 300


def test_store_float_then_load():
    """Store a float, load it back — verifies non-integer types work."""
    alloc = memory.StackAllocateOp(element_type=Float64(), type=_ref(Float64()))
    val = ConstantOp(value=3.14, type=Float64())
    store = memory.StoreOp(mem=alloc, value=val, ptr=alloc)
    load = memory.LoadOp(mem=store, ptr=alloc, type=Float64())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[]),
        result=Float64(),
        type=Function(result=Float64()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 3.14


def test_overwrite_with_input_arg():
    """Store a constant, then overwrite with a function argument, load back.

    The mem chain ensures the argument store happens after the constant
    store. Returns the argument, not the constant.
    """
    arg = BlockArgument(name="x", type=Index())
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    store1 = memory.StoreOp(
        mem=alloc, value=ConstantOp(value=999, type=Index()), ptr=alloc
    )
    store2 = memory.StoreOp(mem=store1, value=arg, ptr=alloc)
    load = memory.LoadOp(mem=store2, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[arg]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run(42).to_json() == 42
    assert exe.run(0).to_json() == 0


def test_double_read_modify_write():
    """Two sequential read-modify-write cycles: 0 → +5 → +3 → must be 8.

    Each RMW cycle: load, add, store. The second cycle's load must see
    the first cycle's store. This would break without mem ordering.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    init = memory.StoreOp(mem=alloc, value=ConstantOp(value=0, type=Index()), ptr=alloc)

    # Cycle 1: load, +5, store
    load1 = memory.LoadOp(mem=init, ptr=alloc, type=Index())
    add1 = algebra.AddOp(
        left=load1, right=ConstantOp(value=5, type=Index()), type=Index()
    )
    store1 = memory.StoreOp(mem=load1, value=add1, ptr=alloc)

    # Cycle 2: load, +3, store
    load2 = memory.LoadOp(mem=store1, ptr=alloc, type=Index())
    add2 = algebra.AddOp(
        left=load2, right=ConstantOp(value=3, type=Index()), type=Index()
    )
    store2 = memory.StoreOp(mem=load2, value=add2, ptr=alloc)

    final = memory.LoadOp(mem=store2, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=final, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 8


def test_mem_from_chain_op():
    """The mem operand can be a ChainOp result, not just a load/store.

    This tests interop: ChainOp orders a loop, then its result
    feeds as mem to a subsequent load.
    """
    alloc = memory.StackAllocateOp(element_type=Index(), type=_ref(Index()))
    store = memory.StoreOp(
        mem=alloc, value=ConstantOp(value=77, type=Index()), ptr=alloc
    )
    # ChainOp makes alloc depend on store (alloc's value, store's ordering)
    chained = ChainOp(lhs=alloc, rhs=store, type=_ref(Index()))
    # Load uses the chained result as mem
    load = memory.LoadOp(mem=chained, ptr=alloc, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 77


def test_cross_location_ordering():
    """Store to loc_a, then load loc_a into loc_b, read loc_b.

    The cross-location dependency is: store_a → load_a → store_b → load_b.
    load_a's mem depends on store_a; store_b's mem depends on load_a;
    load_b's mem depends on store_b. Must return 55.
    """
    alloc_a = memory.StackAllocateOp(name="a", element_type=Index(), type=_ref(Index()))
    alloc_b = memory.StackAllocateOp(name="b", element_type=Index(), type=_ref(Index()))
    store_a = memory.StoreOp(
        mem=alloc_a, value=ConstantOp(value=55, type=Index()), ptr=alloc_a
    )
    load_a = memory.LoadOp(mem=store_a, ptr=alloc_a, type=Index())
    store_b = memory.StoreOp(mem=load_a, value=load_a, ptr=alloc_b)
    load_b = memory.LoadOp(mem=store_b, ptr=alloc_b, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=load_b, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 55


def test_swap_via_mem_ordering():
    """Swap two locations: load both, then cross-store, then read back.

    Start: a=10, b=20. After swap: a=20, b=10. Return a+b*100 = 1020.
    This requires precise ordering: both loads must happen before either
    store, or the swap reads stale/overwritten data.
    """
    alloc_a = memory.StackAllocateOp(name="a", element_type=Index(), type=_ref(Index()))
    alloc_b = memory.StackAllocateOp(name="b", element_type=Index(), type=_ref(Index()))
    # Initialize
    init_a = memory.StoreOp(
        mem=alloc_a, value=ConstantOp(value=10, type=Index()), ptr=alloc_a
    )
    init_b = memory.StoreOp(
        mem=alloc_b, value=ConstantOp(value=20, type=Index()), ptr=alloc_b
    )
    # Load both BEFORE any swap store
    load_a = memory.LoadOp(mem=init_a, ptr=alloc_a, type=Index())
    load_b = memory.LoadOp(mem=init_b, ptr=alloc_b, type=Index())
    # Cross-store: a←b, b←a. Each swap store depends on BOTH loads.
    # Use ChainOp to make swap_a depend on load_b too (it already depends on load_a via init_a's chain)
    swap_a_mem = ChainOp(lhs=load_b, rhs=load_a, type=Index())
    swap_a = memory.StoreOp(mem=swap_a_mem, value=load_b, ptr=alloc_a)
    swap_b_mem = ChainOp(lhs=load_a, rhs=load_b, type=Index())
    swap_b = memory.StoreOp(mem=swap_b_mem, value=load_a, ptr=alloc_b)
    # Read back
    final_a = memory.LoadOp(mem=swap_a, ptr=alloc_a, type=Index())
    final_b = memory.LoadOp(mem=swap_b, ptr=alloc_b, type=Index())
    # a=20, b=10. Return a + b*100 = 20 + 1000 = 1020
    hundred = ConstantOp(value=100, type=Index())
    b_scaled = algebra.MultiplyOp(left=final_b, right=hundred, type=Index())
    result = algebra.AddOp(left=final_a, right=b_scaled, type=Index())

    func = FunctionOp(
        name="main",
        body=Block(result=result, args=[]),
        result=Index(),
        type=Function(result=Index()),
    )
    exe = _compile(func)
    assert exe.run().to_json() == 1020
