"""Semantic tests for memory load/store with mem ordering.

These tests compile memory operations through the full pipeline and JIT-execute
them, verifying that the mem operand correctly enforces ordering. Each test is
designed so that incorrect ordering would produce the wrong result.
"""

from dgen.asm.parser import parse
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.testing import strip_prefix


def _jit(ir: str, *args: object) -> object:
    """Parse IR, compile through memory pipeline, JIT-run, return JSON result."""
    value = parse(strip_prefix(ir))
    compiler: Compiler[Executable] = Compiler(
        [ControlFlowToGoto(), MemoryToLLVM(), BuiltinToLLVM(), AlgebraToLLVM()],
        LLVMCodegen(),
    )
    exe = compiler.compile(value)
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], number.Float64> = function.function<number.Float64>() body():
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
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
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

    %both chains both loads into a single ordering point. Both swap stores
    use %both as mem, ensuring they execute after both loads complete.
    """
    assert (
        _jit("""
        | import algebra
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %a : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v10 : index.Index = 10
        |     %ia : Nil = memory.store(%a, %v10, %a)
        |     %la : index.Index = memory.load(%ia, %a)
        |     %b : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %v20 : index.Index = 20
        |     %ib : Nil = memory.store(%b, %v20, %b)
        |     %lb : index.Index = memory.load(%ib, %b)
        |     %both : index.Index = chain(%la, %lb)
        |     %sa : Nil = memory.store(%both, %lb, %a)
        |     %sb : Nil = memory.store(%both, %la, %b)
        |     %fa : index.Index = memory.load(%sa, %a)
        |     %fb : index.Index = memory.load(%sb, %b)
        |     %h : index.Index = 100
        |     %bs : index.Index = algebra.multiply(%fb, %h)
        |     %result : index.Index = algebra.add(%fa, %bs)
    """)
        == 1020
    )


def test_for_loop_accumulator():
    """Store 0, loop 5 times adding 1 each iteration, load — must read 5.

    The for loop body loads, increments, and stores. The final load uses
    the loop result as mem, ensuring it sees all iterations' stores.
    Without mem ordering the final load could read 0 (before any iteration).
    """
    assert (
        _jit("""
        | import algebra
        | import control_flow
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %zero : index.Index = 0
        |     %init : Nil = memory.store(%alloc, %zero, %alloc)
        |     %loop : Nil = control_flow.for<0, 5>([]) body(%iv: index.Index) captures(%alloc, %init):
        |         %cur : index.Index = memory.load(%init, %alloc)
        |         %one : index.Index = 1
        |         %next : index.Index = algebra.add(%cur, %one)
        |         %_ : Nil = memory.store(%cur, %next, %alloc)
        |     %result : index.Index = memory.load(%loop, %alloc)
    """)
        == 5
    )


def test_for_loop_last_iv_stored():
    """Store the loop IV each iteration; final load reads the last IV (4).

    Each iteration overwrites the accumulator with the current IV.
    The mem chain inside the loop body ensures each store sees the
    prior iteration's state. After the loop, load via mem=%loop.
    """
    assert (
        _jit("""
        | import control_flow
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %loop : Nil = control_flow.for<0, 5>([]) body(%iv: index.Index) captures(%alloc):
        |         %_ : Nil = memory.store(%alloc, %iv, %alloc)
        |     %result : index.Index = memory.load(%loop, %alloc)
    """)
        == 4
    )


def test_if_else_stores_to_same_location():
    """If/else branches store different values; load after sees the taken branch.

    condition=1 (truthy) → then branch stores 42. Load after if uses
    %if result as mem, ensuring it reads the branch's store.
    """
    assert (
        _jit("""
        | import control_flow
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %cond : index.Index = 1
        |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |         %t : index.Index = 42
        |         %_ : Nil = memory.store(%alloc, %t, %alloc)
        |     else_body() captures(%alloc):
        |         %f : index.Index = 99
        |         %_ : Nil = memory.store(%alloc, %f, %alloc)
        |     %result : index.Index = memory.load(%if, %alloc)
    """)
        == 42
    )


def test_if_else_false_branch():
    """Same as above but condition=0 (falsy) → else branch stores 99."""
    assert (
        _jit("""
        | import control_flow
        | import function
        | import index
        | import memory
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %alloc : memory.Reference<index.Index> = memory.stack_allocate<index.Index>()
        |     %cond : index.Index = 0
        |     %if : Nil = control_flow.if(%cond, [], []) then_body() captures(%alloc):
        |         %t : index.Index = 42
        |         %_ : Nil = memory.store(%alloc, %t, %alloc)
        |     else_body() captures(%alloc):
        |         %f : index.Index = 99
        |         %_ : Nil = memory.store(%alloc, %f, %alloc)
        |     %result : index.Index = memory.load(%if, %alloc)
    """)
        == 99
    )
