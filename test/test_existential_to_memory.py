"""Tests for the existential → memory lowering pass."""

from __future__ import annotations

import dgen
from dgen import Block
from dgen.block import BlockArgument
from dgen.builtins import pack as pack_values
from dgen.dialects import existential, memory
from dgen.dialects.function import Function, FunctionOp
from dgen.dialects.index import Index
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.existential_to_memory import ExistentialToMemory


def _full_compile(value: dgen.Value):
    """Compile through the existential-aware pipeline (matches __main__)."""
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            ExistentialToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    ).run(value)


def test_lower_pack_produces_memory_sequence() -> None:
    """``existential.pack<bound>(value)`` lowers to a memory.load on top of
    a chain of stack_allocate / store / offset / store ops."""
    arg = BlockArgument(name="x", type=Index())
    pack_op = existential.PackOp(
        bound=Index(),
        value=arg,
        type=existential.Some(bound=Index()),
    )
    func = FunctionOp(
        name="main",
        body=Block(result=pack_op, args=[arg]),
        result_type=pack_op.type,
        type=Function(arguments=pack_values([Index()]), result_type=pack_op.type),
    )

    lowered = Compiler([ExistentialToMemory()], IdentityPass()).run(func)

    # The result of the function body is now a ChainOp with type
    # Some<Index>; its lhs is the some_ref heap allocation, its rhs is
    # the last store in the chain.
    from dgen.dialects.builtin import ChainOp

    new_result = lowered.body.result
    assert isinstance(new_result, ChainOp)
    assert isinstance(new_result.type, existential.Some)
    assert isinstance(new_result.lhs, memory.HeapAllocateOp)
    assert isinstance(new_result.rhs, memory.StoreOp)

    # Walk back through the chain via .mem to verify the expected ops.
    chain: list[type] = []
    cursor: dgen.Value = new_result.rhs
    while True:
        chain.append(type(cursor))
        if not hasattr(cursor, "mem"):
            break
        cursor = cursor.mem
    # We expect: StoreOp(value field) -> StoreOp(inner) -> StoreOp(witness)
    # -> HeapAllocateOp(some).
    assert chain == [
        memory.StoreOp,
        memory.StoreOp,
        memory.StoreOp,
        memory.HeapAllocateOp,
    ]


def test_pack_compiles_through_full_pipeline() -> None:
    """Pack lowering compiles through the full LLVM pipeline.

    We don't read the result back yet: there's a separate latent issue
    with how non-register-passable ``TypeType`` constants are emitted
    (the LLVM value is the buffer address rather than the buffer's
    contents — fine for arg/return passthrough, but the wrong shape
    for storing into a Some struct's witness field). The full byte-
    level round-trip needs that resolved (probably by making `Record`
    register-passable for small records, the user's suggestion) plus
    `unpack` for the value-side. Both tracked under "Existentials" in
    TODO.md.
    """
    arg = BlockArgument(name="x", type=Index())
    pack_op = existential.PackOp(
        bound=Index(),
        value=arg,
        type=existential.Some(bound=Index()),
    )
    func = FunctionOp(
        name="main",
        body=Block(result=pack_op, args=[arg]),
        result_type=pack_op.type,
        type=Function(arguments=pack_values([Index()]), result_type=pack_op.type),
    )
    # Compile only — the Some bytes returned via pointer have a
    # downstream issue with the witness pointer's indirection level.
    exe = _full_compile(func)
    assert exe is not None


def test_pack_lowering_pass_is_no_op_on_unrelated_ir() -> None:
    """ExistentialToMemory only fires on PackOps; unrelated IR passes through."""
    func = FunctionOp(
        name="main",
        body=Block(result=Index().constant(42)),
        result_type=Index(),
        type=Function(arguments=pack_values([]), result_type=Index()),
    )
    lowered = Compiler([ExistentialToMemory()], IdentityPass()).run(func)
    assert isinstance(lowered.body.result.type, Index)
