"""Lower memory dialect ops to LLVM dialect ops.

memory.heap_allocate(count) → llvm.call<"malloc">(count * element_size)
memory.stack_allocate → llvm.alloca
memory.load(ptr) → llvm.load(ptr)
memory.store(val, ptr) → llvm.store(val, ptr)
memory.offset(ptr, index) → llvm.gep(ptr, index)
memory.deallocate → no-op (leak for now)
"""

from __future__ import annotations

import dgen
from dgen.dialects import llvm, memory
from dgen.dialects.builtin import Nil, String
from dgen.dialects.index import Index
from dgen.module import ConstantOp, pack
from dgen.passes.pass_ import Pass, lowering_for


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(memory.HeapAllocateOp)
    def lower_heap_allocate(self, op: memory.HeapAllocateOp) -> dgen.Value | None:
        # malloc(count * 8) for 8-byte doubles
        byte_count = ConstantOp(value=8, type=Index())
        total = llvm.MulOp(lhs=op.count, rhs=byte_count)
        return llvm.CallOp(
            callee=String().constant("malloc"), args=pack([total]), type=llvm.Ptr()
        )

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(self, op: memory.StackAllocateOp) -> dgen.Value | None:
        return llvm.AllocaOp(elem_count=Index().constant(1))

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp) -> dgen.Value | None:
        return ConstantOp(value=0, type=Nil())

    @lowering_for(memory.LoadOp)
    def lower_load(self, op: memory.LoadOp) -> dgen.Value | None:
        return llvm.LoadOp(ptr=op.ptr, type=op.type)

    @lowering_for(memory.StoreOp)
    def lower_store(self, op: memory.StoreOp) -> dgen.Value | None:
        return llvm.StoreOp(value=op.value, ptr=op.ptr)

    @lowering_for(memory.OffsetOp)
    def lower_offset(self, op: memory.OffsetOp) -> dgen.Value | None:
        return llvm.GepOp(base=op.ptr, index=op.index)
