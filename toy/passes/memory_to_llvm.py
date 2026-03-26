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
from dgen.dialects import builtin, index, llvm, memory
from dgen.dialects.builtin import Nil, String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(memory.HeapAllocateOp)
    def lower_heap_allocate(
        self, op: memory.HeapAllocateOp, rewriter: Rewriter
    ) -> bool:
        # malloc(count * 8) for 8-byte doubles
        byte_count = ConstantOp(value=8, type=Index())
        total = llvm.MulOp(lhs=op.count, rhs=byte_count)
        malloc_args = PackOp(values=[total], type=builtin.List(element_type=Index()))
        malloc_op = llvm.CallOp(
            callee=String().constant("malloc"), args=malloc_args, type=llvm.Ptr()
        )
        rewriter.replace_uses(op, malloc_op)
        return True

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(
        self, op: memory.StackAllocateOp, rewriter: Rewriter
    ) -> bool:
        rewriter.replace_uses(op, llvm.AllocaOp(elem_count=Index().constant(1)))
        return True

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, ConstantOp(value=0, type=Nil()))
        return True

    @lowering_for(memory.LoadOp)
    def lower_load(self, op: memory.LoadOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, llvm.LoadOp(ptr=op.ptr, type=op.type))
        return True

    @lowering_for(memory.StoreOp)
    def lower_store(self, op: memory.StoreOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, llvm.StoreOp(value=op.value, ptr=op.ptr))
        return True

    @lowering_for(memory.OffsetOp)
    def lower_offset(self, op: memory.OffsetOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, llvm.GepOp(base=op.ptr, index=op.index))
        return True
