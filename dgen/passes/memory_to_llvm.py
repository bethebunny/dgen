"""Lower memory dialect ops to LLVM dialect ops.

memory.heap_allocate(count) → extern<"malloc"> + function.call
memory.stack_allocate → llvm.alloca
memory.offset(ptr, index) → llvm.gep(ptr, index)
memory.deallocate → no-op (leak for now)

memory.load and memory.store pass through to codegen unchanged.
"""

from __future__ import annotations

import dgen
from dgen.dialects import function, llvm, memory
from dgen.dialects.builtin import ChainOp, ExternOp, Nil, String
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
        malloc = ExternOp(
            symbol=String().constant("malloc"),
            type=function.Function(arguments=pack([Index()]), result_type=llvm.Ptr()),
        )
        return function.CallOp(callee=malloc, arguments=pack([total]), type=llvm.Ptr())

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(self, op: memory.StackAllocateOp) -> dgen.Value | None:
        return llvm.AllocaOp(elem_count=Index().constant(1))

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp) -> dgen.Value | None:
        return ChainOp(lhs=ConstantOp(value=0, type=Nil()), rhs=op.mem, type=Nil())

    @lowering_for(memory.OffsetOp)
    def lower_offset(self, op: memory.OffsetOp) -> dgen.Value | None:
        return llvm.GepOp(base=op.ptr, index=op.index)
