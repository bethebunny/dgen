"""Lower memory dialect ops to LLVM dialect ops.

Allocation/deallocation ops lower here:

    memory.heap_allocate<T>()       → extern<"malloc"> + function.call(byte_size(T))
    memory.stack_allocate<T>()      → llvm.alloca(byte_size(T))
    memory.deallocate(ptr)          → no-op (leak for now)
    memory.buffer_allocate<T>(n)    → extern<"malloc"> + function.call(n * 8)
    memory.buffer_deallocate(_, _)  → no-op

memory.load / memory.store / memory.buffer_load / memory.buffer_store
pass through to codegen unchanged — codegen emits the LLVM load/store
plus, for ``load``, the ``insertvalue`` chain that builds the
``Tuple<T, Reference<T>>`` aggregate result.
"""

from __future__ import annotations

import dgen
from dgen.dialects import function, llvm, memory
from dgen.dialects.builtin import ChainOp, ExternOp, Nil, String
from dgen.dialects.record import GetOp as RecordGetOp
from dgen.dialects.index import Index
from dgen.builtins import pack
from dgen.layout import align_up
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import constant


def _malloc_call(byte_count: dgen.Value) -> dgen.Value:
    """Build a `function.call<malloc>(byte_count)` returning `llvm.Ptr`."""
    malloc = ExternOp(
        symbol=String().constant("malloc"),
        type=function.Function(arguments=pack([Index()]), result_type=llvm.Ptr()),
    )
    return function.CallOp(callee=malloc, arguments=pack([byte_count]), type=llvm.Ptr())


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(memory.HeapAllocateOp)
    def lower_heap_allocate(self, op: memory.HeapAllocateOp) -> dgen.Value | None:
        element_type = constant(op.element_type)
        assert isinstance(element_type, dgen.Type)
        byte_size = max(1, align_up(element_type.__layout__.byte_size, 8))
        return _malloc_call(Index().constant(byte_size))

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(self, op: memory.StackAllocateOp) -> dgen.Value | None:
        element_type = constant(op.element_type)
        byte_size = element_type.__layout__.byte_size
        # alloca uses 8-byte (double) units; round up.
        count = max(1, align_up(byte_size, 8))
        return llvm.AllocaOp(elem_count=Index().constant(count))

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp) -> dgen.Value | None:
        return ChainOp(lhs=Nil().constant(None), rhs=op.ptr, type=Nil())

    @lowering_for(memory.BufferAllocateOp)
    def lower_buffer_allocate(self, op: memory.BufferAllocateOp) -> dgen.Value | None:
        # buffer_allocate counts in 8-byte units (matches Float64 / GEP stride).
        int64 = llvm.Int(bits=Index().constant(64))
        total = llvm.MulOp(lhs=op.count, rhs=int64.constant(8))
        return _malloc_call(total)

    @lowering_for(memory.BufferStackAllocateOp)
    def lower_buffer_stack_allocate(
        self, op: memory.BufferStackAllocateOp
    ) -> dgen.Value | None:
        # alloca counts in 8-byte (double) units — matches `count`'s stride.
        return llvm.AllocaOp(elem_count=op.count)

    @lowering_for(memory.BufferDeallocateOp)
    def lower_buffer_deallocate(
        self, op: memory.BufferDeallocateOp
    ) -> dgen.Value | None:
        return ChainOp(lhs=Nil().constant(None), rhs=op.mem, type=Nil())

    @lowering_for(RecordGetOp)
    def lower_record_get(self, op: RecordGetOp) -> dgen.Value | None:
        from dgen.llvm.ffi import _LLVM, _struct_fields

        record_type = constant(op.record.type)
        assert isinstance(record_type, dgen.Type)
        index = constant(op.index)
        assert isinstance(index, int)

        # Determine the LLVM type of the extracted field from the struct format.
        fmt_fields = _struct_fields(record_type.__layout__.struct.format)
        field_llvm = _LLVM.get(fmt_fields[index], "i8")
        result_llvm = _LLVM.get(
            _struct_fields(op.type.__layout__.struct.format)[0], "i8"
        )

        extract = llvm.ExtractValueOp(index=op.index, aggregate=op.record, type=op.type)
        # If the LLVM types differ (e.g. extracting ptr but expecting i64),
        # insert a ptrtoint cast.
        if field_llvm == "ptr" and result_llvm != "ptr":
            return llvm.PtrtointOp(input=extract, type=op.type)
        return extract
