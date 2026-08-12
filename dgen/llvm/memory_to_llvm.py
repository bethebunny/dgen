"""Lower memory dialect ops to LLVM dialect ops.

Allocation/deallocation ops lower here:

    memory.heap_allocate<T>()       → extern<"malloc"> + function.call(byte_size(T)),
                                      packed as the 1-field aggregate the
                                      allocation's Tuple<[Reference<T>, Origin]>
                                      consumers expect. Origin is zero-sized.
    memory.stack_allocate<T>()      → llvm.alloca(byte_size(T)), packed likewise
    memory.deallocate(origin, ref)  → extern<"free"> + function.call(ref)
    memory.destroy(origin)          → error. LowerDestroy applies each
                                      origin's attached destructor before
                                      this pass runs.
    memory.buffer_allocate<T>(n)    → extern<"malloc"> + function.call(n * 8)
    memory.buffer_deallocate(_, _)  → no-op

memory.load / memory.store / memory.buffer_load / memory.buffer_store
pass through to codegen unchanged. Codegen emits the LLVM load or store
plus, for ``load``, the ``insertvalue`` that builds the
``Tuple<[T, Origin]>`` aggregate result. The Origin field erases.
"""

from __future__ import annotations

import dgen
from dgen.builtins import pack
from dgen.dialects import function, llvm, memory, record
from dgen.dialects.builtin import ChainOp, ExternOp, Nil, String
from dgen.dialects.index import Index
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


class UnloweredDestroyError(Exception):
    """A memory.destroy survived to MemoryToLLVM. Run LowerDestroy
    earlier in the pipeline."""


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(memory.HeapAllocateOp)
    def lower_heap_allocate(self, op: memory.HeapAllocateOp) -> dgen.Value | None:
        element_type = constant(op.element_type)
        assert isinstance(element_type, dgen.Type)
        byte_size = max(1, align_up(element_type.__layout__.byte_size, 8))
        # The op's Tuple<[Reference<T>, Origin]> consumers see a
        # 1-field aggregate. Origin is zero-sized and erases here.
        return pack([_malloc_call(Index().constant(byte_size))])

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(self, op: memory.StackAllocateOp) -> dgen.Value | None:
        element_type = constant(op.element_type)
        byte_size = element_type.__layout__.byte_size
        # alloca counts in 8-byte (double) units.
        count = max(1, align_up(byte_size, 8) // 8)
        return pack([llvm.AllocaOp(elem_count=Index().constant(count))])

    @lowering_for(memory.DestroyOp)
    def lower_destroy(self, op: memory.DestroyOp) -> dgen.Value | None:
        raise UnloweredDestroyError(
            f"memory.destroy %{op.name} reached MemoryToLLVM; run "
            f"LowerDestroy earlier in the pipeline"
        )

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp) -> dgen.Value | None:
        # Chain the origin into the pointer argument so the free stays
        # ordered after the accesses that produced the origin.
        ref = ChainOp(lhs=op.ref, rhs=op.origin, type=op.ref.type)
        free = ExternOp(
            symbol=String().constant("free"),
            type=function.Function(arguments=pack([llvm.Ptr()]), result_type=Nil()),
        )
        return function.CallOp(callee=free, arguments=pack([ref]), type=Nil())

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

    @lowering_for(record.GetOp)
    def lower_record_get(self, op: record.GetOp) -> dgen.Value | None:
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
