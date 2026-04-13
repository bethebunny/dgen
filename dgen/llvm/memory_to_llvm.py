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
from dgen.dialects.builtin import ChainOp, ExternOp, Nil, RecordGetOp, String
from dgen.dialects.index import Index
from dgen.builtins import pack
from dgen.layout import align_up
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import constant


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(memory.HeapAllocateOp)
    def lower_heap_allocate(self, op: memory.HeapAllocateOp) -> dgen.Value | None:
        # malloc(count * 8) for 8-byte doubles
        int64 = llvm.Int(bits=Index().constant(64))
        total = llvm.MulOp(lhs=op.count, rhs=int64.constant(8))
        malloc = ExternOp(
            symbol=String().constant("malloc"),
            type=function.Function(arguments=pack([Index()]), result_type=llvm.Ptr()),
        )
        return function.CallOp(callee=malloc, arguments=pack([total]), type=llvm.Ptr())

    @lowering_for(memory.StackAllocateOp)
    def lower_stack_allocate(self, op: memory.StackAllocateOp) -> dgen.Value | None:
        element_type = constant(op.element_type)
        byte_size = element_type.__layout__.byte_size
        # alloca uses 8-byte (double) units; round up.
        count = max(1, align_up(byte_size, 8))
        return llvm.AllocaOp(elem_count=Index().constant(count))

    @lowering_for(memory.DeallocateOp)
    def lower_deallocate(self, op: memory.DeallocateOp) -> dgen.Value | None:
        return ChainOp(lhs=Nil().constant(None), rhs=op.mem, type=Nil())

    @lowering_for(memory.OffsetOp)
    def lower_offset(self, op: memory.OffsetOp) -> dgen.Value | None:
        return llvm.GepOp(base=op.ptr, index=op.index)

    @lowering_for(RecordGetOp)
    def lower_record_get(self, op: RecordGetOp) -> dgen.Value | None:
        from dgen.llvm.ffi import _struct_fields, _LLVM

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
