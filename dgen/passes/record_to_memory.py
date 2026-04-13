"""Lower record ops to memory dialect ops.

record_pack([v0, v1, ...])        →  stack_allocate + store-per-field + load
record_get<i>(record)             →  lowered by MemoryToLLVM (extractvalue)
record_set<i>(mem, ptr, value)    →  offset + store
"""

from __future__ import annotations

import dgen
from dgen.builtins import unpack
from dgen.dialects import memory
from dgen.dialects.builtin import RecordPackOp, RecordSetOp
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import constant


def _field_ptr(ptr: dgen.Value, index: int, ref_type: memory.Reference) -> dgen.Value:
    """Return a pointer to field *index* of the record at *ptr*."""
    if index == 0:
        return ptr
    return memory.OffsetOp(ptr=ptr, index=Index().constant(index), type=ref_type)


class RecordToMemory(Pass):
    allow_unregistered_ops = True

    @lowering_for(RecordPackOp)
    def lower_pack(self, op: RecordPackOp) -> dgen.Value | None:
        record_type = op.type
        ref_type = memory.Reference(element_type=record_type)
        fields = unpack(op.values)

        ref = memory.StackAllocateOp(element_type=record_type, type=ref_type)

        mem: dgen.Value = ref
        for i, field_val in enumerate(fields):
            ptr = _field_ptr(ref, i, ref_type)
            mem = memory.StoreOp(mem=mem, value=field_val, ptr=ptr)

        return memory.LoadOp(mem=mem, ptr=ref, type=record_type)

    @lowering_for(RecordSetOp)
    def lower_set(self, op: RecordSetOp) -> dgen.Value | None:
        index = constant(op.index)
        assert isinstance(index, int)
        ref_type = op.ptr.type
        assert isinstance(ref_type, memory.Reference)
        ptr = _field_ptr(op.ptr, index, ref_type)
        return memory.StoreOp(mem=op.mem, value=op.value, ptr=ptr)
