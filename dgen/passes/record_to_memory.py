"""Lower record ops to memory dialect ops.

Records follow C-struct semantics: mutable in place, copied at boundaries.
The pass allocates one stack slot per record value, shared across all
record_get/record_set calls on that value so mutations are visible.

    record_pack([v0, v1, ...])       →  stack_allocate + store-per-field + load
    record_get<i>(mem, record)       →  slot_for(record) + offset + load
    record_set<i>(mem, record, val)  →  slot_for(record) + offset + store
"""

from __future__ import annotations

import dgen
from dgen.builtins import unpack
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp, RecordGetOp, RecordPackOp, RecordSetOp
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

    def __init__(self) -> None:
        # Shared stack slots: record value → (ref, initial store op).
        # All get/set on the same record value within a block share the
        # same backing slot. Reset per block to avoid cross-block sharing.
        self._slots: dict[dgen.Value, tuple[dgen.Value, dgen.Value]] = {}

    def _lower_block(self, block: dgen.Block) -> None:
        saved = self._slots
        self._slots = {}
        super()._lower_block(block)
        self._slots = saved

    def _slot_for(
        self, record: dgen.Value, mem: dgen.Value
    ) -> tuple[dgen.Value, dgen.Value]:
        """Get or create a shared stack slot for *record*.

        On first access, allocates a stack slot and stores the record value
        into it (so un-touched fields are readable). Returns (ref, initial_store).
        """
        if record not in self._slots:
            ref_type = memory.Reference(element_type=record.type)
            ref = memory.StackAllocateOp(element_type=record.type, type=ref_type)
            store = memory.StoreOp(mem=mem, value=record, ptr=ref)
            self._slots[record] = (ref, store)
        return self._slots[record]

    @lowering_for(RecordPackOp)
    def lower_pack(self, op: RecordPackOp) -> dgen.Value | None:
        record_type = op.type
        ref_type = memory.Reference(element_type=record_type)
        fields = unpack(op.values)

        # Stack-allocate a slot for the record.
        ref = memory.StackAllocateOp(element_type=record_type, type=ref_type)

        # Store each field value, threading the mem chain.
        mem: dgen.Value = ref
        for i, field_val in enumerate(fields):
            ptr = _field_ptr(ref, i, ref_type)
            mem = memory.StoreOp(mem=mem, value=field_val, ptr=ptr)

        # Load the complete record by value.
        return memory.LoadOp(mem=mem, ptr=ref, type=record_type)

    @lowering_for(RecordGetOp)
    def lower_get(self, op: RecordGetOp) -> dgen.Value | None:
        index = constant(op.index)
        assert isinstance(index, int)
        ref, initial_store = self._slot_for(op.record, op.mem)
        # Depend on both the initial store and op.mem (e.g. a prior record_set).
        mem = ChainOp(lhs=op.mem, rhs=initial_store, type=op.mem.type)
        ref_type = memory.Reference(element_type=op.record.type)
        ptr = _field_ptr(ref, index, ref_type)
        return memory.LoadOp(mem=mem, ptr=ptr, type=op.type)

    @lowering_for(RecordSetOp)
    def lower_set(self, op: RecordSetOp) -> dgen.Value | None:
        index = constant(op.index)
        assert isinstance(index, int)
        ref, initial_store = self._slot_for(op.record, op.mem)
        # Depend on both the initial store and op.mem.
        mem = ChainOp(lhs=op.mem, rhs=initial_store, type=op.mem.type)
        ref_type = memory.Reference(element_type=op.record.type)
        ptr = _field_ptr(ref, index, ref_type)
        return memory.StoreOp(mem=mem, value=op.value, ptr=ptr)
