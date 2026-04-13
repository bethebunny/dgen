"""Lower record ops to memory dialect ops.

Records follow C-struct semantics: mutable in place, copied at boundaries.
record_pack allocates a stack slot, stores fields, and loads by value.
record_get/record_set look up the backing slot from record_pack and
operate on it directly.

    record_pack([v0, v1, ...])       →  stack_allocate + store-per-field + load
    record_get<i>(mem, record)       →  offset(slot, i) + load
    record_set<i>(mem, record, val)  →  offset(slot, i) + store
"""

from __future__ import annotations

import dgen
from dgen.builtins import unpack
from dgen.dialects import memory
from dgen.dialects.builtin import RecordGetOp, RecordPackOp, RecordSetOp
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
        # Maps a record_pack's result (LoadOp) → its backing stack slot (ref).
        # record_get/record_set look up their record operand here to find the
        # slot that record_pack already allocated.
        self._slots: dict[dgen.Value, dgen.Value] = {}

    def _slot_for(
        self, record: dgen.Value, mem: dgen.Value
    ) -> tuple[dgen.Value, dgen.Value]:
        """Look up the backing slot for *record*, or create one on demand.

        Records from record_pack already have a slot (cached in lower_pack).
        Records from other sources (e.g. function arguments) get a fresh
        slot — not cached, to avoid sharing across blocks.

        Returns (ref, mem) where mem chains through the initial store
        if one was created.
        """
        if record in self._slots:
            return self._slots[record], mem
        ref_type = memory.Reference(element_type=record.type)
        ref = memory.StackAllocateOp(element_type=record.type, type=ref_type)
        store = memory.StoreOp(mem=mem, value=record, ptr=ref)
        return ref, store

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

        # Load the complete record by value (for function boundaries).
        result = memory.LoadOp(mem=mem, ptr=ref, type=record_type)

        # Cache: after replace_uses_of, record_get/record_set will see
        # this LoadOp as their record operand and can look up the slot.
        self._slots[result] = ref

        return result

    @lowering_for(RecordGetOp)
    def lower_get(self, op: RecordGetOp) -> dgen.Value | None:
        index = constant(op.index)
        assert isinstance(index, int)
        ref, mem = self._slot_for(op.record, op.mem)
        ref_type = memory.Reference(element_type=op.record.type)
        ptr = _field_ptr(ref, index, ref_type)
        return memory.LoadOp(mem=mem, ptr=ptr, type=op.type)

    @lowering_for(RecordSetOp)
    def lower_set(self, op: RecordSetOp) -> dgen.Value | None:
        index = constant(op.index)
        assert isinstance(index, int)
        ref, mem = self._slot_for(op.record, op.mem)
        ref_type = memory.Reference(element_type=op.record.type)
        ptr = _field_ptr(ref, index, ref_type)
        return memory.StoreOp(mem=mem, value=op.value, ptr=ptr)
