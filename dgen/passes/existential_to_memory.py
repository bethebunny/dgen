"""Lower existential dialect ops to memory ops.

``existential.pack(value)`` becomes:

    %witness    = builtin.type(value)
    %inner_ref  = memory.heap_allocate<some_type>(N)
    %_          = memory.store(inner_ref, witness, inner_ref)
    %val_field  = memory.offset(inner_ref, 1)
    %_          = memory.store(_, value, val_field)
    %result     = inner_ref  (the pointer IS the Some)

``existential.unpack(box)`` dereferences the pointer and reads the
value from offset 1:

    %val_field  = memory.offset(box, 1)
    %result     = memory.load(box, val_field)
"""

from __future__ import annotations

import dgen
from dgen.dialects import existential, memory
from dgen.dialects.builtin import ChainOp, TypeOp
from dgen.dialects.index import Index
from dgen.layout import align_up
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import TypeType, constant


class ExistentialToMemory(Pass):
    allow_unregistered_ops = True

    @lowering_for(existential.PackOp)
    def lower_pack(self, op: existential.PackOp) -> dgen.Value | None:
        value_type = constant(op.value.type)
        assert isinstance(value_type, dgen.Type)
        some_type = op.type

        # Witness type as a compile-time TypeType constant.
        witness = TypeOp(value=op.value, type=TypeType())

        # Heap-allocate a block for {TypeValue, value_bytes}.
        # TypeValue is 8 bytes (one pointer), value follows inline.
        inner_bytes = 8 + value_type.__layout__.byte_size
        inner_count = max(1, align_up(inner_bytes, 8))
        ref_type = memory.Reference(element_type=some_type)
        inner_ref = memory.HeapAllocateOp(
            element_type=some_type, count=Index().constant(inner_count), type=ref_type
        )

        # Store witness (TypeValue) at offset 0.
        witness_store = memory.StoreOp(mem=inner_ref, value=witness, ptr=inner_ref)

        # Store value at offset 1 (8-byte stride).
        value_field = memory.OffsetOp(
            ptr=inner_ref, index=Index().constant(1), type=ref_type
        )
        value_store = memory.StoreOp(mem=witness_store, value=op.value, ptr=value_field)

        # The pointer itself IS the Some value.
        return ChainOp(lhs=inner_ref, rhs=value_store, type=inner_ref.type)

    @lowering_for(existential.UnpackOp)
    def lower_unpack(self, op: existential.UnpackOp) -> dgen.Value | None:
        witness_type = op.type
        assert isinstance(witness_type, dgen.Type)

        # The Some value is a pointer to {TypeValue, value_inline}.
        # The value is at offset 1 (after the 8-byte TypeValue).
        some_ref_type = memory.Reference(element_type=op.box.type)
        value_field = memory.OffsetOp(
            ptr=op.box, index=Index().constant(1), type=some_ref_type
        )
        return memory.LoadOp(mem=op.box, ptr=value_field, type=witness_type)
