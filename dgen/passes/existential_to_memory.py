"""Lower existential dialect ops to record and memory ops.

``existential.pack(value)`` becomes:

    %witness   = builtin.type(value)
    %inner_ref = memory.heap_allocate<value_type>(N)
    %_         = memory.store(inner_ref, value, inner_ref)
    %chained   = chain(inner_ref, _)
    %result    = builtin.record_pack([witness, chained])

``existential.unpack(box)`` extracts the inner pointer via record_get
and loads the value through it:

    %inner_ptr = builtin.record_get<1>(box, box)
    %result    = memory.load(inner_ptr, inner_ptr)

RecordToMemory then lowers the record ops to stack-allocate/store/load.
"""

from __future__ import annotations

import dgen
from dgen.builtins import pack
from dgen.dialects import existential, memory
from dgen.dialects.builtin import ChainOp, RecordGetOp, RecordPackOp, TypeOp
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

        # Heap-allocate storage for the inner value and write the bytes.
        inner_ref_type = memory.Reference(element_type=value_type)
        inner_count = max(1, align_up(value_type.__layout__.byte_size, 8))
        inner_ref = memory.HeapAllocateOp(
            element_type=value_type,
            count=Index().constant(inner_count),
            type=inner_ref_type,
        )
        inner_store = memory.StoreOp(
            mem=inner_ref,
            value=op.value,
            ptr=inner_ref,
        )

        # Chain inner_ref through inner_store to ensure the heap write
        # happens before the record is constructed.
        chained_ref = ChainOp(lhs=inner_ref, rhs=inner_store, type=inner_ref.type)

        # Pack the Some record: [witness, inner_pointer].
        return RecordPackOp(
            values=pack([witness, chained_ref]),
            type=some_type,
        )

    @lowering_for(existential.UnpackOp)
    def lower_unpack(self, op: existential.UnpackOp) -> dgen.Value | None:
        witness_type = op.type
        assert isinstance(witness_type, dgen.Type)

        # Extract the inner pointer from field 1 of the Some.
        inner_ref_type = memory.Reference(element_type=witness_type)
        inner_ptr = RecordGetOp(
            index=Index().constant(1),
            mem=op.box,
            record=op.box,
            type=inner_ref_type,
        )

        # Load the inner value through the pointer.
        return memory.LoadOp(
            mem=inner_ptr,
            ptr=inner_ptr,
            type=witness_type,
        )
