"""Lower existential dialect ops to memory dialect ops.

``existential.pack(value)`` becomes:

    %witness   = builtin.type(value)
    %inner_ref = memory.heap_allocate<value_type>(N)
    %_         = memory.store(inner_ref, value, inner_ref)
    %some_ref  = memory.stack_allocate<Some<bound>>()
    %_         = memory.store(_, witness, some_ref)
    %val_field = memory.offset(some_ref, 1)
    %_         = memory.store(_, inner_ref, val_field)
    %result    = memory.load(_, some_ref)

The Some record is stack-allocated and loaded by value — since
``Some<bound>`` is ≤ 16 bytes it's register-passable (after the
register-passable-aggregates PR), so ``memory.load`` produces a
``{i64, i64}`` value in registers. The inner value is heap-allocated
because it must outlive the function frame (the pointer is embedded in
the Some and returned to the caller).

``existential.unpack(box)`` is the reverse: store the by-value Some
into a stack slot, read the value pointer from offset 1, and load the
inner value through it.

No direct LLVM emitter needed.
"""

from __future__ import annotations

import dgen
from dgen.dialects import existential, memory
from dgen.dialects.builtin import TypeOp
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
        some_ref_type = memory.Reference(element_type=some_type)

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

        # Stack-allocate the 16-byte Some record.
        some_ref = memory.StackAllocateOp(
            element_type=some_type,
            type=some_ref_type,
        )

        # Store witness at offset 0.
        witness_store = memory.StoreOp(
            mem=inner_store,
            value=witness,
            ptr=some_ref,
        )

        # Store inner pointer at offset 8 (element index 1, 8-byte stride).
        value_field_ref = memory.OffsetOp(
            ptr=some_ref,
            index=Index().constant(1),
            type=some_ref_type,
        )
        ref_store = memory.StoreOp(
            mem=witness_store,
            value=inner_ref,
            ptr=value_field_ref,
        )

        # Load the 16-byte Some by value (register-passable).
        return memory.LoadOp(
            mem=ref_store,
            ptr=some_ref,
            type=some_type,
        )

    @lowering_for(existential.UnpackOp)
    def lower_unpack(self, op: existential.UnpackOp) -> dgen.Value | None:
        witness_type = op.type
        assert isinstance(witness_type, dgen.Type)
        some_type = constant(op.box.type)
        some_ref_type = memory.Reference(element_type=some_type)

        # Stack-allocate a slot and store the by-value Some into it.
        some_ref = memory.StackAllocateOp(
            element_type=some_type,
            type=some_ref_type,
        )
        box_store = memory.StoreOp(
            mem=some_ref,
            value=op.box,
            ptr=some_ref,
        )

        # Read the inner pointer from offset 1.
        value_field_ref = memory.OffsetOp(
            ptr=some_ref,
            index=Index().constant(1),
            type=some_ref_type,
        )
        inner_ref_type = memory.Reference(element_type=witness_type)
        inner_ptr = memory.LoadOp(
            mem=box_store,
            ptr=value_field_ref,
            type=inner_ref_type,
        )

        # Load the inner value through the pointer.
        return memory.LoadOp(
            mem=inner_ptr,
            ptr=inner_ptr,
            type=witness_type,
        )
