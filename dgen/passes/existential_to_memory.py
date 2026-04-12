"""Lower existential dialect ops to memory dialect ops.

``existential.pack<bound>(value)`` becomes a small sequence of heap
allocations and stores that build a ``Some<bound>`` struct:

    pack<bound>(value)
      → some_ref   = heap_allocate<Some<bound>>(2)   # 16 bytes
        _          = store(some_ref, builtin.type(value), some_ref)
        inner_ref  = heap_allocate<value.type>(1)    # 8 bytes (rounded up)
        _          = store(_, value, inner_ref)
        field_ref  = offset(some_ref, 1)
        _          = store(_, inner_ref, field_ref)
        result     = chain(some_ref, _) typed Some<bound>

The witness type travels in the first 8 bytes (a ``TypeValue`` pointer
that ``builtin.type`` resolves to a TypeType constant), and the
heap-allocated value buffer's address travels in the second 8 bytes.
``memory.offset`` advances by one element — the existing GEP lowering
uses an 8-byte stride, which matches both fields' size.

Heap rather than stack: the packed value escapes its packing function
(returning a ``Some<bound>`` to the host, threading it through other
calls, etc.), so the storage has to outlive the producing frame. The
allocations are currently leaked — there's no destructor or GC — and
the inner value buffer is sized in 8-byte chunks because the existing
``memory.heap_allocate`` lowering hard-codes a doubles-stride malloc.
Both are tracked under "Existentials" in TODO.md.

The final result is a ``ChainOp`` rather than a ``memory.load``: at the
LLVM level, ``Some<bound>`` is non-register-passable and flows as a
``ptr``, so the heap pointer *is* the value. Emitting an actual ``load``
would read only the first 8 bytes of the 16-byte struct. The chain op
re-types the heap pointer as ``Some<bound>`` while pinning a use-def
edge to the last store so the chain executes in order.

No direct LLVM emitter needed: the result of this pass is plain memory
ops that ``MemoryToLLVM`` already handles, plus a ``ChainOp`` which
codegen treats as a no-op pass-through of its left-hand value.
"""

from __future__ import annotations

import dgen
from dgen.dialects import existential, memory
from dgen.dialects.builtin import ChainOp, TypeOp
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import TypeType, constant


def _eight_byte_units(byte_size: int) -> int:
    """Number of 8-byte chunks needed to hold ``byte_size`` bytes (rounded up).

    The existing ``memory.heap_allocate`` lowering hardcodes a malloc
    of ``count * 8`` bytes, so any allocation has to be expressed in
    8-byte units. Round up so we never under-allocate. The minimum is
    one chunk so a 0-byte type still gets a real (if useless) pointer.
    """
    return max(1, (byte_size + 7) // 8)


class ExistentialToMemory(Pass):
    allow_unregistered_ops = True

    @lowering_for(existential.PackOp)
    def lower_pack(self, op: existential.PackOp) -> dgen.Value | None:
        bound = constant(op.bound)
        assert isinstance(bound, dgen.Type)
        value_type = constant(op.value.type)
        assert isinstance(value_type, dgen.Type)
        some_type = existential.Some(bound=bound)
        some_ref_type = memory.Reference(element_type=some_type)

        # Heap-allocate the Some struct (16 bytes = 2 × 8-byte fields).
        some_ref = memory.HeapAllocateOp(
            element_type=some_type,
            count=Index().constant(2),
            type=some_ref_type,
        )

        # Witness type as a runtime TypeType constant. ``builtin.type(value)``
        # is the staging-friendly way to spell "the type of value as a
        # first-class SSA value"; ``BuiltinToLLVM`` later folds it to a
        # constant pointer at codegen time.
        witness = TypeOp(value=op.value, type=TypeType())

        # Store the witness at offset 0 (the allocation pointer itself).
        witness_store = memory.StoreOp(
            mem=some_ref,
            value=witness,
            ptr=some_ref,
        )

        # Heap-allocate storage for the inner value and write the bytes.
        inner_ref_type = memory.Reference(element_type=value_type)
        inner_count = _eight_byte_units(value_type.__layout__.byte_size)
        inner_ref = memory.HeapAllocateOp(
            element_type=value_type,
            count=Index().constant(inner_count),
            type=inner_ref_type,
        )
        inner_store = memory.StoreOp(
            mem=witness_store,
            value=op.value,
            ptr=inner_ref,
        )

        # Pointer to the Some's value field (offset 8 = element index 1
        # under the existing GEP lowering's 8-byte stride).
        value_field_ref = memory.OffsetOp(
            ptr=some_ref,
            index=Index().constant(1),
            type=some_ref_type,
        )

        # Store the inner reference at offset 8.
        ref_store = memory.StoreOp(
            mem=inner_store,
            value=inner_ref,
            ptr=value_field_ref,
        )

        # Re-type the heap pointer as ``Some<bound>`` and pin the chain
        # dependency on the stores. ``Some<bound>`` is non-register-passable
        # so the LLVM-level value is "ptr" — the same shape as the
        # heap pointer — meaning no actual load is needed.
        return ChainOp(lhs=some_ref, rhs=ref_store, type=some_type)
