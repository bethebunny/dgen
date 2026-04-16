"""Helpers for emitting memory allocation + store patterns."""

from __future__ import annotations

import dgen
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index
from dgen.layout import align_up
from dgen.type import constant


def heap_box(value: dgen.Value) -> dgen.Value:
    """Heap-allocate space for *value*, store it, return the reference.

    Returns a ``ChainOp`` whose value is the ``Reference`` pointer with
    a use-def dependency on the store (so the store is not dead).
    """
    element_type = value.type
    resolved = constant(element_type)
    assert isinstance(resolved, dgen.Type)
    count = max(1, align_up(resolved.__layout__.byte_size, 8))
    ref_type = memory.Reference(element_type=element_type)
    ref = memory.HeapAllocateOp(
        element_type=element_type, count=Index().constant(count), type=ref_type
    )
    store = memory.StoreOp(mem=ref, value=value, ptr=ref)
    return ChainOp(lhs=ref, rhs=store, type=ref_type)


def stack_box(value: dgen.Value) -> dgen.Value:
    """Stack-allocate space for *value*, store it, load it back.

    Forces materialisation through memory and returns a by-value result
    with the original type.
    """
    ref_type = memory.Reference(element_type=value.type)
    ref = memory.StackAllocateOp(element_type=value.type, type=ref_type)
    store = memory.StoreOp(mem=ref, value=value, ptr=ref)
    return memory.LoadOp(mem=store, ptr=ref, type=value.type)
