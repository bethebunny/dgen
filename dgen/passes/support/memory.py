"""Helpers for emitting memory allocation + store patterns."""

from __future__ import annotations

import dgen
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp
from dgen.dialects.index import Index


def heap_box(value: dgen.Value) -> dgen.Value:
    """Heap-allocate a single-element ``Buffer`` for *value*, store it,
    and return the buffer (typed ``memory.Buffer<value.type>``).

    Used to lift an SSA value into a heap pointer that downstream code
    can pass around as a ``Some<T>`` / ``Any`` / similar pointer-shaped
    type — Buffer is non-linear, so the resulting pointer can be aliased
    freely (matching Some/Any's existing semantics).
    """
    element_type = value.type
    buf_type = memory.Buffer(element_type=element_type)
    buf = memory.BufferAllocateOp(
        element_type=element_type, count=Index().constant(1), type=buf_type
    )
    store = memory.BufferStoreOp(
        mem=buf, buf=buf, index=Index().constant(0), value=value
    )
    return ChainOp(lhs=buf, rhs=store, type=buf_type)
