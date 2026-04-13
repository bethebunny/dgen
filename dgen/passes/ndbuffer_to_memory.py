"""Lower ndbuffer dialect to memory dialect.

Converts shaped N-dimensional buffer ops to flat pointer ops with
index linearization.

    ndbuffer.alloc(shape) → memory.heap_allocate(total)
    ndbuffer.load(mem, buf, [i, j]) → memory.load(mem, memory.offset(buf, i*stride+j))
    ndbuffer.store(mem, val, buf, [i, j]) → memory.store(mem, val, memory.offset(buf, ...))
    ndbuffer.dealloc → memory.deallocate
    ndbuffer.print_memref → llvm.call<"print_memref">
"""

from __future__ import annotations

from math import prod

import dgen
from dgen.builtins import PackOp, pack
from dgen.dialects import algebra, function, llvm, memory, ndbuffer
from dgen.dialects.builtin import ExternOp, Nil, RecordGetOp, String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.passes.pass_ import Pass, lowering_for


def _shape_of(val: dgen.Value) -> list[int]:
    """Extract static shape from a shaped type (NDBuffer or any type with .shape)."""
    result = val.type.shape.__constant__.to_json()
    assert isinstance(result, list)
    return result


def _linearize(shape: list[int], indices: list[dgen.Value]) -> dgen.Value:
    """Compute 1-D offset from multi-dimensional indices and static shape."""
    if len(indices) == 1:
        return indices[0]
    result: dgen.Value | None = None
    for i, idx in enumerate(indices):
        stride = prod(shape[i + 1 :])
        term: dgen.Value = (
            algebra.MultiplyOp(left=idx, right=Index().constant(stride), type=Index())
            if stride != 1
            else idx
        )
        result = (
            algebra.AddOp(left=result, right=term, type=Index())
            if result is not None
            else term
        )
    assert result is not None
    return result


def _deref(val: dgen.Value) -> dgen.Value:
    """If val's type is a shaped buffer (not Reference or NDBuffer), extract the data pointer."""
    if isinstance(val.type, (memory.Reference, ndbuffer.NDBuffer)):
        return val
    # Upstream types (e.g. toy.Tensor) have a Span-shaped layout where
    # field 0 is the data pointer. Use record_get to extract it.
    return RecordGetOp(
        index=Index().constant(0),
        mem=val,
        record=val,
        type=memory.Reference(element_type=Float64()),
    )


class NDBufferToMemory(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        # Track shapes for allocs that have been lowered to References.
        self._shapes: dict[dgen.Value, list[int]] = {}

    def _resolve_shape(self, val: dgen.Value) -> list[int]:
        """Get shape from NDBuffer type or from tracked alloc shapes."""
        if val in self._shapes:
            return self._shapes[val]
        return _shape_of(val)

    @lowering_for(ndbuffer.AllocOp)
    def lower_alloc(self, op: ndbuffer.AllocOp) -> dgen.Value | None:
        assert isinstance(op.type, ndbuffer.NDBuffer)
        shape = _shape_of(op)
        total = prod(shape)
        dtype = Float64()
        alloc = memory.HeapAllocateOp(
            element_type=dtype,
            count=Index().constant(total),
            type=memory.Reference(element_type=dtype),
        )
        self._shapes[alloc] = shape
        return alloc

    @lowering_for(ndbuffer.DeallocOp)
    def lower_dealloc(self, op: ndbuffer.DeallocOp) -> dgen.Value | None:
        return memory.DeallocateOp(mem=op.mem, ptr=op.buffer)

    @lowering_for(ndbuffer.LoadOp)
    def lower_load(self, op: ndbuffer.LoadOp) -> dgen.Value | None:
        ptr = _deref(op.buffer)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.buffer)
        offset = _linearize(shape, list(op.indices))
        ref_type = (
            ptr.type
            if isinstance(ptr.type, memory.Reference)
            else memory.Reference(element_type=Float64())
        )
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ref_type)
        return memory.LoadOp(mem=op.mem, ptr=offset_ptr, type=op.type)

    @lowering_for(ndbuffer.StoreOp)
    def lower_store(self, op: ndbuffer.StoreOp) -> dgen.Value | None:
        ptr = _deref(op.buffer)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.buffer)
        offset = _linearize(shape, list(op.indices))
        ref_type = (
            ptr.type
            if isinstance(ptr.type, memory.Reference)
            else memory.Reference(element_type=Float64())
        )
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ref_type)
        return memory.StoreOp(mem=op.mem, value=op.value, ptr=offset_ptr)

    @lowering_for(ndbuffer.PrintMemrefOp)
    def lower_print(self, op: ndbuffer.PrintMemrefOp) -> dgen.Value | None:
        ptr = _deref(op.buffer)
        shape = self._resolve_shape(op.buffer)
        size = prod(shape)
        print_memref = ExternOp(
            symbol=String().constant("print_memref"),
            type=function.Function(
                arguments=pack([llvm.Ptr(), Index()]), result_type=Nil()
            ),
        )
        args = pack([ptr, Index().constant(size)])
        return function.CallOp(callee=print_memref, arguments=args, type=Nil())
