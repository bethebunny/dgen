"""Lower ndbuffer dialect to memory dialect.

Converts shaped N-dimensional buffer ops to flat pointer ops with
index linearization.

    ndbuffer.alloc(shape) → record.pack([heap_allocate(total), total])
    ndbuffer.load(mem, buf, [i, j]) → memory.load(mem, memory.offset(deref(buf), linear))
    ndbuffer.store(mem, val, buf, [i, j]) → memory.store(mem, val, memory.offset(...))
    ndbuffer.dealloc → memory.deallocate
    ndbuffer.print_memref → llvm.call<"print_memref">
"""

from __future__ import annotations

from math import prod

import dgen
from dgen.builtins import PackOp, pack
from dgen.dialects import algebra, function, llvm, memory, ndbuffer
from dgen.dialects.builtin import ExternOp, Nil, String
from dgen.dialects.record import GetOp as RecordGetOp, PackOp as RecordPackOp
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import constant


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
    """Extract the data pointer from a shaped buffer (NDBuffer or Tensor)."""
    return RecordGetOp(
        index=Index().constant(0),
        record=val,
        type=memory.Reference(element_type=Float64()),
    )


class NDBufferToMemory(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        # Track shapes for allocs that have been lowered.
        self._shapes: dict[dgen.Value, list[int]] = {}

    def _resolve_shape(self, val: dgen.Value) -> list[int]:
        """Get shape from tracked alloc or from the type's shape parameter."""
        if val in self._shapes:
            return self._shapes[val]
        result = constant(val.type.shape)
        assert isinstance(result, list)
        return result

    @lowering_for(ndbuffer.AllocOp)
    def lower_alloc(self, op: ndbuffer.AllocOp) -> dgen.Value | None:
        assert isinstance(op.type, ndbuffer.NDBuffer)
        shape = constant(op.type.shape)
        assert isinstance(shape, list)
        total = prod(shape)
        dtype = Float64()
        alloc = memory.HeapAllocateOp(
            element_type=dtype,
            count=Index().constant(total),
            type=memory.Reference(element_type=dtype),
        )
        result = RecordPackOp(
            values=pack([alloc, Index().constant(total)]),
            type=op.type,
        )
        self._shapes[result] = shape
        return result

    @lowering_for(ndbuffer.DeallocOp)
    def lower_dealloc(self, op: ndbuffer.DeallocOp) -> dgen.Value | None:
        return memory.DeallocateOp(mem=op.mem, ptr=_deref(op.buffer))

    @lowering_for(ndbuffer.LoadOp)
    def lower_load(self, op: ndbuffer.LoadOp) -> dgen.Value | None:
        ptr = _deref(op.buffer)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.buffer)
        offset = _linearize(shape, list(op.indices))
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ptr.type)
        return memory.LoadOp(mem=op.mem, ptr=offset_ptr, type=op.type)

    @lowering_for(ndbuffer.StoreOp)
    def lower_store(self, op: ndbuffer.StoreOp) -> dgen.Value | None:
        ptr = _deref(op.buffer)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.buffer)
        offset = _linearize(shape, list(op.indices))
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ptr.type)
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
