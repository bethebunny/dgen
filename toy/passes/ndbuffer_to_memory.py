"""Lower ndbuffer dialect to memory dialect.

Converts shaped N-dimensional buffer ops to flat pointer ops with
index linearization.

    ndbuffer.alloc(shape) → memory.heap_allocate(total)
    ndbuffer.load(buf, [i, j]) → memory.load(memory.offset(buf, i*stride+j))
    ndbuffer.store(val, buf, [i, j]) → memory.store(val, memory.offset(buf, ...))
    ndbuffer.dealloc → memory.deallocate
    ndbuffer.print_memref → llvm.call<"print_memref">
"""

from __future__ import annotations

from math import prod

import dgen
from dgen.dialects import algebra, builtin, llvm, memory
from dgen.dialects.builtin import String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import ndbuffer, toy


def _shape_of(val: dgen.Value) -> list[int]:
    assert isinstance(val.type, (toy.Tensor, ndbuffer.NDBuffer))
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
            algebra.MultiplyOp(
                left=idx, right=ConstantOp(value=stride, type=Index()), type=Index()
            )
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
    """If val is a tensor constant, load its data pointer."""
    if not isinstance(val.type, toy.Tensor):
        return val
    return memory.LoadOp(ptr=val, type=memory.Reference(element_type=Float64()))


class NDBufferToMemory(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        # Track shapes for allocs that have been lowered to References.
        self._shapes: dict[int, list[int]] = {}

    def _resolve_shape(self, val: dgen.Value) -> list[int]:
        """Get shape from NDBuffer type or from tracked alloc shapes."""
        if id(val) in self._shapes:
            return self._shapes[id(val)]
        return _shape_of(val)

    @lowering_for(ndbuffer.AllocOp)
    def lower_alloc(self, op: ndbuffer.AllocOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.type, ndbuffer.NDBuffer)
        shape = _shape_of(op)
        total = prod(shape)
        dtype = Float64()
        alloc = memory.HeapAllocateOp(
            element_type=dtype,
            count=ConstantOp(value=total, type=Index()),
            type=memory.Reference(element_type=dtype),
        )
        self._shapes[id(alloc)] = shape
        rewriter.replace_uses(op, alloc)
        return True

    @lowering_for(ndbuffer.DeallocOp)
    def lower_dealloc(self, op: ndbuffer.DeallocOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, memory.DeallocateOp(ptr=op.input))
        return True

    @lowering_for(ndbuffer.LoadOp)
    def lower_load(self, op: ndbuffer.LoadOp, rewriter: Rewriter) -> bool:
        ptr = _deref(op.memref)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.memref)
        offset = _linearize(shape, list(op.indices.values))
        ref_type = (
            ptr.type
            if isinstance(ptr.type, memory.Reference)
            else memory.Reference(element_type=Float64())
        )
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ref_type)
        rewriter.replace_uses(op, memory.LoadOp(ptr=offset_ptr, type=op.type))
        return True

    @lowering_for(ndbuffer.StoreOp)
    def lower_store(self, op: ndbuffer.StoreOp, rewriter: Rewriter) -> bool:
        ptr = _deref(op.memref)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.memref)
        offset = _linearize(shape, list(op.indices.values))
        ref_type = (
            ptr.type
            if isinstance(ptr.type, memory.Reference)
            else memory.Reference(element_type=Float64())
        )
        offset_ptr = memory.OffsetOp(ptr=ptr, index=offset, type=ref_type)
        rewriter.replace_uses(op, memory.StoreOp(value=op.value, ptr=offset_ptr))
        return True

    @lowering_for(ndbuffer.PrintMemrefOp)
    def lower_print(self, op: ndbuffer.PrintMemrefOp, rewriter: Rewriter) -> bool:
        ptr = _deref(op.input)
        shape = self._resolve_shape(op.input)
        size = prod(shape)
        pack = PackOp(
            values=[ptr, ConstantOp(value=size, type=Index())],
            type=builtin.List(element_type=ptr.type),
        )
        rewriter.replace_uses(
            op, llvm.CallOp(callee=String().constant("print_memref"), args=pack)
        )
        return True
