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
from typing import TYPE_CHECKING

import dgen
from dgen.dialects import algebra, llvm, memory
from dgen.dialects.builtin import String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, Module, PackOp, pack
from dgen.passes.pass_ import Pass, lowering_for
from dgen.dialects import ndbuffer
from toy.dialects import toy

if TYPE_CHECKING:
    from dgen.compiler import Compiler


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
    return memory.LoadOp(
        mem=val, ptr=val, type=memory.Reference(element_type=Float64())
    )


class NDBufferToMemory(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        # Track shapes for allocs that have been lowered to References.
        self._shapes: dict[int, list[int]] = {}

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        self._shapes.clear()
        return super().run(module, compiler)

    def _resolve_shape(self, val: dgen.Value) -> list[int]:
        """Get shape from NDBuffer type or from tracked alloc shapes."""
        if id(val) in self._shapes:
            return self._shapes[id(val)]
        return _shape_of(val)

    @lowering_for(ndbuffer.AllocOp)
    def lower_alloc(self, op: ndbuffer.AllocOp) -> dgen.Value | None:
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
        return alloc

    @lowering_for(ndbuffer.DeallocOp)
    def lower_dealloc(self, op: ndbuffer.DeallocOp) -> dgen.Value | None:
        return memory.DeallocateOp(mem=op.mem, ptr=op.input)

    @lowering_for(ndbuffer.LoadOp)
    def lower_load(self, op: ndbuffer.LoadOp) -> dgen.Value | None:
        ptr = _deref(op.memref)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.memref)
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
        ptr = _deref(op.memref)
        assert isinstance(op.indices, PackOp)
        shape = self._resolve_shape(op.memref)
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
        ptr = _deref(op.input)
        shape = self._resolve_shape(op.input)
        size = prod(shape)
        args = pack([ptr, ConstantOp(value=size, type=Index())])
        return llvm.CallOp(callee=String().constant("print_memref"), args=args)
