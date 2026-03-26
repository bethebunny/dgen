"""Lower memory dialect ops to LLVM dialect ops.

memory.alloc(shape) → llvm.call<"malloc">(total * 8)
memory.load(ref, [i, j]) → llvm.gep + llvm.load (with index linearization)
memory.store(val, ref, [i, j]) → llvm.gep + llvm.store
memory.dealloc → no-op
memory.print_memref → llvm.call<"print_memref">
"""

from __future__ import annotations

from math import prod

import dgen
from dgen.dialects import builtin, index, llvm
from dgen.dialects.builtin import Nil, String
from dgen.module import ConstantOp, PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import memory, toy

_PTR_TYPE = llvm.Ptr()


def _shape_of(val: dgen.Value) -> list[int]:
    """Extract the static shape from a Tensor or MemRef typed value."""
    assert isinstance(val.type, (toy.Tensor, memory.MemRef))
    result = val.type.shape.__constant__.to_json()
    assert isinstance(result, list)
    return result


def _linearize(shape: list[int], indices: list[dgen.Value]) -> dgen.Value:
    """Compute a 1-D offset from multi-dimensional indices and a static shape."""
    if len(indices) == 1:
        return indices[0]
    result: dgen.Value | None = None
    for i, idx in enumerate(indices):
        stride = prod(shape[i + 1 :])
        term: dgen.Value = (
            llvm.MulOp(lhs=idx, rhs=ConstantOp(value=stride, type=index.Index()))
            if stride != 1
            else idx
        )
        result = llvm.AddOp(lhs=result, rhs=term) if result is not None else term
    assert result is not None
    return result


def _deref(val: dgen.Value) -> dgen.Value:
    """If val is a tensor, load its data pointer. Otherwise return as-is."""
    if not isinstance(val.type, toy.Tensor):
        return val
    return llvm.LoadOp(ptr=val, type=_PTR_TYPE)


def _extract_indices(pack: dgen.Value) -> list[dgen.Value]:
    assert isinstance(pack, PackOp)
    return list(pack.values)


class MemoryToLLVM(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}

    @lowering_for(memory.AllocOp)
    def lower_alloc(self, op: memory.AllocOp, rewriter: Rewriter) -> bool:
        assert isinstance(op.type, memory.MemRef)
        shape = _shape_of(op)
        total = prod(shape)
        byte_count = ConstantOp(value=total * 8, type=index.Index())
        malloc_args = PackOp(
            values=[byte_count], type=builtin.List(element_type=index.Index())
        )
        malloc_op = llvm.CallOp(
            callee=String().constant("malloc"), args=malloc_args, type=_PTR_TYPE
        )
        self.alloc_shapes[malloc_op] = shape
        rewriter.replace_uses(op, malloc_op)
        return True

    @lowering_for(memory.DeallocOp)
    def lower_dealloc(self, op: memory.DeallocOp, rewriter: Rewriter) -> bool:
        rewriter.replace_uses(op, ConstantOp(value=0, type=Nil()))
        return True

    def _resolve_shape(self, ref: dgen.Value) -> list[int]:
        if ref in self.alloc_shapes:
            return self.alloc_shapes[ref]
        return _shape_of(ref)

    @lowering_for(memory.LoadOp)
    def lower_load(self, op: memory.LoadOp, rewriter: Rewriter) -> bool:
        ref = _deref(op.memref)
        indices = _extract_indices(op.indices)
        shape = self._resolve_shape(op.memref)
        gep = llvm.GepOp(base=ref, index=_linearize(shape, indices))
        rewriter.replace_uses(op, llvm.LoadOp(ptr=gep))
        return True

    @lowering_for(memory.StoreOp)
    def lower_store(self, op: memory.StoreOp, rewriter: Rewriter) -> bool:
        ref = _deref(op.memref)
        indices = _extract_indices(op.indices)
        shape = self._resolve_shape(op.memref)
        gep = llvm.GepOp(base=ref, index=_linearize(shape, indices))
        rewriter.replace_uses(op, llvm.StoreOp(value=op.value, ptr=gep))
        return True

    @lowering_for(memory.PrintMemrefOp)
    def lower_print(self, op: memory.PrintMemrefOp, rewriter: Rewriter) -> bool:
        ptr = _deref(op.input)
        shape = self._resolve_shape(op.input)
        size = prod(shape)
        pack = PackOp(
            values=[ptr, ConstantOp(value=size, type=index.Index())],
            type=builtin.List(element_type=ptr.type),
        )
        rewriter.replace_uses(
            op, llvm.CallOp(callee=String().constant("print_memref"), args=pack)
        )
        return True
