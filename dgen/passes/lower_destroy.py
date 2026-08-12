"""Lower memory.destroy to its allocation's base destructor.

Each destroy consumes an origin. This pass walks the origin's use-def
thread back to the allocation that created it. Heap destroys become
``memory.deallocate(origin, ref)``, which MemoryToLLVM lowers to a
``free`` call. Stack destroys keep no runtime action.

The walk follows store results, origin-typed chains, and unpack block
arguments back through their tuples. A thread the walk cannot resolve
(an origin from a function parameter or a branch result) raises
``UnresolvableOriginError`` rather than silently leaking. Resolving
those threads is the attach/destructor-stack design in
``docs/origins.md``.

The reference for a heap free is the allocation unpack's first block
argument. When the destroy sits in a deeper block, this pass adds the
reference to the capture list of each block between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dgen
from dgen.block import Block, BlockArgument
from dgen.dialects import memory
from dgen.dialects.builtin import ChainOp, Nil, UnpackOp
from dgen.ir.traversal import all_values
from dgen.passes.pass_ import Pass, lowering_for

if TYPE_CHECKING:
    from dgen.passes.compiler import Compiler


class UnresolvableOriginError(Exception):
    """A destroy's origin thread cannot be statically resolved to its
    allocation."""


class LowerDestroy(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._arg_owner: dict[BlockArgument, tuple[dgen.Op, int, Block]] = {}
        self._block_stack: list[Block] = []

    def run(self, value: dgen.Value, compiler: Compiler[object]) -> dgen.Value:
        self._arg_owner = {}
        self._block_stack = []
        for v in all_values(value):
            if not isinstance(v, dgen.Op):
                continue
            for _, block in v.blocks:
                for i, arg in enumerate(block.args):
                    self._arg_owner[arg] = (v, i, block)
        return super().run(value, compiler)

    def _lower_block(self, block: Block) -> None:
        self._block_stack.append(block)
        super()._lower_block(block)
        self._block_stack.pop()

    @lowering_for(memory.DestroyOp)
    def lower_destroy(self, op: memory.DestroyOp) -> dgen.Value | None:
        alloc, ref, ref_block = self._resolve_allocation(op.origin)
        if isinstance(alloc, memory.StackAllocateOp):
            # No runtime action yet. LLVM lifetime markers are future
            # work; the alloca is reclaimed with the frame.
            return ChainOp(lhs=Nil().constant(None), rhs=op.origin, type=Nil())
        self._thread_capture(ref, ref_block)
        return memory.DeallocateOp(origin=op.origin, ref=ref)

    def _resolve_allocation(
        self, origin: dgen.Value
    ) -> tuple[dgen.Op, dgen.Value, Block]:
        """Walk *origin*'s thread to its allocation. Returns the
        allocation op, the reference value, and the block binding it."""
        v = origin
        while True:
            if isinstance(v, memory.StoreOp):
                v = v.origin
            elif isinstance(v, ChainOp):
                v = v.lhs
            elif isinstance(v, BlockArgument):
                owner = self._arg_owner.get(v)
                if owner is None:
                    break
                owner_op, index, block = owner
                if not isinstance(owner_op, UnpackOp) or index != 1:
                    break
                tup = owner_op.tuple
                if isinstance(tup, memory.LoadOp):
                    v = tup.origin
                elif isinstance(tup, (memory.HeapAllocateOp, memory.StackAllocateOp)):
                    return tup, block.args[0], block
                else:
                    break
            else:
                break
        raise UnresolvableOriginError(
            f"cannot statically resolve the allocation of origin "
            f"%{origin.name} destroyed by this op; origins from function "
            f"parameters or branch results await the destructor design "
            f"in docs/origins.md"
        )

    def _thread_capture(self, ref: dgen.Value, ref_block: Block) -> None:
        """Capture *ref* into each block between its binding block and
        the destroy's block."""
        idx = self._block_stack.index(ref_block)
        for block in self._block_stack[idx + 1 :]:
            if ref not in block.captures:
                block.captures = [*block.captures, ref]
