"""Lower C lvalue ops to memory dialect ops.

Memory ordering is carried by the use-def graph: each LvalueVarOp.source
points to the prior operation on that variable. After replace_uses_of
runs on earlier ops, sources become StoreOp/LoadOp/PackOp/control-flow
values — any of these serve as mem tokens for the next operation.

Only BlockArgument and ConstantOp indicate the first operation on a
variable — those fall back to the alloca as the initial mem token.

Reads depend on the latest write. Writes depend on all pending reads
(via a PackOp source). Operations on different variables are independent.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.dialects import memory
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import Constant

from dcc.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp

# After replace_uses_of, a LvalueVarOp.source that was originally an
# AssignOp/LvalueToRvalueOp becomes the StoreOp/LoadOp/PackOp that
# replaced it. Control flow ops (IfOp, WhileOp, RegionOp) also appear
# as ordering fences when variables are read/written inside control flow.
# All of these are valid mem tokens — use them directly.
# Only BlockArgument and Constant indicate the first operation on a
# variable — those fall back to the alloca as the initial mem token.
_INITIAL_VALUE_TYPES = (BlockArgument, Constant)


def _var_name(lvalue: LvalueVarOp) -> str:
    """Extract the variable name string from a LvalueVarOp."""
    return lvalue.var_name.__constant__.to_json()


class CLvalueToMemory(Pass):
    """Lower lvalue ops to memory dialect ops."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._alloca: dict[str, memory.StackAllocateOp] = {}
        self._alloca_owner: dict[str, dgen.Block] = {}
        self._block_stack: list[dgen.Block] = []

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._alloca = {}
        self._alloca_owner = {}
        self._block_stack = []
        return super().run(value, compiler)

    def _lower_block(self, block: dgen.Block) -> None:
        self._block_stack.append(block)
        super()._lower_block(block)
        self._block_stack.pop()

    def _capture_alloca(self, name: str, alloca: memory.StackAllocateOp) -> None:
        """Add alloca as a capture to every block in the stack that isn't
        the block where the alloca was created."""
        owner = self._alloca_owner[name]
        for block in self._block_stack:
            if block is not owner and alloca not in block.captures:
                block.captures = [*block.captures, alloca]

    @lowering_for(AssignOp)
    def lower_assign(self, op: AssignOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._ensure_alloca(name, op.rvalue.type)
        self._capture_alloca(name, alloca)
        source = op.lvalue.source
        mem = alloca if isinstance(source, _INITIAL_VALUE_TYPES) else source
        return memory.StoreOp(mem=mem, value=op.rvalue, ptr=alloca)

    @lowering_for(LvalueToRvalueOp)
    def lower_lvalue_to_rvalue(self, op: LvalueToRvalueOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.lvalue.source
        self._capture_alloca(name, alloca)
        source = op.lvalue.source
        mem = alloca if isinstance(source, _INITIAL_VALUE_TYPES) else source
        return memory.LoadOp(mem=mem, ptr=alloca, type=op.type)

    def _ensure_alloca(
        self, name: str, element_type: dgen.Type
    ) -> memory.StackAllocateOp:
        """Get or create a stack allocation for a variable."""
        alloca = self._alloca.get(name)
        if alloca is not None:
            return alloca
        alloca = memory.StackAllocateOp(
            element_type=element_type,
            type=memory.Reference(element_type=element_type),
        )
        self._alloca[name] = alloca
        # Allocas belong to the outermost block (function body), not
        # wherever they're first encountered. The first block in the
        # stack is the root wrapper; the second is the function body.
        self._alloca_owner[name] = self._block_stack[1]
        return alloca
