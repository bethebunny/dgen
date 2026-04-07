"""Lower C lvalue ops to memory dialect ops.

Memory ordering is carried by the use-def graph: each LvalueVarOp.source
points to the prior operation on that variable. After replace_uses_of
runs on earlier ops, sources become StoreOp/LoadOp/PackOp values —
exactly the mem tokens the next operation needs.

Reads depend on the latest write. Writes depend on all pending reads
(via a PackOp source). Operations on different variables are independent.
"""

from __future__ import annotations

import dgen
from dgen.dialects import memory
from dgen.module import PackOp
from dgen.passes.pass_ import Pass, lowering_for

from dcc2.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp

# After replace_uses_of, a LvalueVarOp.source that was originally an
# AssignOp/LvalueToRvalueOp becomes the StoreOp/LoadOp/PackOp that
# replaced it. These types are valid mem tokens — use them directly.
# Anything else (BlockArgument, ConstantOp) means this is the first
# operation on the variable — use the alloca as the initial mem token.
_MEM_TOKEN_TYPES = (memory.StoreOp, memory.LoadOp, PackOp)


def _var_name(lvalue: LvalueVarOp) -> str:
    """Extract the variable name string from a LvalueVarOp."""
    return lvalue.var_name.__constant__.to_json()


class CLvalueToMemory(Pass):
    """Lower lvalue ops to memory dialect ops."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._alloca: dict[str, memory.StackAllocateOp] = {}

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._alloca = {}
        return super().run(value, compiler)

    @lowering_for(AssignOp)
    def lower_assign(self, op: AssignOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._ensure_alloca(name, op.rvalue.type)
        source = op.lvalue.source
        mem = source if isinstance(source, _MEM_TOKEN_TYPES) else alloca
        return memory.StoreOp(mem=mem, value=op.rvalue, ptr=alloca)

    @lowering_for(LvalueToRvalueOp)
    def lower_lvalue_to_rvalue(self, op: LvalueToRvalueOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.lvalue.source
        source = op.lvalue.source
        mem = source if isinstance(source, _MEM_TOKEN_TYPES) else alloca
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
        return alloca
