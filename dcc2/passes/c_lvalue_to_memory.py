"""Lower C lvalue ops to memory dialect ops.

This pass eliminates lvalue ops by converting them to stack allocations,
loads, and stores with mem-token threading for precise ordering.

AssignOp (on LvalueVarOp)       → StackAllocateOp + StoreOp
LvalueToRvalueOp (on LvalueVarOp) → LoadOp

Variables are identified by name. Nested-scope shadowing (two variables
named "x" in different scopes) is a Brick 6 concern — it requires inner
blocks where each block has its own pass state.
"""

from __future__ import annotations

import dgen
from dgen.dialects import memory
from dgen.passes.pass_ import Pass, lowering_for

from dcc2.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp


def _var_name(lvalue: LvalueVarOp) -> str:
    """Extract the variable name string from a LvalueVarOp."""
    return lvalue.var_name.__constant__.to_json()


class CLvalueToMemory(Pass):
    """Lower lvalue ops to memory dialect ops with mem-token threading."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._alloca: dict[str, memory.StackAllocateOp] = {}
        self._mem: dict[str, dgen.Value] = {}

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._alloca = {}
        self._mem = {}
        return super().run(value, compiler)

    @lowering_for(AssignOp)
    def lower_assign(self, op: AssignOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._ensure_alloca(name, op.rvalue.type)
        mem = self._mem_for(name)
        store = memory.StoreOp(mem=mem, value=op.rvalue, ptr=alloca)
        self._mem[name] = store
        return store

    @lowering_for(LvalueToRvalueOp)
    def lower_lvalue_to_rvalue(self, op: LvalueToRvalueOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._alloca.get(name)
        if alloca is None:
            # No alloca for this name — the source value is the value itself.
            return op.lvalue.source
        mem = self._mem_for(name)
        return memory.LoadOp(mem=mem, ptr=alloca, type=op.type)

    def _ensure_alloca(self, name: str, elem_type: dgen.Type) -> memory.StackAllocateOp:
        """Get or create a stack allocation for a variable."""
        alloca = self._alloca.get(name)
        if alloca is not None:
            return alloca
        alloca = memory.StackAllocateOp(
            element_type=elem_type,
            type=memory.Reference(element_type=elem_type),
        )
        self._alloca[name] = alloca
        return alloca

    def _mem_for(self, name: str) -> dgen.Value:
        """Get the current memory token for a variable (alloca if no prior store)."""
        return self._mem.get(name) or self._alloca[name]
