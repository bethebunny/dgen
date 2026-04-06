"""Lower C lvalue ops to memory dialect ops.

This pass eliminates lvalue ops by converting them to stack allocations,
loads, and stores. Memory ordering is carried by the use-def graph itself:
each LvalueVarOp.source points to the prior operation on that variable
(the prior AssignOp, which the pass replaces with a StoreOp). After
replace_uses_of runs, the source becomes the StoreOp — exactly the mem
token the next operation needs.

The per-variable ordering chain emerges naturally:

    alloca → store_init → load_1 → store_2 → load_2 → ...

because each op's source/mem traces back to the prior op on that variable.
Operations on different variables have no spurious dependencies.
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
    """Lower lvalue ops to memory dialect ops.

    Memory ordering comes from the use-def graph: each LvalueVarOp.source
    chains to the prior operation on that variable. After prior ops are
    lowered and replace_uses_of runs, source becomes the StoreOp/LoadOp
    that replaced them — the correct mem token.
    """

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
        # op.lvalue.source is the prior op on this variable. After
        # replace_uses_of, it's the StoreOp/LoadOp that replaced the
        # prior AssignOp/LvalueToRvalueOp. For the first assignment,
        # source is the init value (ConstantOp or BlockArgument) —
        # use the alloca as mem in that case.
        mem = self._mem_from_source(name, op.lvalue.source)
        store = memory.StoreOp(mem=mem, value=op.rvalue, ptr=alloca)
        return store

    @lowering_for(LvalueToRvalueOp)
    def lower_lvalue_to_rvalue(self, op: LvalueToRvalueOp) -> dgen.Value | None:
        if not isinstance(op.lvalue, LvalueVarOp):
            return None
        name = _var_name(op.lvalue)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.lvalue.source
        mem = self._mem_from_source(name, op.lvalue.source)
        return memory.LoadOp(mem=mem, ptr=alloca, type=op.type)

    def _mem_from_source(self, name: str, source: dgen.Value) -> dgen.Value:
        """Derive the mem token from a LvalueVarOp's source.

        If source is a memory op (StoreOp, LoadOp) — it's the replaced
        prior operation on this variable. Use it directly as mem.
        Otherwise (ConstantOp, BlockArgument) — this is the first
        operation on this variable. Use the alloca.
        """
        if isinstance(source, (memory.StoreOp, memory.LoadOp)):
            return source
        return self._alloca[name]

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
