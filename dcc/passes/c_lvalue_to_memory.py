"""Lower lvalue ops to memory dialect ops.

This pass handles the new lvalue variable model:

lvalue_var           → stack_allocate (lazily, one per variable name)
lvalue_to_rvalue     → load (with per-variable mem token)
lvalue_assign        → store (updates per-variable mem token)
lvalue_compound_assign → load + binop + store
lvalue_pre/post_increment/decrement → load + add/sub + store

It runs before CToMemory, which handles old-style variable ops during
the incremental migration.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, memory
from dgen.dialects.builtin import Nil
from dgen.module import ConstantOp, string_value
from dgen.passes.pass_ import Pass, lowering_for

from dcc.dialects.c import (
    LvalueAssignOp,
    LvalueCompoundAssignOp,
    LvaluePostDecrementOp,
    LvaluePostIncrementOp,
    LvaluePreDecrementOp,
    LvaluePreIncrementOp,
    LvalueToRvalueOp,
    LvalueVarOp,
    ModuloOp,
    ShiftLeftOp,
    ShiftRightOp,
)

_BINOP_TABLE: dict[str, type[dgen.Op]] = {
    "+": algebra.AddOp,
    "-": algebra.SubtractOp,
    "*": algebra.MultiplyOp,
    "/": algebra.DivideOp,
    "%": ModuloOp,
    "&": algebra.MeetOp,
    "|": algebra.JoinOp,
    "^": algebra.SymmetricDifferenceOp,
    "<<": ShiftLeftOp,
    ">>": ShiftRightOp,
}


class CLvalueToMemory(Pass):
    """Lower lvalue ops to memory dialect ops.

    Overrides ``_lower_block`` to thread mem tokens across control-flow
    boundaries.  After recursing into an IfOp/WhileOp's child blocks,
    any variable whose mem token changed inside is updated to point to
    the control-flow op itself.  This makes subsequent loads depend on
    the control-flow op, keeping the inner stores reachable from
    ``block.result``.
    """

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._alloca: dict[str, dgen.Value] = {}
        self._mem: dict[str, dgen.Value] = {}
        # Reverse mapping: alloca value → variable name, so downstream
        # handlers can recover the variable name after replace_uses_of
        # has replaced lvalue_var with the alloca.
        self._alloca_to_name: dict[dgen.Value, str] = {}
        # Track which block each alloca was created in, so we can add
        # captures when an alloca is referenced from a different block.
        self._alloca_block: dict[str, dgen.Block] = {}
        # The block currently being lowered (for adding captures).
        self._current_block: dgen.Block | None = None

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._alloca = {}
        self._mem = {}
        self._alloca_to_name = {}
        self._alloca_block = {}
        self._current_block = None
        return super().run(value, compiler)

    def _lower_block(self, block: dgen.Block) -> None:
        """Lower ops, threading mem tokens across control-flow boundaries."""
        outer_block = self._current_block
        self._current_block = block

        # Save and reset per-variable state when entering a function body
        # (function bodies have args — they're the top-level scope for a function).
        saved_state: (
            tuple[
                dict[str, dgen.Value],
                dict[str, dgen.Value],
                dict[dgen.Value, str],
                dict[str, dgen.Block | None],
            ]
            | None
        ) = None
        if outer_block is not None and block.args:
            saved_state = (
                self._alloca,
                self._mem,
                self._alloca_to_name,
                self._alloca_block,
            )
            self._alloca = {}
            self._mem = {}
            self._alloca_to_name = {}
            self._alloca_block = {}

        for v in block.values:
            result = self._dispatch_handlers(v)
            effective = result or v

            # Snapshot mem state before recursing into child blocks.
            child_blocks = list(effective.blocks)
            saved_mem = dict(self._mem) if child_blocks else None

            for _, child_block in child_blocks:
                self._lower_block(child_block)

            # After processing a control-flow op's inner blocks, redirect
            # mem tokens for any variable modified inside to the control-flow
            # op.  The subsequent load will depend on the control-flow op,
            # making the inner block (and its stores) reachable.
            if saved_mem is not None:
                for name, token in list(self._mem.items()):
                    if token is not saved_mem.get(name):
                        self._mem[name] = effective

            if result is not None:
                block.replace_uses_of(v, result)

        if saved_state is not None:
            self._alloca, self._mem, self._alloca_to_name, self._alloca_block = (
                saved_state
            )
        self._current_block = outer_block

    def _get_or_create_alloca(self, name: str, var_type: dgen.Type) -> dgen.Value:
        """Return the alloca for a variable, creating it on first access."""
        if name not in self._alloca:
            alloca = memory.StackAllocateOp(
                element_type=var_type,
                type=memory.Reference(element_type=var_type),
            )
            self._alloca[name] = alloca
            self._mem[name] = alloca
            self._alloca_to_name[alloca] = name
            self._alloca_block[name] = self._current_block
        return self._alloca[name]

    def _get_mem(self, name: str) -> dgen.Value:
        return self._mem.get(
            name, self._alloca.get(name, ConstantOp(value=None, type=Nil()))
        )

    def _name_for_alloca(self, alloca: dgen.Value) -> str:
        return self._alloca_to_name[alloca]

    # --- Lvalue var ---

    @lowering_for(LvalueVarOp)
    def lower_lvalue_var(self, op: LvalueVarOp) -> dgen.Value | None:
        name = string_value(op.variable_name)
        first_access = name not in self._alloca
        alloca = self._get_or_create_alloca(name, op.type)
        # If the alloca was created in a different block, it must be a
        # capture of the current block so codegen doesn't re-emit it.
        if self._alloca_block[name] is not self._current_block:
            if alloca not in self._current_block.captures:
                self._current_block.captures.append(alloca)
        # For function parameters (BlockArguments), store the parameter
        # value into the alloca on first access so reads get the arg value.
        if first_access and isinstance(op.source, BlockArgument):
            store = memory.StoreOp(mem=alloca, value=op.source, ptr=alloca)
            self._mem[name] = store
        return alloca

    # --- Lvalue to rvalue (load) ---

    @lowering_for(LvalueToRvalueOp)
    def lower_lvalue_to_rvalue(self, op: LvalueToRvalueOp) -> dgen.Value | None:
        alloca = op.lvalue
        name = self._name_for_alloca(alloca)
        mem = self._get_mem(name)
        return memory.LoadOp(mem=mem, ptr=alloca, type=op.type)

    # --- Lvalue assign (store) ---

    @lowering_for(LvalueAssignOp)
    def lower_lvalue_assign(self, op: LvalueAssignOp) -> dgen.Value | None:
        alloca = op.lvalue
        name = self._name_for_alloca(alloca)
        mem = self._get_mem(name)
        store = memory.StoreOp(mem=mem, value=op.rvalue, ptr=alloca)
        self._mem[name] = store
        return store

    # --- Compound assign ---

    @lowering_for(LvalueCompoundAssignOp)
    def lower_lvalue_compound_assign(
        self, op: LvalueCompoundAssignOp
    ) -> dgen.Value | None:
        alloca = op.lvalue
        name = self._name_for_alloca(alloca)
        mem = self._get_mem(name)
        elem_type = op.type
        load = memory.LoadOp(mem=mem, ptr=alloca, type=elem_type)
        operator = string_value(op.operator)
        binop_cls = _BINOP_TABLE.get(operator)
        if binop_cls is None:
            return None
        if "left" in binop_cls.__dataclass_fields__:
            result = binop_cls(left=load, right=op.rvalue, type=elem_type)
        else:
            result = binop_cls(lhs=load, rhs=op.rvalue, type=elem_type)
        store = memory.StoreOp(mem=load, value=result, ptr=alloca)
        self._mem[name] = store
        return store

    # --- Increment / decrement ---

    @lowering_for(LvaluePreIncrementOp)
    def lower_pre_increment(self, op: LvaluePreIncrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=False, negate=False)

    @lowering_for(LvaluePostIncrementOp)
    def lower_post_increment(self, op: LvaluePostIncrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=True, negate=False)

    @lowering_for(LvaluePreDecrementOp)
    def lower_pre_decrement(self, op: LvaluePreDecrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=False, negate=True)

    @lowering_for(LvaluePostDecrementOp)
    def lower_post_decrement(self, op: LvaluePostDecrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=True, negate=True)

    def _lower_increment(self, op: dgen.Op, *, post: bool, negate: bool) -> dgen.Value:
        alloca = op.lvalue
        name = self._name_for_alloca(alloca)
        mem = self._get_mem(name)
        elem_type = op.type
        load = memory.LoadOp(mem=mem, ptr=alloca, type=elem_type)
        one = ConstantOp(value=1, type=elem_type)
        if negate:
            updated = algebra.SubtractOp(left=load, right=one, type=elem_type)
        else:
            updated = algebra.AddOp(left=load, right=one, type=elem_type)
        store = memory.StoreOp(mem=load, value=updated, ptr=alloca)
        self._mem[name] = store
        return load if post else updated
