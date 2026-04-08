"""Lower C variable and struct ops to memory dialect ops.

This pass owns all memory state: variable allocations, mem-token
threading, and struct field access lowering.

variable_declaration → stack_allocate + store
read_variable        → load (with per-variable mem token)
assign               → store (updates per-variable mem token)
pre/post_increment   → load + add + store
pre/post_decrement   → load + subtract + store
member_access        → GEP + load
pointer_member_access → GEP + load
"""

from __future__ import annotations

import dgen
from dgen.dialects import algebra, llvm, memory
from dgen.dialects.builtin import Nil
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for

from dcc.dialects.c import (
    AddressOfOp,
    AssignOp,
    CompoundAssignOp,
    DereferenceOp,
    ElementAddressOp,
    FieldAddressOp,
    MemberAccessOp,
    PointerMemberAccessOp,
    PostDecrementOp,
    PostIncrementOp,
    PreDecrementOp,
    PreIncrementOp,
    ReadVariableOp,
    StoreIndirectOp,
    SubscriptOp,
    VariableDeclarationOp,
)


from dcc.dialects.c import ModuloOp, ShiftLeftOp, ShiftRightOp

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


def _variable_name(op: dgen.Op) -> str:
    """Extract the variable_name parameter from a C variable op."""
    return op.variable_name.__constant__.to_json()


class CToMemory(Pass):
    """Lower C variable/struct ops to memory dialect ops."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        # Per-variable state: alloca and last memory token
        self._alloca: dict[str, dgen.Value] = {}
        self._mem: dict[str, dgen.Value] = {}

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        """Run on a value, resetting per-variable state."""
        self._alloca = {}
        self._mem = {}
        return super().run(value, compiler)

    def _get_mem(self, name: str) -> dgen.Value:
        """Get the current memory token for a variable."""
        return self._mem.get(name, self._alloca.get(name, Nil().constant(None)))

    # --- Variable declaration ---

    @lowering_for(VariableDeclarationOp)
    def lower_variable_declaration(
        self, op: VariableDeclarationOp
    ) -> dgen.Value | None:
        name = _variable_name(op)
        var_type = op.variable_type
        if not isinstance(var_type, dgen.Type):
            var_type = dgen.type.type_constant(var_type)
        alloca = memory.StackAllocateOp(
            element_type=var_type,
            type=memory.Reference(element_type=var_type),
        )
        self._alloca[name] = alloca
        store = memory.StoreOp(mem=alloca, value=op.initializer, ptr=alloca)
        self._mem[name] = store
        return store

    # --- Read variable ---

    @lowering_for(ReadVariableOp)
    def lower_read_variable(self, op: ReadVariableOp) -> dgen.Value | None:
        name = _variable_name(op)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.source
        mem = self._get_mem(name)
        return memory.LoadOp(mem=mem, ptr=alloca, type=op.type)

    # --- Assign ---

    @lowering_for(AssignOp)
    def lower_assign(self, op: AssignOp) -> dgen.Value | None:
        name = _variable_name(op)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.value
        mem = self._get_mem(name)
        store = memory.StoreOp(mem=mem, value=op.value, ptr=alloca)
        self._mem[name] = store
        return store

    # --- Compound assign ---

    @lowering_for(CompoundAssignOp)
    def lower_compound_assign(self, op: CompoundAssignOp) -> dgen.Value | None:
        from dgen.module import string_value

        name = _variable_name(op)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.operand
        mem = self._get_mem(name)
        elem_type = op.type
        load = memory.LoadOp(mem=mem, ptr=alloca, type=elem_type)
        operator = string_value(op.operator)
        binop_cls = _BINOP_TABLE.get(operator)
        if binop_cls is None:
            return op.operand
        if "left" in binop_cls.__dataclass_fields__:
            result = binop_cls(left=load, right=op.operand, type=elem_type)
        else:
            result = binop_cls(lhs=load, rhs=op.operand, type=elem_type)
        store = memory.StoreOp(mem=load, value=result, ptr=alloca)
        self._mem[name] = store
        return store

    # --- Pre/post increment ---

    @lowering_for(PreIncrementOp)
    def lower_pre_increment(self, op: PreIncrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=False, negate=False)

    @lowering_for(PostIncrementOp)
    def lower_post_increment(self, op: PostIncrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=True, negate=False)

    @lowering_for(PreDecrementOp)
    def lower_pre_decrement(self, op: PreDecrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=False, negate=True)

    @lowering_for(PostDecrementOp)
    def lower_post_decrement(self, op: PostDecrementOp) -> dgen.Value | None:
        return self._lower_increment(op, post=True, negate=True)

    def _lower_increment(self, op: dgen.Op, *, post: bool, negate: bool) -> dgen.Value:
        name = _variable_name(op)
        alloca = self._alloca.get(name)
        if alloca is None:
            return op.target
        mem = self._get_mem(name)
        elem_type = op.type
        load = memory.LoadOp(mem=mem, ptr=alloca, type=elem_type)
        if negate:
            updated = algebra.SubtractOp(
                left=load, right=elem_type.constant(1), type=elem_type
            )
        else:
            updated = algebra.AddOp(
                left=load, right=elem_type.constant(1), type=elem_type
            )
        store = memory.StoreOp(mem=load, value=updated, ptr=alloca)
        self._mem[name] = store
        return load if post else updated

    # --- Struct member access ---

    @lowering_for(MemberAccessOp)
    def lower_member_access(self, op: MemberAccessOp) -> dgen.Value | None:
        return op.base  # simplified: struct values pass through

    @lowering_for(PointerMemberAccessOp)
    def lower_pointer_member_access(
        self, op: PointerMemberAccessOp
    ) -> dgen.Value | None:
        gep = llvm.GepOp(
            base=op.base, index=llvm.Int(bits=Index().constant(64)).constant(0)
        )
        return memory.LoadOp(mem=gep, ptr=gep, type=op.type)

    # --- Pointer/array ops ---

    @lowering_for(DereferenceOp)
    def lower_dereference(self, op: DereferenceOp) -> dgen.Value | None:
        return memory.LoadOp(mem=op.pointer, ptr=op.pointer, type=op.type)

    @lowering_for(AddressOfOp)
    def lower_address_of(self, op: AddressOfOp) -> dgen.Value | None:
        return op.operand  # the operand is already a pointer/reference

    @lowering_for(SubscriptOp)
    def lower_subscript(self, op: SubscriptOp) -> dgen.Value | None:
        ref_type = memory.Reference(element_type=op.type)
        offset = memory.OffsetOp(ptr=op.base, index=op.index, type=ref_type)
        return memory.LoadOp(mem=offset, ptr=offset, type=op.type)

    @lowering_for(ElementAddressOp)
    def lower_element_address(self, op: ElementAddressOp) -> dgen.Value | None:
        return memory.OffsetOp(ptr=op.base, index=op.index, type=op.type)

    @lowering_for(FieldAddressOp)
    def lower_field_address(self, op: FieldAddressOp) -> dgen.Value | None:
        # TODO: use real struct field offsets. PointerMemberAccessOp has the
        # same placeholder — GEP with index 0 just returns the base pointer.
        return llvm.GepOp(
            base=op.base, index=llvm.Int(bits=Index().constant(64)).constant(0)
        )

    @lowering_for(StoreIndirectOp)
    def lower_store_indirect(self, op: StoreIndirectOp) -> dgen.Value | None:
        return memory.StoreOp(mem=op.target, value=op.value, ptr=op.target)
