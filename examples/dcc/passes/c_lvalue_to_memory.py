"""Lower C lvalue ops to memory dialect ops.

Memory ordering is carried by the use-def graph: each LvalueVarOp.source
points to the prior operation on that variable. After replace_uses_of
runs on earlier ops, sources become BufferStoreOp/BufferLoadOp/PackOp/
control-flow values — any of these serve as mem tokens for the next
operation.

Only BlockArgument and ConstantOp indicate the first operation on a
variable — those fall back to the alloca as the initial mem token.

Reads depend on the latest write. Writes depend on all pending reads
(via a PackOp source). Operations on different variables are independent.

Each variable is a single-cell ``memory.Buffer<T>`` (count=1) — Buffer
is non-linear, so it can be captured into nested control-flow blocks
and used many times across the function body. Reference would not
survive captures (Linear discipline).
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import control_flow, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.ir.traversal import interior_values
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import Constant

from dcc.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp

# After replace_uses_of, a LvalueVarOp.source that was originally an
# AssignOp/LvalueToRvalueOp becomes the BufferStoreOp/BufferLoadOp/PackOp
# that replaced it. Control flow ops (IfOp, WhileOp, RegionOp) also
# appear as ordering fences when variables are read/written inside
# control flow. All of these are valid mem tokens — use them directly.
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
        self._alloca: dict[str, memory.BufferStackAllocateOp] = {}
        self._alloca_owner: dict[str, dgen.Block] = {}
        self._block_stack: list[dgen.Block] = []
        self._loop_counter = 0

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._alloca = {}
        self._alloca_owner = {}
        self._block_stack = []
        self._loop_counter = 0
        return super().run(value, compiler)

    def _lower_block(self, block: dgen.Block) -> None:
        self._block_stack.append(block)
        super()._lower_block(block)
        self._block_stack.pop()
        # After super() returns, every op in this block and its nested
        # blocks is fully buffer-lowered. Thread the memory effect token
        # through any loops that live DIRECTLY in this block, so their
        # cross-iteration memory ordering is explicit (sequential
        # iterations). Inner loops in nested blocks were already threaded
        # when their containing block was processed by super()'s recursion.
        for op in block.ops:
            if isinstance(op, control_flow.WhileOp):
                self._thread_loop_tokens(op)

    def _thread_loop_tokens(self, w: control_flow.WhileOp) -> None:
        """Thread a memory effect token through a loop's carry.

        A loop that threads an effect token has SEQUENTIAL iterations; one
        with no such carry has CONCURRENT (reorderable) iterations. dcc
        loops mutate memory, so they must thread a token.

        For each of the condition and body blocks, a fresh ``Nil`` block
        argument is added and the ``mem`` operand of every buffer load/store
        in the block (at any nesting depth) whose ``mem`` is defined OUTSIDE
        the loop block is rewired to read that carried arg. Only the ``mem``
        field is touched — the ``buf`` pointer is left alone even when an
        alloca serves as both its own buffer and initial mem token.
        ``_fix_captures`` then repairs the captures of every block in the
        subtree (nested blocks capture the new token; rewired external tokens
        no longer referenced are dropped). The entry token folds together all
        the rewired external ``mem`` values (defined in the parent block) into
        one value and feeds it as ``initial_arguments``. The body result
        becomes a 1-tuple so it lines up with the single carried arg.
        """
        lid = self._loop_counter
        self._loop_counter += 1

        # Collect, by identity, the external ``mem`` values rewired across
        # both blocks — these are the loop's entry dependencies.
        entry_mems: list[dgen.Value] = []

        for blk in (w.condition, w.body):
            tok = BlockArgument(name=f"mem{lid}", type=Nil())
            # Values defined inside this loop block's subtree (its ops and
            # any args/ops of nested blocks). A buffer-op ``mem`` operand
            # not in this set is loop-external and must thread the carry.
            defined = self._defined_values(blk)
            rewired: list[dgen.Value] = []
            for o in self._buffer_ops(blk):
                if any(o.mem is d for d in defined):
                    continue
                m = o.mem
                if not any(m is e for e in entry_mems):
                    entry_mems.append(m)
                if not any(m is r for r in rewired):
                    rewired.append(m)
                # Rewire ONLY the ``mem`` field. ``replace_operand`` would
                # also clobber ``buf`` when an alloca serves as both its own
                # buffer and its initial mem token (``buf is mem``).
                o.mem = tok
            blk.args = [*blk.args, tok]
            # Capture fixup across the whole subtree: every nested block that
            # still references the carried token (now used as a buffer ``mem``)
            # must capture it, and any rewired external token that is no longer
            # referenced anywhere must be dropped from captures. Allocas (used
            # as ``buf``) stay referenced, so they survive.
            self._fix_captures(blk, tok, rewired)

        # Entry token: fold all external mem values into one. ChainOp's
        # result depends on both operands, so a left-fold makes one value
        # that transitively depends on every rewired token. Order is
        # irrelevant — we only need a single dependency carrier.
        if entry_mems:
            entry: dgen.Value = entry_mems[0]
            for m in entry_mems[1:]:
                entry = ChainOp(lhs=entry, rhs=m, type=Nil())
        else:
            entry = Nil().constant(None)

        w.body.result = pack([w.body.result])
        w.initial_arguments = pack([entry])

    @staticmethod
    def _buffer_ops(
        block: dgen.Block,
    ) -> list[memory.BufferLoadOp | memory.BufferStoreOp]:
        """Every buffer load/store reachable from ``block``, including those
        inside nested control-flow blocks."""
        ops: list[memory.BufferLoadOp | memory.BufferStoreOp] = []
        for v in block.values:
            if isinstance(v, (memory.BufferLoadOp, memory.BufferStoreOp)):
                ops.append(v)
            for nested in interior_values(v):
                if isinstance(nested, (memory.BufferLoadOp, memory.BufferStoreOp)):
                    ops.append(nested)
        return ops

    @staticmethod
    def _defined_values(block: dgen.Block) -> list[dgen.Value]:
        """All values defined within ``block``'s subtree: its own values and
        the args/values of every nested block."""
        defined: list[dgen.Value] = list(block.args)
        for v in block.values:
            defined.append(v)
            for _, nested_block in v.blocks:
                defined.extend(nested_block.args)
            for nested in interior_values(v):
                defined.append(nested)
                for _, nb in nested.blocks:
                    defined.extend(nb.args)
        return defined

    @classmethod
    def _fix_captures(
        cls, block: dgen.Block, tok: BlockArgument, rewired: list[dgen.Value]
    ) -> None:
        """Adjust captures of ``block`` and its nested blocks after the
        ``mem``-operand rewire.

        For each block in the subtree (excluding ``block`` itself, whose
        ``tok`` is a local arg): ensure ``tok`` is captured iff some op in
        that block (directly) references it, and drop any ``rewired`` external
        token no longer referenced by the block subtree. Other captures are
        left untouched — they were already correct.
        """
        for nb in cls._nested_blocks(block):
            uses_tok = any(o is tok for o in cls._direct_operands(nb))
            if uses_tok and not any(c is tok for c in nb.captures):
                nb.captures = [*nb.captures, tok]
            nb.captures = [
                c
                for c in nb.captures
                if not any(c is r for r in rewired)
                or any(c is u for u in cls._subtree_operands(nb))
            ]
        # The loop block itself: prune rewired tokens it no longer references.
        block.captures = [
            c
            for c in block.captures
            if not any(c is r for r in rewired)
            or any(c is u for u in cls._subtree_operands(block))
        ]

    @staticmethod
    def _nested_blocks(block: dgen.Block) -> list[dgen.Block]:
        """Every block nested within ``block`` (recursively), excluding
        ``block`` itself."""
        nested: list[dgen.Block] = []
        for v in block.values:
            for _, nb in v.blocks:
                nested.append(nb)
                nested.extend(CLvalueToMemory._nested_blocks(nb))
        return nested

    @staticmethod
    def _direct_operands(block: dgen.Block) -> list[dgen.Value]:
        """Operands of ops directly in ``block`` (not nested blocks)."""
        operands: list[dgen.Value] = []
        for v in block.values:
            for _, operand in v.operands:
                operands.append(operand)
        return operands

    @classmethod
    def _subtree_operands(cls, block: dgen.Block) -> list[dgen.Value]:
        """Operands of every op in ``block`` and its nested blocks."""
        operands = cls._direct_operands(block)
        for nb in cls._nested_blocks(block):
            operands.extend(cls._direct_operands(nb))
        return operands

    def _capture_alloca(self, name: str, alloca: memory.BufferStackAllocateOp) -> None:
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
        return memory.BufferStoreOp(
            mem=mem, buf=alloca, index=Index().constant(0), value=op.rvalue
        )

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
        return memory.BufferLoadOp(
            mem=mem, buf=alloca, index=Index().constant(0), type=op.type
        )

    def _ensure_alloca(
        self, name: str, element_type: dgen.Type
    ) -> memory.BufferStackAllocateOp:
        """Get or create a stack-allocated 1-cell buffer for a variable."""
        alloca = self._alloca.get(name)
        if alloca is not None:
            return alloca
        alloca = memory.BufferStackAllocateOp(
            element_type=element_type,
            count=Index().constant(1),
            type=memory.Buffer(element_type=element_type),
        )
        self._alloca[name] = alloca
        # Allocas belong to the outermost block (function body), not
        # wherever they're first encountered. The first block in the
        # stack is the root wrapper; the second is the function body.
        self._alloca_owner[name] = self._block_stack[1]
        return alloca
