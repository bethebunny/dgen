"""Thread a memory effect token through loop carries.

This pass runs AFTER ``CLvalueToMemory`` (which lowers C lvalue ops to
``memory`` dialect buffer loads/stores) and BEFORE ``ControlFlowToGoto``.
By that point every op inside a loop's condition/body is already
buffer-lowered, so a declarative ``@lowering_for(control_flow.WhileOp)``
handler can see the buffer ops and thread the loop-carried memory token.

## Why a loop must thread a token

A loop's iterations are SEQUENTIAL iff its carry threads a dataflow/effect
token — a value produced by iteration *n* and consumed by iteration *n+1*.
dcc's C ``while`` loops mutate shared memory, so they MUST thread such a
token to obtain sequential semantics. A loop with no carried token has
CONCURRENT (reorderable) iterations. See
``dgen/passes/control_flow_to_goto.py`` for the iteration contract.

## Top-down traversal and nesting

The pass framework's ``_lower_block`` dispatches an op's handler BEFORE
recursing into the op's child blocks. So this handler fires TOP-DOWN:
the outer loop's handler runs before the framework descends into the
outer loop's body, hence before any inner loop's handler runs.

To keep nesting correct, the threading for a given loop processes ONLY
THAT LOOP'S OWN LEVEL. When scanning the condition/body for buffer ops,
collecting defined values, or fixing captures, we descend into nested
NON-LOOP blocks (e.g. ``control_flow.IfOp`` bodies) but STOP at nested
``control_flow.WhileOp`` / ``control_flow.ForOp`` bodies — those are
threaded by their own handler invocation when the framework recurses into
them. This mirrors ``_resolve_jump_markers`` in
``dgen/passes/control_flow_to_goto.py``, which recurses into child blocks
but skips nested loop bodies.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import control_flow, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.passes.pass_ import Pass, lowering_for

_LOOP_OPS = (control_flow.WhileOp, control_flow.ForOp)


class ThreadLoopMemory(Pass):
    """Thread a memory effect token through each loop's carry."""

    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._loop_counter = 0

    def run(self, value: dgen.Value, compiler: object) -> dgen.Value:
        self._loop_counter = 0
        return super().run(value, compiler)

    @lowering_for(control_flow.WhileOp)
    def thread(self, w: control_flow.WhileOp) -> dgen.Value | None:
        """Thread a memory effect token through this loop's carry.

        For each of the condition and body blocks, a fresh ``Nil`` block
        argument is added and the ``mem`` operand of every buffer load/store
        in the block (at any non-loop nesting depth) whose ``mem`` is defined
        OUTSIDE the loop block is rewired to read that carried arg. Only the
        ``mem`` field is touched — the ``buf`` pointer is left alone even when
        an alloca serves as both its own buffer and initial mem token.
        ``_fix_captures`` then repairs the captures of every block in the
        subtree (nested non-loop blocks capture the new token; rewired
        external tokens no longer referenced are dropped). The entry token
        folds together all the rewired external ``mem`` values (defined in the
        parent block) into one value and feeds it as ``initial_arguments``.
        The body result becomes a 1-tuple so it lines up with the single
        carried arg.

        Only THIS loop's level is processed: nested loop bodies are treated
        as opaque leaves, threaded by their own handler invocation when the
        framework recurses into them.
        """
        lid = self._loop_counter
        self._loop_counter += 1

        # Collect, by identity, the external ``mem`` values rewired across
        # both blocks — these are the loop's entry dependencies.
        entry_mems: list[dgen.Value] = []

        for blk in (w.condition, w.body):
            tok = BlockArgument(name=f"mem{lid}", type=Nil())
            # Values defined inside this loop block's subtree (its ops and
            # any args/ops of nested NON-LOOP blocks). A buffer-op ``mem``
            # operand not in this set is loop-external and must thread the
            # carry.
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
            # Capture fixup across the whole non-loop subtree: every nested
            # block that still references the carried token (now used as a
            # buffer ``mem``) must capture it, and any rewired external token
            # that is no longer referenced anywhere must be dropped from
            # captures. Allocas (used as ``buf``) stay referenced, so they
            # survive.
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
        return w

    @classmethod
    def _buffer_ops(
        cls,
        block: dgen.Block,
    ) -> list[memory.BufferLoadOp | memory.BufferStoreOp]:
        """Every buffer load/store reachable from ``block``, descending into
        nested NON-LOOP blocks but treating nested loop ops as opaque leaves
        (their buffer ops are threaded by their own handler invocation)."""
        ops: list[memory.BufferLoadOp | memory.BufferStoreOp] = []
        for v in block.values:
            if isinstance(v, (memory.BufferLoadOp, memory.BufferStoreOp)):
                ops.append(v)
            if not isinstance(v, _LOOP_OPS):
                for _, nb in v.blocks:
                    ops.extend(cls._buffer_ops(nb))
        return ops

    @classmethod
    def _defined_values(cls, block: dgen.Block) -> list[dgen.Value]:
        """All values defined within ``block``'s subtree: its own values and
        the args/values of every nested NON-LOOP block. Nested loop ops are
        opaque leaves — their bodies are not descended into."""
        defined: list[dgen.Value] = list(block.args)
        for v in block.values:
            defined.append(v)
            if not isinstance(v, _LOOP_OPS):
                for _, nb in v.blocks:
                    defined.extend(cls._defined_values(nb))
        return defined

    @classmethod
    def _fix_captures(
        cls, block: dgen.Block, tok: BlockArgument, rewired: list[dgen.Value]
    ) -> None:
        """Adjust captures of ``block`` and its nested NON-LOOP blocks after
        the ``mem``-operand rewire.

        For each block in the (non-loop) subtree (excluding ``block`` itself,
        whose ``tok`` is a local arg): ensure ``tok`` is captured iff some op
        in that block (directly) references it, and drop any ``rewired``
        external token no longer referenced by the block subtree. Other
        captures are left untouched — they were already correct.
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

    @classmethod
    def _nested_blocks(cls, block: dgen.Block) -> list[dgen.Block]:
        """Every NON-LOOP block nested within ``block`` (recursively),
        excluding ``block`` itself. Nested loop ops are opaque leaves."""
        nested: list[dgen.Block] = []
        for v in block.values:
            if isinstance(v, _LOOP_OPS):
                continue
            for _, nb in v.blocks:
                nested.append(nb)
                nested.extend(cls._nested_blocks(nb))
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
        """Operands of every op in ``block`` and its nested NON-LOOP blocks."""
        operands = cls._direct_operands(block)
        for nb in cls._nested_blocks(block):
            operands.extend(cls._direct_operands(nb))
        return operands
