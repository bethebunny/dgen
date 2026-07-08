"""Thread a memory effect token through loop carries.

Runs after ``CLvalueToMemory`` (every op inside a loop is already
buffer-lowered) and before ``ControlFlowToGoto``. C loops mutate shared
memory, so their iterations must be sequential; per the loop iteration
contract (see ``dgen/passes/control_flow_to_goto.py``), that ordering must
be an explicit dataflow edge: a token produced by iteration *n* and
consumed by iteration *n+1*, carried as a loop block argument.

The pass framework dispatches an op's handler before recursing into its
child blocks, so the handler fires top-down: each invocation threads only
its own loop's level, treating nested loop bodies as opaque leaves to be
threaded by their own invocation (mirroring ``_resolve_jump_markers``).
Every helper below observes the same boundary — it descends into nested
non-loop blocks (e.g. ``if`` bodies) but stops at nested loops.
"""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import control_flow, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.ir.traversal import all_values
from dgen.passes.pass_ import Pass, lowering_for

_LOOP_OPS = (control_flow.WhileOp, control_flow.ForOp)
_BUFFER_OPS = (memory.BufferLoadOp, memory.BufferStoreOp)


def _buffer_ops(
    block: dgen.Block,
) -> Iterator[memory.BufferLoadOp | memory.BufferStoreOp]:
    """Buffer loads/stores in ``block``'s non-loop subtree."""
    for value in block.values:
        if isinstance(value, _BUFFER_OPS):
            yield value
        if not isinstance(value, _LOOP_OPS):
            for _, nested in value.blocks:
                yield from _buffer_ops(nested)


def _defined_values(block: dgen.Block) -> Iterator[dgen.Value]:
    """Values defined in ``block``'s non-loop subtree: its args and ops,
    and those of every nested non-loop block."""
    yield from block.args
    for value in block.values:
        yield value
        if not isinstance(value, _LOOP_OPS):
            for _, nested in value.blocks:
                yield from _defined_values(nested)


def _nested_blocks(block: dgen.Block) -> Iterator[dgen.Block]:
    """Non-loop blocks nested within ``block``, recursively."""
    for value in block.values:
        if isinstance(value, _LOOP_OPS):
            continue
        for _, nested in value.blocks:
            yield nested
            yield from _nested_blocks(nested)


def _direct_operands(block: dgen.Block) -> set[dgen.Value]:
    """Operands of ops directly in ``block`` (not nested blocks)."""
    return {operand for value in block.values for _, operand in value.operands}


def _subtree_operands(block: dgen.Block) -> set[dgen.Value]:
    """Operands of every op in ``block``'s non-loop subtree."""
    return _direct_operands(block).union(
        *(_direct_operands(nested) for nested in _nested_blocks(block))
    )


def _fix_captures(
    block: dgen.Block, token: BlockArgument, rewired: set[dgen.Value]
) -> None:
    """Repair captures after the ``mem`` rewire: nested non-loop blocks
    that now reference ``token`` capture it, and rewired external mems
    no longer referenced anywhere are dropped. Allocas (still used as
    ``buf``) stay referenced, so they survive the drop."""
    for nested in _nested_blocks(block):
        if token in _direct_operands(nested) and token not in nested.captures:
            nested.captures = [*nested.captures, token]
        nested.captures = [
            capture
            for capture in nested.captures
            if capture not in rewired or capture in _subtree_operands(nested)
        ]
    block.captures = [
        capture
        for capture in block.captures
        if capture not in rewired or capture in _subtree_operands(block)
    ]


class ThreadLoopMemory(Pass):
    """Thread a memory effect token through each loop's carry."""

    allow_unregistered_ops = True

    def verify_postconditions(self, value: dgen.Value) -> None:
        """Every buffer op inside a loop reads a loop-internal mem token.

        This is what makes the iteration contract sound for dcc: a buffer op
        whose ``mem`` reaches outside its loop has no cross-iteration edge,
        so the loop would wrongly read as concurrent."""
        super().verify_postconditions(value)
        for op in all_values(value):
            if not isinstance(op, control_flow.WhileOp):
                continue
            for block in (op.condition, op.body):
                defined = set(_defined_values(block))
                for buffer_op in _buffer_ops(block):
                    if buffer_op.mem not in defined:
                        raise ValueError(
                            f"buffer op {buffer_op.name!r} in a loop reads "
                            f"loop-external mem {buffer_op.mem.name!r}; "
                            "its ordering must thread the loop carry"
                        )

    @lowering_for(control_flow.WhileOp)
    def thread(self, op: control_flow.WhileOp) -> dgen.Value | None:
        """Thread a fresh ``Nil`` token through this loop's carry.

        The condition and body each gain a token block argument; every
        buffer op in the block whose ``mem`` is loop-external is rewired to
        read it. The rewired external mems fold into a single entry token
        fed via ``initial_arguments``, and the body result becomes a 1-tuple
        so the final token feeds back through the carry. The ``Nil`` token
        has no runtime representation — codegen erases its phi.
        """
        entry_mems: list[dgen.Value] = []
        seen_entry_mems: set[dgen.Value] = set()

        for block in (op.condition, op.body):
            token = BlockArgument(name="mem", type=Nil())
            defined = set(_defined_values(block))
            rewired: set[dgen.Value] = set()
            # Materialize before rewiring: the walks behind these helpers
            # follow the very ``mem`` operand edges the loop mutates.
            for buffer_op in list(_buffer_ops(block)):
                if buffer_op.mem in defined:
                    continue
                mem = buffer_op.mem
                if mem not in seen_entry_mems:
                    seen_entry_mems.add(mem)
                    entry_mems.append(mem)
                rewired.add(mem)
                # Assign only the ``mem`` field. ``replace_operand`` would
                # also clobber ``buf`` when an alloca serves as both its own
                # buffer and its initial mem token (``buf is mem``).
                buffer_op.mem = token
            block.args = [*block.args, token]
            _fix_captures(block, token, rewired)

        # ChainOp's result depends on both operands, so a left-fold yields
        # one value that transitively depends on every rewired mem.
        if entry_mems:
            entry: dgen.Value = entry_mems[0]
            for mem in entry_mems[1:]:
                entry = ChainOp(result=entry, effect=mem, type=Nil())
        else:
            entry = Nil().constant(None)

        op.body.result = pack([op.body.result])
        op.initial_arguments = pack([entry])
        return op
