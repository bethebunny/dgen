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
"""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.block import BlockArgument
from dgen.builtins import pack
from dgen.dialects import control_flow, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.ir.traversal import all_values
from dgen.ir.uses import rebind_all, uses_of
from dgen.passes.pass_ import Pass, lowering_for

_LOOP_OPS = (control_flow.WhileOp, control_flow.ForOp)
_BUFFER_OPS = (memory.BufferLoadOp, memory.BufferStoreOp)


def _own_level(op: dgen.Value) -> bool:
    """Descend into non-loop child blocks only — nested loops are threaded
    by their own handler invocation."""
    return not isinstance(op, _LOOP_OPS)


def _buffer_ops(
    block: dgen.Block,
) -> Iterator[memory.BufferLoadOp | memory.BufferStoreOp]:
    """Buffer loads/stores in ``block``'s non-loop subtree."""
    for value in block.values:
        if isinstance(value, _BUFFER_OPS):
            yield value
        if _own_level(value):
            for _, nested in value.blocks:
                yield from _buffer_ops(nested)


def _defined_values(block: dgen.Block) -> Iterator[dgen.Value]:
    """Values defined in ``block``'s non-loop subtree: its args and ops,
    and those of every nested non-loop block."""
    yield from block.args
    for value in block.values:
        yield value
        if _own_level(value):
            for _, nested in value.blocks:
                yield from _defined_values(nested)


class ThreadLoopMemory(Pass):
    """Thread a memory effect token through each loop's carry."""

    allow_unregistered_ops = True

    def verify_postconditions(self, value: dgen.Value) -> None:
        """Every buffer op inside a loop reads a loop-internal mem token.

        This is what makes the iteration contract sound for dcc: a buffer op
        whose ``mem`` reaches outside its loop has no cross-iteration edge,
        so the loop would wrongly read as concurrent. Checked with an
        exhaustive walk, independent of the capture-guided queries the
        threading itself relies on."""
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

        The condition and body each gain a token block argument, and every
        ``mem`` edge on a buffer op that references a loop-external value —
        by closed blocks, exactly the block's captures — is rebound to it.
        The rewired externals fold into a single entry token fed via
        ``initial_arguments``, and the body result becomes a 1-tuple so the
        final token feeds back through the carry. The ``Nil`` token has no
        runtime representation — codegen erases its phi.
        """
        entry_mems: list[dgen.Value] = []

        for block in (op.condition, op.body):
            token = BlockArgument(name="mem", type=Nil())
            block.args = [*block.args, token]
            for external in list(block.captures):
                mem_edges = (
                    use
                    for use in uses_of(external, within=block, into=_own_level)
                    if use.field == "mem" and isinstance(use.owner, _BUFFER_OPS)
                )
                if rebind_all(mem_edges, token) and not any(
                    mem is external for mem in entry_mems
                ):
                    entry_mems.append(external)

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
