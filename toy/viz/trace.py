"""Record events during an optimization pass for visualization.

Run ToyOptimize through TracingToyOptimize to capture each "examine op"
and "replace_uses" event, along with the block's ASM state at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import dgen
from dgen.block import Block
from dgen.compiler import Compiler
from dgen.module import Module
from dgen.passes.pass_ import Rewriter
from toy.passes.optimize import ToyOptimize


def _block_asm_lines(block: Block) -> list[str]:
    """Return the current ASM text for each op reachable in the block."""
    lines: list[str] = []
    for op in block.ops:
        for line in op.asm:
            lines.append(line)
    return lines


@dataclass
class ExamineEvent:
    """The pass is about to run handlers for this op."""

    op_id: int
    op_name: str
    op_type: str  # e.g. "toy.transpose"
    asm_lines: list[str]  # block state at the moment of examination


@dataclass
class MatchEvent:
    """A handler fired and called replace_uses(old, new)."""

    old_id: int
    old_name: str
    new_id: int
    new_name: str
    asm_lines_after: list[str]  # block state immediately after replacement


Event = ExamineEvent | MatchEvent


class _TracingRewriter(Rewriter):
    def __init__(self, block: Block, events: list[Event]) -> None:
        super().__init__(block)
        self._events = events

    def replace_uses(self, old: dgen.Value, new: dgen.Value) -> bool:
        result = super().replace_uses(old, new)
        old_name = old.name if old.name is not None else "?"
        new_name = new.name if new.name is not None else "?"
        self._events.append(
            MatchEvent(
                old_id=id(old),
                old_name=old_name,
                new_id=id(new),
                new_name=new_name,
                asm_lines_after=_block_asm_lines(self._block),
            )
        )
        return result


class TracingToyOptimize(ToyOptimize):
    """ToyOptimize subclass that records per-op events for visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []
        self.initial_asm_lines: list[str] = []

    def _run_block(self, block: Block) -> None:
        rewriter = _TracingRewriter(block, self.events)
        for op in list(block.ops):
            op_name = op.name if op.name is not None else "?"
            op_type = f"{op.dialect.name}.{op.asm_name}"
            self.events.append(
                ExamineEvent(
                    op_id=id(op),
                    op_name=op_name,
                    op_type=op_type,
                    asm_lines=_block_asm_lines(block),
                )
            )
            handlers = self._handlers.get(type(op), [])
            handled = False
            for handler in handlers:
                if handler(self, op, rewriter):
                    handled = True
                    break
            if not handled:
                for _, child_block in op.blocks:
                    self._run_block(child_block)

    def run(self, module: Module, compiler: Compiler[object]) -> Module:
        for func in module.functions:
            self.initial_asm_lines = _block_asm_lines(func.body)
            self._run_block(func.body)
        return module
