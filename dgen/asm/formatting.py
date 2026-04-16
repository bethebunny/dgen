"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

from collections.abc import Iterable

from dgen.block import Block, BlockArgument, BlockParameter
from dgen.builtins import ConstantOp, PackOp
from dgen.type import constant, format_value

from ..op import Op
from ..type import Value


def indent(it: Iterable[str], prefix: str = "    ") -> Iterable[str]:
    for line in it:
        yield f"{prefix}{line}" if line else ""


# ===----------------------------------------------------------------------=== #
# SlotTracker
# ===----------------------------------------------------------------------=== #


class SlotTracker:
    """Assigns sequential %0, %1, ... names to unnamed Values."""

    def __init__(self) -> None:
        self._slots: dict[Value, str] = {}
        self._used: set[str] = set()
        self._counter = 0

    def register(self, ops: Iterable[Op]) -> None:
        """Pre-register all ops in a tracker so slot numbers are stable."""
        for op in ops:
            if _is_sugar_op(op):
                continue
            if op in self._slots:
                continue
            self.track_name(op)
            for _, block in op.blocks:
                self._register_block(block)

    def _register_block(self, block: Block) -> None:
        self.track_name(block)
        for param in block.parameters:
            self.track_name(param)
        for arg in block.args:
            self.track_name(arg)
        for v in block.values:
            if isinstance(v, Block):
                self._register_block(v)
            elif isinstance(v, Op) and not _is_sugar_op(v):
                self.track_name(v)

    def __getitem__(self, value: Value) -> str:
        return self._slots[value]

    def track_name(self, value: Value) -> str:
        if value in self._slots:
            return self._slots[value]
        if value.name is not None and value.name not in self._used:
            name = value.name
            if name.isdigit():
                self._counter = max(self._counter, int(name) + 1)
        else:
            name = str(self._counter)
            self._counter += 1
        self._used.add(name)
        self._slots[value] = name
        return name


# ===----------------------------------------------------------------------=== #
# Expression formatter
# ===----------------------------------------------------------------------=== #


def _is_sugar_op(op: Op) -> bool:
    """PackOps are always inlined as [...] sugar, never emitted standalone."""
    return isinstance(op, PackOp)


def _format_expr(value: object, tracker: SlotTracker) -> str:
    """Format a value as an expression string.

    Values dispatch to ``Value.format_asm``; plain Python literals (int, float,
    str, list, dict) are formatted as JSON.
    """
    return format_value(value, tracker.track_name)


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


def _format_block_arg(arg: BlockArgument | BlockParameter, tracker: SlotTracker) -> str:
    return f"%{tracker.track_name(arg)}: {_format_expr(arg.type, tracker)}"


def block_asm(
    block: Block,
    tracker: SlotTracker,
    formatted: set[Value],
) -> Iterable[str]:
    """Emit a block as a standalone named value with indented body."""
    if block in formatted:
        return
    formatted.add(block)

    name = tracker.track_name(block)
    type_str = _format_expr(block.type, tracker)

    args_str = ", ".join(_format_block_arg(a, tracker) for a in block.args)
    header = "block"
    if block.parameters:
        params_str = ", ".join(_format_block_arg(p, tracker) for p in block.parameters)
        header += f"<{params_str}>"
    header += f"({args_str})"
    if block.captures:
        caps_str = ", ".join(f"%{tracker.track_name(v)}" for v in block.captures)
        header += f" captures({caps_str})"
    header += ":"

    yield f"%{name} : {type_str} = {header}"

    for v in block.values:
        if isinstance(v, Block):
            yield from indent(block_asm(v, tracker, formatted))
        elif isinstance(v, Op) and not _is_sugar_op(v):
            yield from indent(op_asm(v, tracker, formatted))

    yield ""


def op_asm(
    op: Op,
    tracker: SlotTracker | None = None,
    formatted: set[Value] | None = None,
) -> Iterable[str]:
    """Generic asm emitter driven by field declarations."""
    if formatted is None:
        formatted = set()
    if op in formatted:
        return
    formatted.add(op)

    if tracker is None:
        tracker = SlotTracker()
        tracker.register([op])

    cls = type(op)
    param_parts = [_format_expr(param, tracker) for _, param in op.parameters]
    operand_parts = [_format_expr(operand, tracker) for _, operand in op.operands]

    result_name = tracker.track_name(op)
    type_str = _format_expr(op.type, tracker)
    parts = [f"%{result_name} : {type_str} = "]
    if isinstance(op, ConstantOp):
        parts.append(format_value(constant(op), tracker.track_name))
    else:
        op_str = op.dialect.qualified_name(cls.asm_name)
        if param_parts:
            op_str += f"<{', '.join(param_parts)}>"
        op_str += f"({', '.join(operand_parts)})"
        parts.append(op_str)

    yield "".join(parts)
