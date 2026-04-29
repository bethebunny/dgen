"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

from collections.abc import Iterable

from dgen.block import Block, BlockArgument, BlockParameter
from dgen.builtins import ConstantOp, PackOp, _aggregate_elements
from dgen.dialects.builtin import Array, Tuple
from dgen.type import Constant, SlotFn, _default_slot, constant, format_value

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
                for param in block.parameters:
                    self.track_name(param)
                for arg in block.args:
                    self.track_name(arg)
                self.register(block.ops)

    def __getitem__(self, value: Value) -> str:
        return self._slots[value]

    def track_name(self, value: Value) -> str:
        if value in self._slots:
            return self._slots[value]
        if value.name is not None and value.name not in self._used:
            name = value.name
            # If it's a numeric name, advance counter past it
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
# Aggregate-Constant formatting
# ===----------------------------------------------------------------------=== #


def _format_aggregate_constant(c: Constant, slot: SlotFn) -> str:
    """Inline format ``[elem0, elem1, ...]`` for aggregate ``Constant``s.

    Each element is reconstructed as a ``Value`` of its field type
    (``_aggregate_elements``) and formatted by its own ``format_asm`` —
    a Type instance prints as its name, a scalar Constant prints as
    ``Type(value)``, and a nested aggregate recurses through this same
    path. Without this, ``Constant.format_asm``'s default ``Type(value)``
    shape would emit verbose ``Tuple<...>(dict)`` for compile-time
    aggregates produced by ``pack()``.
    """
    return "[" + ", ".join(format_value(e, slot) for e in _aggregate_elements(c)) + "]"


def _constant_format_asm(self: Constant, slot: SlotFn = _default_slot) -> str:
    if isinstance(self.type, (Array, Tuple)):
        return _format_aggregate_constant(self, slot)
    body = format_value(constant(self), slot)
    return f"{self.type.format_asm(slot)}({body})"


# Override ``Constant.format_asm`` here (rather than at class definition)
# so the asm-specific aggregate-shape rendering lives next to the rest
# of the formatter, not inside the type-system core.
Constant.format_asm = _constant_format_asm


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


def _format_block_arg(arg: BlockArgument | BlockParameter, tracker: SlotTracker) -> str:
    return f"%{tracker.track_name(arg)}: {_format_expr(arg.type, tracker)}"


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

    # If no tracker provided, create one and register this op
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

    def _format_block_header(name: str, block: Block) -> str:
        args_str = ", ".join(_format_block_arg(a, tracker) for a in block.args)
        header = name
        if block.parameters:
            params_str = ", ".join(
                _format_block_arg(p, tracker) for p in block.parameters
            )
            header += f"<{params_str}>"
        header += f"({args_str})"
        if block.captures:
            caps_str = ", ".join(f"%{tracker.track_name(v)}" for v in block.captures)
            header += f" captures({caps_str})"
        return header + ":"

    blocks = list(op.blocks)
    if blocks:
        first_block_name, first_block = blocks[0]
        parts.append(f" {_format_block_header(first_block_name, first_block)}")
    else:
        parts.append("")

    line = "".join(parts)
    yield line

    for block_idx, (block_name, block) in enumerate(blocks):
        if block_idx > 0:
            yield _format_block_header(block_name, block)
        for child_op in block.ops:
            if _is_sugar_op(child_op):
                continue
            yield from indent(op_asm(child_op, tracker, formatted))

    if blocks:
        yield ""
