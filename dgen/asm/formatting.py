"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects.builtin import Nil
from dgen.module import ConstantOp, PackOp

from ..op import Op
from ..type import Type, Value, format_value


def indent(it: Iterable[str], prefix: str = "    ") -> Iterable[str]:
    for line in it:
        yield f"{prefix}{line}"


# ===----------------------------------------------------------------------=== #
# SlotTracker
# ===----------------------------------------------------------------------=== #


class SlotTracker:
    """Assigns sequential %0, %1, ... names to unnamed Values."""

    def __init__(self) -> None:
        self._slots: dict[Value, str] = {}
        self._used: set[str] = set()
        self._counter = 0
        self.dialects: set[str] = set()

    def register(self, ops: Sequence[Op]) -> None:
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


def _record_dialects(value: object, dialects: set[str]) -> None:
    """Recursively record all non-builtin dialect names referenced by a value."""
    if isinstance(value, Type):
        if value.dialect.name != "builtin":
            dialects.add(value.dialect.name)
        for _, param in value.parameters:
            _record_dialects(param, dialects)
    elif isinstance(value, PackOp):
        for v in value:
            _record_dialects(v, dialects)
    elif isinstance(value, Value):
        # For non-Type Values (Constants, SSA refs), record dialects from
        # their type and recurse into the type's parameters.
        _record_dialects(value.type, dialects)


def format_expr(value: object, tracker: SlotTracker | None = None) -> str:
    """Format a value as an expression string.

    Values dispatch to ``Value.format_asm``; Nil is special-cased to ``()``;
    plain Python literals (int, float, str, list, dict) are formatted as JSON.
    """
    if isinstance(value, Nil):
        return "()"
    if tracker is not None:
        _record_dialects(value, tracker.dialects)
        return format_value(value, tracker.track_name)
    return format_value(value)


def type_asm(type_obj: Type, tracker: SlotTracker | None = None) -> str:
    """Format a Type as ASM text."""
    if tracker is not None:
        _record_dialects(type_obj, tracker.dialects)
        return type_obj.format_asm(tracker.track_name)
    return type_obj.format_asm()


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


def _format_block_arg(arg: BlockArgument | BlockParameter, tracker: SlotTracker) -> str:
    type_str = (
        type_asm(arg.type, tracker)
        if isinstance(arg.type, Type)
        else format_expr(arg.type, tracker)
    )
    return f"%{tracker.track_name(arg)}: {type_str}"


def op_asm(
    op: Op,
    tracker: SlotTracker | None = None,
    _formatted: set[int] | None = None,
) -> Iterable[str]:
    """Generic asm emitter driven by field declarations."""
    if _formatted is None:
        _formatted = set()
    oid = id(op)
    if oid in _formatted:
        return
    _formatted.add(oid)

    cls = type(op)
    asm_name = cls.asm_name
    dialect_name = op.dialect.name

    # Record the op's dialect for import generation.
    if tracker is not None and dialect_name != "builtin":
        tracker.dialects.add(dialect_name)

    # If no tracker provided, create one and register this op
    if tracker is None:
        tracker = SlotTracker()
        tracker.register([op])

    # Build args from declared fields — type-kinded parameters use type_asm
    # (no Nil→() mapping), while value-kinded operands use format_expr.
    param_parts = [
        type_asm(param, tracker)
        if isinstance(param, Type)
        else format_expr(param, tracker)
        for _, param in op.parameters
    ]
    operand_parts = [format_expr(operand, tracker) for _, operand in op.operands]

    # Build the line — type annotation uses type_asm (Nil stays "Nil").
    result_name = tracker.track_name(op)
    type_str = (
        type_asm(op.type, tracker)
        if isinstance(op.type, Type)
        else format_expr(op.type, tracker)
    )
    parts = [f"%{result_name} : {type_str} = "]
    prefix = "" if dialect_name == "builtin" else f"{dialect_name}."
    if isinstance(op, ConstantOp):
        parts.append(format_expr(op.value, tracker))
    else:
        op_str = f"{prefix}{asm_name}"
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
            yield from indent(op_asm(child_op, tracker, _formatted))
