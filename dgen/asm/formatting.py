"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dgen.dialects.builtin import FunctionOp, Nil, PackOp

from ..op import Op
from ..type import Memory, Type
from ..type import Constant, Value


def indent(it: Iterable[str], prefix: str = "    ") -> Iterable[str]:
    for line in it:
        yield f"{prefix}{line}"


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def format_float(v: float) -> str:
    iv = int(v)
    if float(iv) == v:
        return f"{iv}.0"
    return str(v)


# ===----------------------------------------------------------------------=== #
# SlotTracker
# ===----------------------------------------------------------------------=== #


class SlotTracker:
    """Assigns sequential %0, %1, ... names to unnamed Values."""

    def __init__(self) -> None:
        self._slots: dict[int, str] = {}  # id(value) -> name
        self._counter = 0

    def register(self, ops: Sequence[Op]) -> None:
        """Pre-register all ops in a tracker so slot numbers are stable."""
        for op in ops:
            if _is_sugar_op(op):
                continue
            self.track_name(op)
            for _, block in op.blocks:
                for arg in block.args:
                    self.track_name(arg)
                self.register(block.ops)

    def __getitem__(self, value: Value) -> str:
        return self._slots[id(value)]

    def track_name(self, value: Value) -> str:
        vid = id(value)
        if vid in self._slots:
            return self._slots[vid]
        if value.name is not None:
            name = value.name
            # If it's a numeric name, advance counter past it
            if name.isdigit():
                self._counter = max(self._counter, int(name) + 1)
        else:
            name = str(self._counter)
            self._counter += 1
        self._slots[vid] = name
        return name


# ===----------------------------------------------------------------------=== #
# Expression formatter — dispatches on value type
# ===----------------------------------------------------------------------=== #


def _is_sugar_op(op: Op) -> bool:
    """PackOps are always inlined as [...] sugar, never emitted standalone."""
    return isinstance(op, PackOp)


def format_expr(value: object, tracker: SlotTracker | None = None) -> str:
    """Format a value as an expression string, dispatching on runtime type."""
    if isinstance(value, Nil):
        return "()"
    # Inline list sugar: PackOp → [elem0, elem1, ...]
    if isinstance(value, PackOp):
        return "[" + ", ".join(format_expr(v, tracker) for v in value.values) + "]"
    if isinstance(value, Constant) and not isinstance(value, Op):
        return format_expr(value.__constant__.to_json(), tracker)
    if isinstance(value, Type):
        return type_asm(value, tracker)
    if isinstance(value, Value):
        if tracker is not None:
            return f"%{tracker.track_name(value)}"
        name = value.name if value.name is not None else "?"
        return f"%{name}"
    if isinstance(value, Memory):
        return format_expr(value.to_json(), tracker)
    if isinstance(value, list):
        return "[" + ", ".join(format_expr(v, tracker) for v in value) + "]"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, bytes):
        return f'"{value.decode("utf-8")}"'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _dialect_prefix(dialect_name: str) -> str:
    """Return 'dialect.' prefix, or '' for builtin."""
    return "" if dialect_name == "builtin" else f"{dialect_name}."


def type_asm(type_obj: Type, tracker: SlotTracker | None = None) -> str:
    """Generic type formatter via field introspection."""
    prefix = _dialect_prefix(type_obj.dialect.name)
    name = f"{prefix}{type_obj.asm_name}"
    params = list(type_obj.parameters)
    if not params:
        return name
    args = ", ".join(format_expr(val, tracker) for _, val in params)
    return f"{name}<{args}>"


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


def op_asm(op: Op, tracker: SlotTracker | None = None) -> Iterable[str]:
    """Generic asm emitter driven by field declarations."""
    cls = type(op)
    asm_name = cls.asm_name
    dialect_name = op.dialect.name

    # If no tracker provided, create one and register this op
    if tracker is None:
        tracker = SlotTracker()
        tracker.register([op])

    # Build args from declared fields (constants first, then runtime)
    param_parts = [format_expr(param, tracker) for _, param in op.parameters]
    operand_parts = [format_expr(operand, tracker) for _, operand in op.operands]

    # Build the line
    result_name = tracker.track_name(op)
    parts = [f"%{result_name} : {format_expr(op.type, tracker)} = "]
    prefix = _dialect_prefix(dialect_name)
    if asm_name == "constant" and dialect_name == "builtin":
        parts.append(", ".join(param_parts + operand_parts))
    else:
        op_str = f"{prefix}{asm_name}"
        if param_parts:
            op_str += f"<{', '.join(param_parts)}>"
        op_str += f"({', '.join(operand_parts)})"
        parts.append(op_str)
    blocks = list(op.blocks)
    for _, block in blocks:
        block_args = ", ".join(
            f"%{tracker.track_name(a)}: {format_expr(a.type, tracker)}"
            for a in block.args
        )
        parts.append(f" ({block_args})")
    if blocks:
        parts.append(":")

    line = "".join(parts)
    yield line

    for _, block in blocks:
        for child_op in block.ops:
            if _is_sugar_op(child_op):
                continue
            yield from indent(op_asm(child_op, tracker))


# ===----------------------------------------------------------------------=== #
# Function-specific ASM formatter
# ===----------------------------------------------------------------------=== #


def format_func(func: FunctionOp) -> Iterable[str]:
    """Format a function op with its signature and body.

    Unlike generic op_asm, this shows the function signature style:
      %name = function (%arg0: type, ...) -> result_type:
          body ops...
    """
    tracker = SlotTracker()
    for arg in func.body.args:
        tracker.track_name(arg)
    tracker.register(func.body.ops)

    name = tracker.track_name(func)
    arg_parts = []
    for a in func.body.args:
        n = tracker.track_name(a)
        if a.type is not None:
            arg_parts.append(f"%{n}: {format_expr(a.type, tracker)}")
        else:
            arg_parts.append(f"%{n}")
    from dgen.module import Function

    args = ", ".join(arg_parts)
    assert isinstance(func.type, Function)
    yield f"%{name} = function ({args}) -> {format_expr(func.type.result, tracker)}:"
    for op in func.body.ops:
        if _is_sugar_op(op):
            continue
        yield from indent(op_asm(op, tracker))
