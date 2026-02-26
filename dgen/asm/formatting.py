"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable
from typing import get_type_hints

from .. import Block
from ..op import Op
from ..type import Memory
from ..value import Constant, Value


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

    def get_name(self, value: Value) -> str:
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


def format_expr(value: object, tracker: SlotTracker | None = None) -> str:
    """Format a value as an expression string, dispatching on runtime type."""
    from dgen.dialects.builtin import Nil, Value

    if isinstance(value, Nil):
        return "()"
    if isinstance(value, Constant) and not isinstance(value, Op):
        from ..layout import Array

        mem = value.__constant__
        vals = mem.unpack()
        if isinstance(mem.layout, Array):
            return "[" + ", ".join(format_expr(v, tracker) for v in vals) + "]"
        return format_expr(vals[0], tracker)
    if isinstance(value, Value):
        if tracker is not None:
            return f"%{tracker.get_name(value)}"
        name = value.name if value.name is not None else "?"
        return f"%{name}"
    if isinstance(value, Memory):
        from ..layout import Array

        vals = value.unpack()
        if isinstance(value.layout, Array):
            return "[" + ", ".join(format_expr(v, tracker) for v in vals) + "]"
        return format_expr(vals[0], tracker)
    if isinstance(value, list):
        return "[" + ", ".join(format_expr(v, tracker) for v in value) + "]"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if hasattr(value, "_asm_name"):
        return type_asm(value, tracker)
    # Fallback for types with hand-written .asm (e.g. llvm types)
    if hasattr(value, "asm") and isinstance(value.asm, str):
        return value.asm
    return str(value)


def _dialect_prefix(dialect_name: str) -> str:
    """Return 'dialect.' prefix, or '' for builtin."""
    return "" if dialect_name == "builtin" else f"{dialect_name}."


def type_asm(type_obj: object, tracker: SlotTracker | None = None) -> str:
    """Generic type formatter via field introspection."""
    cls = type(type_obj)
    dialect = getattr(cls, "dialect", None)
    prefix = _dialect_prefix(dialect.name if dialect is not None else "builtin")
    name = f"{prefix}{getattr(cls, '_asm_name', '')}"
    if dataclasses.is_dataclass(cls):
        fields = dataclasses.fields(cls)
    else:
        fields = ()
    if not fields:
        return name
    args = ", ".join(format_expr(getattr(type_obj, f.name), tracker) for f in fields)
    return f"{name}({args})"


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


@functools.cache
def _class_hints(cls: type) -> dict[str, type]:
    return get_type_hints(cls, include_extras=True)


def op_asm(op: Op, tracker: SlotTracker | None = None) -> Iterable[str]:
    """Generic asm emitter driven by field declarations."""
    from dgen.dialects.builtin import _register_ops

    cls = type(op)
    asm_name = cls._asm_name
    dialect_name = op.dialect.name

    # If no tracker provided, create one and register this op
    if tracker is None:
        tracker = SlotTracker()
        tracker.get_name(op)
        _register_ops(tracker, [op])

    # Build args from declared fields (constants first, then runtime)
    arg_parts = []
    for f_name, _ in cls.__params__:
        arg_parts.append(format_expr(getattr(op, f_name), tracker))
    for f_name in cls.__operands__:
        arg_parts.append(format_expr(getattr(op, f_name), tracker))

    args_str = ", ".join(arg_parts)

    # Build the line
    result_name = tracker.get_name(op)
    parts = [f"%{result_name} : {format_expr(op.type, tracker)} = "]
    prefix = _dialect_prefix(dialect_name)
    if asm_name == "constant" and dialect_name == "builtin":
        parts.append(args_str)
    else:
        parts.append(f"{prefix}{asm_name}({args_str})")
    if cls.__has_body__:
        body: Block = getattr(op, "body")
        if body.args:
            block_args = ", ".join(
                f"%{tracker.get_name(a)}: {format_expr(a.type)}" for a in body.args
            )
            parts.append(f" ({block_args})")
        parts.append(":")

    line = "".join(parts)
    yield line

    if cls.__has_body__:
        body = getattr(op, "body")
        for child_op in body.ops:
            yield from indent(op_asm(child_op, tracker))
