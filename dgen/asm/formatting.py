"""Generic format-driven IR serialization and parsing.

Dialect.op() and Dialect.type() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Iterable
from typing import Union, get_args, get_origin, get_type_hints

from .asm import indent

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

# Fields with special handling (not serialized as args)
_SPECIAL_FIELDS = {"name", "type", "body"}


def format_float(v: float) -> str:
    iv = int(v)
    if float(iv) == v:
        return f"{iv}.0"
    return str(v)


def _is_optional(hint):
    """Check if hint is X | None, return X if so, else None."""
    origin = get_origin(hint)
    if origin is Union or isinstance(hint, types.UnionType):
        args = get_args(hint)
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return None


# ===----------------------------------------------------------------------=== #
# SlotTracker
# ===----------------------------------------------------------------------=== #


class SlotTracker:
    """Assigns sequential %0, %1, ... names to unnamed Values."""

    def __init__(self):
        self._slots: dict[int, str] = {}  # id(value) -> name
        self._counter = 0

    def get_name(self, value) -> str:
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


def format_expr(value, tracker: SlotTracker | None = None) -> str:
    """Format a value as an expression string, dispatching on runtime type."""
    from dgen.dialects.builtin import Nil, Value

    if isinstance(value, Nil):
        return "()"
    if isinstance(value, Value):
        if tracker is not None:
            return f"%{tracker.get_name(value)}"
        name = value.name if value.name is not None else "?"
        return f"%{name}"
    if isinstance(value, list):
        return "[" + ", ".join(format_expr(v, tracker) for v in value) + "]"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if hasattr(value, "_asm_name"):
        return type_asm(value)
    # Fallback for types with hand-written .asm (e.g. llvm types)
    if hasattr(value, "asm") and isinstance(value.asm, str):
        return value.asm
    return str(value)


def type_asm(type_obj) -> str:
    """Generic type formatter via field introspection."""
    cls = type(type_obj)
    prefix = f"{cls.dialect.name}." if cls.dialect.name != "builtin" else ""
    name = f"{prefix}{cls._asm_name}"
    if dataclasses.is_dataclass(cls):
        fields = dataclasses.fields(cls)
    else:
        fields = ()
    if not fields:
        return name
    args = ", ".join(format_expr(getattr(type_obj, f.name)) for f in fields)
    return f"{name}({args})"


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


@functools.cache
def _class_hints(cls: type) -> dict[str, type]:
    return get_type_hints(cls, include_extras=True)


def op_asm(op, tracker: SlotTracker | None = None) -> Iterable[str]:
    """Generic asm emitter. Introspects _asm_name and field types."""
    from dgen.dialects.builtin import _register_ops

    cls = type(op)
    asm_name = cls._asm_name
    dialect_name = op.dialect.name
    hints = _class_hints(cls)
    fields = dataclasses.fields(cls)

    # If no tracker provided, create one and register this op
    if tracker is None:
        tracker = SlotTracker()
        tracker.get_name(op)
        _register_ops(tracker, [op])

    has_body = "body" in hints

    # Build args
    arg_parts = []
    for f in fields:
        if f.name in _SPECIAL_FIELDS:
            continue
        hint = hints[f.name]
        value = getattr(op, f.name)
        # Optional field at end: skip if None
        inner = _is_optional(hint)
        if inner is not None and value is None:
            continue
        arg_parts.append(format_expr(value, tracker))

    args_str = ", ".join(arg_parts)

    # Build the line
    result_name = tracker.get_name(op)
    parts = [f"%{result_name} : {format_expr(op.type)} = "]
    prefix = "" if dialect_name == "builtin" else f"{dialect_name}."
    if asm_name == "constant" and dialect_name == "builtin":
        parts.append(args_str)
    else:
        parts.append(f"{prefix}{asm_name}({args_str})")
    if has_body:
        from dgen.dialects.builtin import Block

        if isinstance(op.body, Block) and op.body.args:
            block_args = ", ".join(
                f"%{tracker.get_name(a)}: {format_expr(a.type)}"
                for a in op.body.args
            )
            parts.append(f" ({block_args})")
        parts.append(":")

    line = "".join(parts)
    yield line

    if has_body:
        from dgen.dialects.builtin import Block

        body_ops = op.body.ops if isinstance(op.body, Block) else op.body
        for child_op in body_ops:
            yield from indent(op_asm(child_op, tracker))
