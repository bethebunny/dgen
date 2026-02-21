"""Generic format-driven IR serialization and parsing.

Type aliases and Dialect.op() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Iterable
from typing import Annotated, Union, get_args, get_origin, get_type_hints

from .asm import indent

# ===----------------------------------------------------------------------=== #
# Annotated field-type aliases
# ===----------------------------------------------------------------------=== #

Sym = Annotated[str, "sym"]  # @name
Shape = Annotated[list[int], "shape"]  # <2x3>

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


def _get_annotation(hint):
    """Return (base_type, tag) for Annotated types, else (hint, None)."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return args[0], args[1]
    return hint, None


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
# Generic serializer
# ===----------------------------------------------------------------------=== #


def _format_value(value, hint, tracker: SlotTracker | None = None) -> str:
    """Format a single value based on its type hint."""
    from dgen.dialects.builtin import Value

    # Value reference — use tracker to get %name
    if isinstance(value, Value):
        if tracker is not None:
            return f"%{tracker.get_name(value)}"
        name = value.name if value.name is not None else "?"
        return f"%{name}"

    # list[Value] — format as [%a, %b]
    if isinstance(value, list) and value and isinstance(value[0], Value):
        if tracker is not None:
            return "[" + ", ".join(f"%{tracker.get_name(v)}" for v in value) + "]"
        return "[" + ", ".join(f"%{v.name or '?'}" for v in value) + "]"

    from dgen.dialects.builtin import StaticString

    # StaticString -> quoted string
    if hint is StaticString:
        return f'"{value}"'
    # list[StaticString] -> ["a", "b"]
    if (
        get_origin(hint) is list
        and get_args(hint)
        and get_args(hint)[0] is StaticString
    ):
        return "[" + ", ".join(f'"{v}"' for v in value) + "]"

    base, tag = _get_annotation(hint)

    # Annotated[str, "sym"] -> @name
    if base is str and tag == "sym":
        return f"@{value}"
    # Annotated[list[int], "shape"] -> <2x3>
    if tag == "shape" and get_origin(base) is list:
        return "<" + "x".join(str(d) for d in value) + ">"
    # Union types (float | int | list[float], etc.)
    if isinstance(hint, types.UnionType):
        if isinstance(value, list):
            for arm in get_args(hint):
                if get_origin(arm) is list:
                    return _format_value(value, arm, tracker)
        if isinstance(value, float):
            return format_float(value)
        if isinstance(value, int):
            return str(value)
    # Plain int
    if hint is int:
        return str(value)
    # Plain float
    if hint is float:
        return format_float(value)
    # list[float]
    if get_origin(hint) is list and get_args(hint) == (float,):
        return "[" + ", ".join(format_float(v) for v in value) + "]"
    # Type protocol (has .asm)
    if hasattr(value, "asm"):
        return value.asm
    return str(value)


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
        effective_hint = inner if inner is not None else hint
        arg_parts.append(_format_value(value, effective_hint, tracker))

    args_str = ", ".join(arg_parts)

    # Build the line
    result_name = tracker.get_name(op)
    parts = [f"%{result_name} : {op.type.asm} = "]
    prefix = "" if dialect_name == "builtin" else f"{dialect_name}."
    parts.append(f"{prefix}{asm_name}({args_str})")
    if has_body:
        from dgen.dialects.builtin import Block

        if isinstance(op.body, Block) and op.body.args:
            block_args = ", ".join(
                f"%{tracker.get_name(a)}: {a.type.asm}" for a in op.body.args
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
