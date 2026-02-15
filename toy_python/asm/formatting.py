"""Generic format-driven IR serialization and parsing.

Type aliases and Dialect.op() drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Annotated, Union, get_args, get_origin, get_type_hints

from .asm import indent

# ===----------------------------------------------------------------------=== #
# Annotated field-type aliases
# ===----------------------------------------------------------------------=== #

Ssa = Annotated[str, "ssa"]  # %name
Sym = Annotated[str, "sym"]  # @name
Bare = Annotated[str, "bare"]  # name (as-is)
Shape = Annotated[list[int], "shape"]  # <2x3>
SsaList = Annotated[list[str], "ssa"]  # [%a, %b]
BareList = Annotated[list[str], "bare"]  # [a, b]

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

# Fields with special handling (not serialized as args)
_SPECIAL_FIELDS = {"result", "type", "body"}


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
    if origin is Union:
        args = get_args(hint)
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return None


# ===----------------------------------------------------------------------=== #
# Generic serializer
# ===----------------------------------------------------------------------=== #


def _format_value(value, hint) -> str:
    """Format a single value based on its type hint."""
    base, tag = _get_annotation(hint)

    # Annotated[str, "ssa"] -> %name
    if base is str and tag == "ssa":
        return f"%{value}"
    # Annotated[str, "sym"] -> @name
    if base is str and tag == "sym":
        return f"@{value}"
    # Annotated[str, "bare"] -> name
    if base is str and tag == "bare":
        return value
    # Annotated[list[int], "shape"] -> <2x3>
    if tag == "shape" and get_origin(base) is list:
        return "<" + "x".join(str(d) for d in value) + ">"
    # Annotated[list[str], "ssa"] -> [%a, %b]
    if tag == "ssa" and get_origin(base) is list:
        return "[" + ", ".join(f"%{v}" for v in value) + "]"
    # Annotated[list[str], "bare"] -> [a, b]
    if tag == "bare" and get_origin(base) is list:
        return "[" + ", ".join(value) + "]"
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
    if hasattr(value, "asm") and callable(getattr(type(value), "asm", None)):
        # It's a property
        return value.asm
    if hasattr(value, "asm"):
        return value.asm
    return str(value)


def op_asm(op) -> Iterable[str]:
    """Generic asm emitter. Introspects _asm_name and field types."""
    cls = type(op)
    asm_name = cls._asm_name
    dialect_name = getattr(cls, "_dialect_name", "builtin")
    hints = get_type_hints(cls, include_extras=True)
    fields = dataclasses.fields(cls)

    has_result = "result" in hints
    has_type = "type" in hints
    has_body = "body" in hints

    # Handle optional result (Ssa | None)
    result_val = getattr(op, "result", None) if has_result else None
    show_result = has_result and result_val is not None

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
        arg_parts.append(_format_value(value, effective_hint))

    args_str = ", ".join(arg_parts)

    # Build the line
    parts = []
    if show_result:
        parts.append(f"%{result_val} = ")
    prefix = "" if dialect_name == "builtin" else f"{dialect_name}."
    parts.append(f"{prefix}{asm_name}({args_str})")
    if has_type:
        parts.append(f" : {op.type.asm}")
    if has_body:
        parts.append(":")

    line = "".join(parts)
    yield line

    if has_body:
        for child_op in op.body:
            yield from indent(child_op.asm)


