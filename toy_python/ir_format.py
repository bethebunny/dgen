"""Generic format-driven IR serialization and parsing.

Type aliases and the @op decorator drive both asm emission and parsing
from dataclass field declarations alone — no per-op asm/parse code needed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Annotated, Union, get_args, get_origin, get_type_hints

from toy_python import asm
from toy_python.dialects.builtin import Op, Type

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
    parts.append(f"{asm_name}({args_str})")
    if has_type:
        parts.append(f" : {op.type.asm}")
    if has_body:
        parts.append(":")

    line = "".join(parts)
    yield line

    if has_body:
        for child_op in op.body:
            yield from asm.indent(child_op.asm)


# ===----------------------------------------------------------------------=== #
# Generic parser
# ===----------------------------------------------------------------------=== #


def parse_op_fields(parser, cls, result=None):
    """Generic op field parser. Introspects field types to parse args."""
    hints = get_type_hints(cls, include_extras=True)
    fields = dataclasses.fields(cls)

    kwargs = {}
    if "result" in hints:
        # result=None for keyword ops; for optional-result ops (Ssa | None)
        # this correctly sets None; for required-result ops the caller provides it
        kwargs["result"] = result

    # Expect opening paren
    parser.expect("(")
    parser.skip_whitespace()

    # Parse non-special fields
    arg_fields = [f for f in fields if f.name not in _SPECIAL_FIELDS]
    for i, f in enumerate(arg_fields):
        hint = hints[f.name]
        inner = _is_optional(hint)

        # Check for optional field (at end of args) — if we see ')' it's None
        if inner is not None:
            parser.skip_whitespace()
            if parser.peek() == ")":
                kwargs[f.name] = None
                continue
            hint_to_parse = inner
        else:
            hint_to_parse = hint

        if i > 0:
            parser.expect(",")
            parser.skip_whitespace()

        kwargs[f.name] = _parse_value(parser, hint_to_parse)
        parser.skip_whitespace()

    parser.expect(")")

    # Type annotation
    if "type" in hints:
        parser.skip_whitespace()
        parser.expect(":")
        parser.skip_whitespace()
        kwargs["type"] = parser.parse_type()

    # Body (indented block)
    if "body" in hints:
        parser.skip_whitespace()
        parser.expect(":")
        kwargs["body"] = parser.parse_indented_block()

    return cls(**kwargs)


def _parse_value(parser, hint):
    """Parse a single value based on its type hint."""
    base, tag = _get_annotation(hint)

    # Annotated[str, "ssa"] -> %name
    if base is str and tag == "ssa":
        return parser.parse_ssa_name()
    # Annotated[str, "sym"] -> @name
    if base is str and tag == "sym":
        parser.expect("@")
        return parser.parse_identifier()
    # Annotated[str, "bare"] -> identifier
    if base is str and tag == "bare":
        return parser.parse_identifier()
    # Annotated[list[int], "shape"] -> <2x3>
    if tag == "shape" and get_origin(base) is list:
        parser.expect("<")
        dims = [parser.parse_int()]
        while parser.peek() == "x":
            parser.expect("x")
            dims.append(parser.parse_int())
        parser.expect(">")
        return dims
    # Annotated[list[str], "ssa"] -> [%a, %b]
    if tag == "ssa" and get_origin(base) is list:
        parser.expect("[")
        parser.skip_whitespace()
        items = []
        if parser.peek() != "]":
            items.append(parser.parse_ssa_name())
            while parser.peek() == ",":
                parser.expect(",")
                parser.skip_whitespace()
                items.append(parser.parse_ssa_name())
        parser.expect("]")
        return items
    # Annotated[list[str], "bare"] -> [a, b]
    if tag == "bare" and get_origin(base) is list:
        parser.expect("[")
        parser.skip_whitespace()
        items = []
        if parser.peek() != "]":
            items.append(parser.parse_identifier())
            while parser.peek() == ",":
                parser.expect(",")
                parser.skip_whitespace()
                items.append(parser.parse_identifier())
        parser.expect("]")
        return items
    # Plain int
    if hint is int:
        return parser.parse_int()
    # Plain float
    if hint is float:
        return parser.parse_number()
    # list[float]
    if get_origin(hint) is list and get_args(hint) == (float,):
        parser.expect("[")
        parser.skip_whitespace()
        items = []
        if parser.peek() != "]":
            items.append(parser.parse_number())
            while parser.peek() == ",":
                parser.expect(",")
                parser.skip_whitespace()
                items.append(parser.parse_number())
        parser.expect("]")
        return items

    raise RuntimeError(f"Don't know how to parse type hint: {hint}")


# ===----------------------------------------------------------------------=== #
# @op decorator and table builder
# ===----------------------------------------------------------------------=== #


def op(asm_name: str):
    """Decorator: attaches _asm_name, injects generic asm property."""

    def decorator(cls):
        cls = dataclasses.dataclass(cls)
        cls._asm_name = asm_name

        @property
        def _asm(self) -> Iterable[str]:
            return op_asm(self)

        cls.asm = _asm
        return cls

    return decorator


def build_tables(
    op_classes: list[type],
) -> tuple[dict[str, type], dict[str, type]]:
    """Build (op_table, keyword_table) from a list of @op-decorated classes.

    - Classes with a 'result' field go in op_table (result ops).
    - Classes without 'result' go in keyword_table.
    - Classes with 'result: Ssa | None' go in BOTH tables.
    """
    op_table: dict[str, type] = {}
    keyword_table: dict[str, type] = {}

    for cls in op_classes:
        name = cls._asm_name
        hints = get_type_hints(cls, include_extras=True)
        has_result = "result" in hints

        if not has_result:
            keyword_table[name] = cls
        else:
            result_hint = hints["result"]
            if _is_optional(result_hint) is not None:
                # Optional result -> both tables
                op_table[name] = cls
                keyword_table[name] = cls
            else:
                op_table[name] = cls

    return op_table, keyword_table
