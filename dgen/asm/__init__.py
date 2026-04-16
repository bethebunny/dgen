"""IR text serialization — format and parse."""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.ir.traversal import all_values, transitive_dependencies
from dgen.memory import Memory
from dgen.type import Type, Value

from .formatting import SlotTracker, _is_sugar_op, indent, op_asm
from .parser import ASMParser, parse, value_expression


def format(value: Value) -> str:
    """Format a value as ASM text with dialect ``import`` lines."""
    return "\n".join(asm_with_imports(value)).rstrip("\n") + "\n"


def asm_with_imports(value: Value) -> Iterator[str]:
    """Format a value and all its dependencies as IR text with import lines.

    Walks ``transitive_dependencies(value)`` in topological order, emitting
    every non-sugar Op as a top-level SSA statement. Sugar ops (``PackOp``)
    are inlined as ``[...]`` where referenced, matching ``Module.asm``.
    Dialects touched anywhere in the value's use-def graph (including nested
    block bodies) are emitted as leading ``import`` lines.
    """
    builtin_dialect = dgen.Dialect.get("builtin")
    dialects: set[dgen.Dialect] = {
        d for v in all_values(value) for d in v.required_dialects()
    }
    dialects.discard(builtin_dialect)

    for d in sorted(dialects, key=lambda d: d.name):
        yield f"import {d.name}"
    if dialects:
        yield ""

    tracker = SlotTracker()
    formatted: set[int] = set()
    for v in transitive_dependencies(value):
        if isinstance(v, dgen.Op) and not _is_sugar_op(v):
            yield from op_asm(v, tracker, formatted=formatted)


def memory_from_asm(type: Type, text: str) -> Memory:
    """Create Memory from a Type and an ASM literal string."""
    parser = ASMParser(text)
    value = value_expression(parser)
    return Memory.from_value(type, value)


__all__ = ["asm_with_imports", "format", "indent", "memory_from_asm", "parse"]
