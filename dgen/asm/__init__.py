"""IR text serialization — format and parse."""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.ir.traversal import all_values, transitive_dependencies
from dgen.memory import Memory
from dgen.type import Constant, Type, Value, constant

from .formatting import SlotTracker, _is_sugar_op, indent, op_asm
from .parser import ASMParser, parse, value_expression


def format(value: Value) -> str:
    """Format a value as ASM text with dialect ``import`` lines."""
    return "\n".join(asm_with_imports(value))


def _embedded_type_dialects(value: object) -> Iterator[dgen.Dialect]:
    """Walk a rich payload and yield dialects of any embedded ``Type`` instances.

    Constants can carry type references in their payload that aren't
    reachable through ``Value.dependencies`` — e.g. a TypeType constant
    whose value is ``index.Index``, or an ``Any`` whose existential field
    is ``number.SignedInteger``. The structurally clean fix would be to
    expose those types as Constant dependencies; until then this walker
    keeps the import discovery honest.
    """
    if isinstance(value, Type):
        yield value.dialect
        for _, param in value.parameters:
            yield from _embedded_type_dialects(param)
    elif isinstance(value, Value):
        yield from _embedded_type_dialects(value.type)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _embedded_type_dialects(v)
    elif isinstance(value, list):
        for item in value:
            yield from _embedded_type_dialects(item)


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
        v.dialect for v in all_values(value) if hasattr(v, "dialect")
    }
    for v in all_values(value):
        if isinstance(v, Constant):
            dialects.update(_embedded_type_dialects(constant(v)))
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
            yield ""


def memory_from_asm(type: Type, text: str) -> Memory:
    """Create Memory from a Type and an ASM literal string."""
    parser = ASMParser(text)
    value = value_expression(parser)
    return Memory.from_value(type, value)


__all__ = ["asm_with_imports", "format", "indent", "memory_from_asm", "parse"]
