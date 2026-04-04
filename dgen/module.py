"""Framework-level IR items: Module, ConstantOp, and helpers.

These complement the generated builtin dialect but live outside the dialect
file to keep it purely generated.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from dgen import Constant, Dialect, Op, Type, TypeType, Value
from dgen.dialects.builtin import (
    Nil,
    Span,
    String,
    builtin,
)
from dgen.dialects.function import FunctionOp
from dgen.type import Fields, Memory, SlotFn, _default_slot, format_value, type_constant

# ===----------------------------------------------------------------------=== #
# ConstantOp (custom __init__, multiple inheritance)
# ===----------------------------------------------------------------------=== #


@builtin.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op, Constant):
    value: object
    type: Value[TypeType]

    @cached_property
    def memory(self) -> Memory:
        if isinstance(self.value, Memory):
            return self.value
        return Memory.from_value(type_constant(self.type), self.value)

    format_asm = Value.format_asm  # SSA reference, not Constant literal

    @property
    def __constant__(self) -> Memory:
        return self.memory

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# ===----------------------------------------------------------------------=== #
# PackOp (sugar op, never emitted standalone in ASM)
# ===----------------------------------------------------------------------=== #


@builtin.op("pack")
@dataclass(eq=False, kw_only=True)
class PackOp(Op):
    """Sugar op: wraps multiple values into a single list-like operand.

    Never emitted standalone in ASM — the formatter inlines it as [...].
    The codegen skips it entirely.
    """

    values: list[Value]
    type: Type
    __operands__: ClassVar[Fields] = ()

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """PackOp always inlines as ``[elem, elem, ...]`` sugar."""
        return "[" + ", ".join(format_value(v, slot) for v in self.values) + "]"

    def __iter__(self) -> Iterator[Value]:
        return iter(self.values)

    @property
    def operands(self) -> Iterator[tuple[str, Value]]:
        for i, v in enumerate(self.values):
            yield f"values[{i}]", v

    def replace_operand(self, old: Value, new: Value) -> None:
        self.values = [new if v is old else v for v in self.values]

    @property
    def __constant__(self) -> Memory:
        json_list = [v.__constant__.to_json() for v in self.values]
        return Memory.from_json(self.type, json_list)


def pack(values: Iterable[Value] = ()) -> PackOp:
    """Create a PackOp, inferring the element type from the values."""
    vals = list(values)
    return PackOp(values=vals, type=Span(pointee=vals[0].type if vals else Nil()))


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def string_value(v: Value[String]) -> str:
    """Extract the Python str from a string Value (must be Constant)."""
    result = v.__constant__.to_json()
    assert isinstance(result, str)
    return result


# ===----------------------------------------------------------------------=== #
# Module (top-level container)
# ===----------------------------------------------------------------------=== #


def _walk_all_ops(op: Op, _visited: set[int] | None = None) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies.

    Tracks visited ops to avoid infinite recursion when label bodies
    reference other labels (e.g. loops with back-edges).
    """
    if _visited is None:
        _visited = set()
    oid = id(op)
    if oid in _visited:
        return
    _visited.add(oid)
    yield op
    for _, block in op.blocks:
        for child in block.ops:
            yield from _walk_all_ops(child, _visited)


@dataclass
class Module:
    ops: list[Op]

    @property
    def functions(self) -> list[FunctionOp]:
        return [op for op in self.ops if isinstance(op, FunctionOp)]

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import SlotTracker, op_asm
        from .graph import transitive_dependencies

        # Format ops first, then collect dialects from the IR graph.
        formatted: set[int] = set()
        op_lines: list[str] = []
        for op in self.ops:
            tracker = SlotTracker()
            op_lines.extend(op_asm(op, tracker, formatted=formatted))
            op_lines.append("")

        # Collect dialects by walking every op and all its value dependencies.
        builtin_dialect = Dialect.get("builtin")
        dialects: set[Dialect] = set()
        for op in self.ops:
            for inner in _walk_all_ops(op):
                for v in transitive_dependencies(inner):
                    if hasattr(v, "dialect"):
                        dialects.add(v.dialect)
        dialects.discard(builtin_dialect)

        for d in sorted(dialects, key=lambda d: d.name):
            yield f"import {d.name}"
        if dialects:
            yield ""

        yield from op_lines


def asm_with_imports(value: Value) -> Iterator[str]:
    """Format a value and all its dependencies as IR text with import lines.

    Walks ``transitive_dependencies(value)`` in topological order, emitting
    every non-sugar Op as a top-level SSA statement. Sugar ops (``PackOp``)
    are inlined as ``[...]`` where referenced, matching ``Module.asm``.
    Dialects touched anywhere in the value's use-def graph (including nested
    block bodies) are emitted as leading ``import`` lines.
    """
    from .asm.formatting import SlotTracker, _is_sugar_op, op_asm
    from .graph import all_values, transitive_dependencies

    builtin_dialect = Dialect.get("builtin")
    dialects: set[Dialect] = {
        v.dialect for v in all_values(value) if hasattr(v, "dialect")
    }
    dialects.discard(builtin_dialect)

    for d in sorted(dialects, key=lambda d: d.name):
        yield f"import {d.name}"
    if dialects:
        yield ""

    tracker = SlotTracker()
    formatted: set[int] = set()
    for v in transitive_dependencies(value):
        if isinstance(v, Op) and not _is_sugar_op(v):
            yield from op_asm(v, tracker, formatted=formatted)
            yield ""


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

builtin.type("Type")(TypeType)
