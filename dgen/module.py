"""Framework-level IR items: Module, ConstantOp, and helpers.

These complement the generated builtin dialect but live outside the dialect
file to keep it purely generated.
"""

from __future__ import annotations

import re
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


from dgen.type import Fields, Memory, type_constant

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


_DIALECT_REF = re.compile(r"\b([a-z_][a-z0-9_]*)\.\w")


def _dialects_from_text(lines: Iterable[str]) -> set[str]:
    """Scan formatted ASM lines for dialect-qualified names (e.g. ``number.Float64``)."""
    registered = Dialect._registry
    result: set[str] = set()
    for line in lines:
        for m in _DIALECT_REF.finditer(line):
            name = m.group(1)
            if name in registered and name != "builtin":
                result.add(name)
    return result


def _collect_dialects(op: Op, dialects: set[Dialect]) -> None:
    """Collect all non-builtin dialects referenced by ops and types."""

    def _check_value(v: object) -> None:
        """Collect dialects from a value and its type, recursively."""
        if isinstance(v, Type):
            if v.dialect.name != "builtin":
                dialects.add(v.dialect)
            for _, param in v.parameters:
                _check_value(param)
        elif isinstance(v, Value):
            _check_value(v.type)

    for child_op in _walk_all_ops(op):
        if child_op.dialect.name != "builtin":
            dialects.add(child_op.dialect)
        _check_value(child_op.type)
        for _, param in child_op.parameters:
            _check_value(param)
        for _, block in child_op.blocks:
            for arg in block.args:
                _check_value(arg.type)


@dataclass
class Module:
    ops: list[Op]

    @property
    def functions(self) -> list[FunctionOp]:
        return [op for op in self.ops if isinstance(op, FunctionOp)]

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        # Format ops first, then scan for dialect references to emit imports.
        # This is more robust than walking the IR data structure, because
        # formatted text may embed dialect-qualified names in places that
        # _collect_dialects cannot easily reach (e.g. type-valued constants
        # inside Tuple or Span<TypeType> parameters).

        # The _formatted set is shared across all top-level ops so that each
        # op is printed at most once.  Ambient ops (no block-argument
        # dependencies — e.g. FunctionOps, constants) are reachable via
        # block.ops from every block that references them, so without sharing
        # they would be printed once at module level AND again inside each
        # referencing block.  Sharing _formatted makes the formatter an
        # implicit scheduler: an ambient op is emitted wherever the formatter
        # first encounters it (here, in module.ops order) and suppressed
        # elsewhere.
        #
        # TODO: formalize scheduling as an explicit pass that assigns each op
        # to exactly one block, rather than relying on first-encounter dedup.
        formatted: set[int] = set()
        op_lines: list[str] = []
        for op in self.ops:
            op_lines.extend(op_asm(op, _formatted=formatted))
            op_lines.append("")

        dialects = _dialects_from_text(op_lines)

        for d in sorted(dialects):
            yield f"import {d}"
        if dialects:
            yield ""

        yield from op_lines


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

builtin.type("Type")(TypeType)
