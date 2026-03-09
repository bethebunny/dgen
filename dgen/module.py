"""Framework-level IR items: Module, ConstantOp, Function, and helpers.

These complement the generated builtin dialect but live outside the dialect
file to keep it purely generated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from dgen import Constant, Dialect, Op, Type, TypeType, Value
from dgen import layout
from dgen.dialects.builtin import (
    FunctionOp,
    HasSingleBlock,
    String,
    builtin,
)

from dgen.type import Fields, Memory, type_constant

# ===----------------------------------------------------------------------=== #
# Function type
# ===----------------------------------------------------------------------=== #


@builtin.type("Function")
@dataclass
class Function(Type):
    """A function signature."""

    __layout__ = layout.Void()
    __params__: ClassVar[Fields] = (("result", Type),)
    result: Value[TypeType]


# ===----------------------------------------------------------------------=== #
# ConstantOp (custom __init__, multiple inheritance)
# ===----------------------------------------------------------------------=== #


@builtin.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op, Constant):
    value: object
    type: Value[TypeType]

    __operands__ = (("value", Type),)

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


def _walk_all_ops(op: Op) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies."""
    yield op
    for _, block in op.blocks:
        for child in block.ops:
            yield from _walk_all_ops(child)


def _collect_dialects(func: FunctionOp, dialects: set[Dialect]) -> None:
    """Collect all non-builtin dialects referenced by ops and types in a function."""

    def _check_type(t: Value) -> None:
        if isinstance(t, Type) and t.dialect.name != "builtin":
            dialects.add(t.dialect)

    for op in _walk_all_ops(func):
        if op.dialect.name != "builtin":
            dialects.add(op.dialect)
        _check_type(op.type)
    for arg in func.body.args:
        _check_type(arg.type)
    _check_type(func.result)


@dataclass
class Module:
    functions: list[FunctionOp]

    @property
    def asm(self) -> Iterable[str]:
        dialects: set[Dialect] = set()
        for func in self.functions:
            _collect_dialects(func, dialects)

        for d in sorted(dialects, key=lambda d: d.name):
            yield f"import {d.name}"
        if dialects:
            yield ""

        for function in self.functions:
            yield from function.asm
            yield ""


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

HasSingleBlock.__annotations__["__blocks__"] = ClassVar[tuple[str, ...]]
builtin.type("Type")(TypeType)
