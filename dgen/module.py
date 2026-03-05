"""Framework-level IR items: Module, ConstantOp, Function, and helpers.

These complement the generated builtin dialect but live outside the dialect
file to keep it purely generated.  Import triggers monkey-patches for
FunctionOp.asm and List.for_value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from dgen import Constant, Dialect, Op, Type, Value
from dgen.asm.formatting import format_func
from dgen import layout
from dgen.dialects.builtin import (
    FunctionOp,
    HasSingleBlock,
    Index,
    List,
    String,
    builtin,
)

from dgen.type import Memory

# ===----------------------------------------------------------------------=== #
# Function type (not dialect-registered)
# ===----------------------------------------------------------------------=== #


@dataclass
class Function(Type):
    """A function signature."""

    __layout__ = layout.Void()
    result: Type


# ===----------------------------------------------------------------------=== #
# ConstantOp (custom __init__, multiple inheritance)
# ===----------------------------------------------------------------------=== #


@builtin.op("constant")
@dataclass(eq=False, kw_only=True, init=False)
class ConstantOp(Op, Constant):
    value: Memory
    type: Type

    __operands__ = (("value", Type),)

    def __init__(self, *, value: object, type: Type, name: str | None = None) -> None:
        self.name = name
        self.type = type
        self.value = (
            value if isinstance(value, Memory) else Memory.from_value(type, value)
        )

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


def _collect_dialects(func: Op, dialects: set[Dialect]) -> None:
    """Collect all non-builtin dialects referenced by ops and types in a function."""

    def _check_type(t: object) -> None:
        if t is None:
            return
        d = getattr(t, "dialect", None)
        if d is not None and d.name != "builtin":
            dialects.add(d)

    for op in _walk_all_ops(func):
        if op.dialect.name != "builtin":
            dialects.add(op.dialect)
        _check_type(op.type)
    for arg in func.body.args:
        _check_type(arg.type)
    _check_type(func.type.result)


@dataclass
class Module:
    functions: list[Op]

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


@classmethod  # type: ignore[misc]
def _list_for_value(cls: type[List], value: object) -> List:
    assert isinstance(value, list)
    return cls(element_type=Index())


List.for_value = _list_for_value  # type: ignore[assignment]


@property  # type: ignore[misc]
def _function_asm(self: FunctionOp) -> Iterable[str]:
    return format_func(self)


FunctionOp.asm = _function_asm  # type: ignore[assignment, misc]
