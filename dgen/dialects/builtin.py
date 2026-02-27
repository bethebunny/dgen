"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from dgen import Block, Constant, Dialect, Op, Type, Value, asm
from dgen.asm.formatting import SlotTracker, format_expr, op_asm
from dgen.layout import BYTE, FLOAT64, INT, VOID, FatPointer
from dgen.type import Memory

# ===----------------------------------------------------------------------=== #
# Builtin ReturnOp
# ===----------------------------------------------------------------------=== #

builtin = Dialect("builtin")


@builtin.type("index")
@dataclass(frozen=True)
class IndexType(Type):
    __layout__ = INT


@builtin.type("f64")
@dataclass(frozen=True)
class F64Type(Type):
    __layout__ = FLOAT64


@builtin.type("Nil")
@dataclass(frozen=True)
class Nil(Type):
    """Represents a void/empty return type."""

    __layout__ = VOID


@dataclass
class Function(Type):
    """A function signature."""

    __layout__ = VOID
    result: Type


@builtin.type("String")
@dataclass(frozen=True)
class String(Type):
    __layout__ = FatPointer(BYTE)


@builtin.type("List")
@dataclass
class List(Type):
    __layout__ = FatPointer(BYTE)
    element_type: Type

    __params__ = (("element_type", Type),)


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


@builtin.op("add_index")
@dataclass(eq=False, kw_only=True)
class AddIndexOp(Op):
    lhs: Value[IndexType]
    rhs: Value[IndexType]
    type: Type = IndexType()

    __operands__ = (("lhs", IndexType), ("rhs", IndexType))


@builtin.op("return")
@dataclass(eq=False, kw_only=True)
class ReturnOp(Op):
    value: Value | Nil = Nil()
    type: Type = Nil()

    __operands__ = (("value", Type),)


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


class HasSingleBlock:
    """Trait for ops with a single block region named."""

    __blocks__: ClassVar[tuple[str, ...]]

    @property
    def __body__(self) -> Block:
        return getattr(self, self.__blocks__[0])


@builtin.op("function")
@dataclass(eq=False, kw_only=True)
class FuncOp(HasSingleBlock, Op):
    body: Block
    type: Function

    __blocks__ = ("body",)

    @property
    def asm(self) -> Iterable[str]:
        tracker = SlotTracker()
        # Pre-register block args and all ops in this function
        for arg in self.body.args:
            tracker.get_name(arg)
        _register_ops(tracker, self.body.ops)

        name = tracker.get_name(self)
        arg_parts = []
        for a in self.body.args:
            n = tracker.get_name(a)
            if a.type is not None:
                arg_parts.append(f"%{n}: {format_expr(a.type, tracker)}")
            else:
                arg_parts.append(f"%{n}")
        args = ", ".join(arg_parts)
        yield f"%{name} = function ({args}) -> {format_expr(self.type.result, tracker)}:"
        for op in self.body.ops:
            yield from asm.indent(op_asm(op, tracker))


def _register_ops(tracker: SlotTracker, ops: list[Op]) -> None:
    """Pre-register all ops in a tracker so slot numbers are stable."""
    for op in ops:
        tracker.get_name(op)
        for _, block in op.blocks:
            for arg in block.args:
                tracker.get_name(arg)
            _register_ops(tracker, block.ops)


def _walk_all_ops(op: Op) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies."""
    yield op
    for _, block in op.blocks:
        for child in block.ops:
            yield from _walk_all_ops(child)


def _collect_type_dialects(func: FuncOp, dialects: set[Dialect]) -> None:
    """Collect non-builtin dialects referenced by types in a function.

    Only collects dialects whose names appear in the ASM text output.
    Memory objects format as their value (e.g. [2, 3]), not as their type
    (e.g. affine.Shape(2)), so Memory's internal type dialect is NOT collected.
    """

    def _check(t: object) -> None:
        if t is None:
            return
        d = getattr(t, "dialect", None)
        if d is not None and d.name != "builtin":
            dialects.add(d)

    for op in _walk_all_ops(func):
        _check(getattr(op, "type", None))
    for arg in func.body.args:
        _check(arg.type)
    _check(func.type.result)


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        # Collect non-builtin dialects used (from ops and types)
        dialects: set[Dialect] = set()
        for func in self.functions:
            for op in _walk_all_ops(func):
                if op.dialect.name != "builtin":
                    dialects.add(op.dialect)
            _collect_type_dialects(func, dialects)

        for d in sorted(dialects, key=lambda d: d.name):
            yield f"import {d.name}"
        if dialects:
            yield ""

        for function in self.functions:
            yield from function.asm
            yield ""
