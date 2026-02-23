"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dgen import Block, Dialect, Op, Type, Value, asm
from dgen.asm.formatting import SlotTracker, format_expr, op_asm
from dgen.layout import BYTE, FLOAT64, INT, FatPointer


# ===----------------------------------------------------------------------=== #
# Builtin ReturnOp
# ===----------------------------------------------------------------------=== #

builtin = Dialect("builtin")


@builtin.type("index")
@dataclass(frozen=True)
class IndexType:
    __layout__ = INT


@builtin.type("f64")
@dataclass(frozen=True)
class F64Type:
    __layout__ = FLOAT64


@builtin.type("Nil")
@dataclass(frozen=True)
class Nil:
    """Represents a void/empty return type."""


@dataclass
class Function:
    """A function signature."""

    result: Type


@builtin.type("String")
@dataclass(frozen=True)
class String:
    __layout__ = FatPointer(BYTE)


@builtin.type("List")
@dataclass
class List:
    element_type: Type


@builtin.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op):
    value: object
    type: Type


@builtin.op("add_index")
@dataclass(eq=False, kw_only=True)
class AddIndexOp(Op):
    lhs: Value
    rhs: Value
    type: Type = IndexType()


@builtin.op("return")
@dataclass(eq=False, kw_only=True)
class ReturnOp(Op):
    value: Value | None = None
    type: Type = Nil()


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


@builtin.op("function")
@dataclass(eq=False, kw_only=True)
class FuncOp(Op):
    body: Block
    type: Function

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


def _register_ops(tracker, ops: list[Op]):
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


def _collect_type_dialects(func: FuncOp, dialects: set):
    """Collect non-builtin dialects referenced by types in a function."""

    def _check(t):
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
