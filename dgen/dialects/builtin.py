"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NewType

from dgen import Block, Dialect, Op, Type, Value, asm
from dgen.asm.formatting import SlotTracker, op_asm
from dgen.layout import BYTE, FLOAT64, INT, FatPointer

StaticString = NewType("StaticString", str)


# ===----------------------------------------------------------------------=== #
# Builtin ReturnOp
# ===----------------------------------------------------------------------=== #

builtin = Dialect("builtin")


@dataclass
class IndexType:
    __layout__ = INT

    @property
    def asm(self) -> str:
        return "index"


@builtin.type("index")
def _parse_index_type(_parser):
    return IndexType()


@dataclass
class F64Type:
    __layout__ = FLOAT64

    @property
    def asm(self) -> str:
        return "f64"


@builtin.type("f64")
def _parse_f64_type(_parser):
    return F64Type()


@builtin.type("Nil")
class Nil:
    """Represents a void/empty return type."""

    @property
    def asm(self) -> str:
        return "()"


@builtin.type("Function")
@dataclass
class Function:
    """A function signature."""

    result: Type


@builtin.type("String")
@dataclass
class String:
    __layout__ = FatPointer(BYTE)

    @property
    def asm(self) -> str:
        return "String"


@builtin.type("List")
@dataclass
class List:
    element_type: Type

    @property
    def asm(self) -> str:
        return f"List[{self.element_type.asm}]"


@builtin.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op):
    value: object
    type: Type


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
                arg_parts.append(f"%{n}: {a.type.asm}")
            else:
                arg_parts.append(f"%{n}")
        args = ", ".join(arg_parts)
        yield f"%{name} = function ({args}) -> {self.type.result.asm}:"
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
        d = getattr(t, "_dialect", None)
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
