"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from dgen import Block, Constant, Dialect, Op, Type, Value, asm
from dgen.asm.formatting import SlotTracker, _is_sugar_op, format_expr, op_asm
from dgen.layout import FLOAT64, INT, VOID, Array, Bytes, Layout
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
    length: Value[IndexType]

    __params__ = (("length", IndexType),)

    @property
    def __layout__(self) -> Bytes:
        return Bytes(self.length.__constant__.unpack()[0])

    @classmethod
    def for_value(cls, value: object) -> String:
        assert isinstance(value, str)
        return cls(length=IndexType().constant(len(value)))


def string_constant(s: str) -> Constant[String]:
    """Create a Constant[String] from a Python str."""
    return String.for_value(s).constant(s)


def string_value(v: Value[String]) -> str:
    """Extract the Python str from a string Value (must be Constant)."""
    return v.__constant__.unpack()[0].decode("utf-8")


@builtin.type("List")
@dataclass
class List(Type):
    element_type: Type
    count: Value[IndexType]

    __params__ = (("element_type", Type), ("count", IndexType))

    @property
    def __layout__(self) -> Layout:
        n = self.count.__constant__.unpack()[0]
        return Array(self.element_type.__layout__, n)

    @classmethod
    def for_value(cls, value: object) -> List:
        assert isinstance(value, list)
        return cls(
            element_type=IndexType(),
            count=IndexType().constant(len(value)),
        )


@builtin.op("pack")
@dataclass(eq=False, kw_only=True)
class PackOp(Op):
    """Pack values into a List."""

    values: list[Value]
    type: List

    __operands__ = (("values", Type),)


@builtin.op("list_get")
@dataclass(eq=False, kw_only=True)
class ListGetOp(Op):
    """Get one element from a list by index."""

    index: Value[IndexType]
    list: Value
    type: Type

    __params__ = (("index", IndexType),)
    __operands__ = (("list", List),)


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
            tracker.track_name(arg)
        tracker.register(self.body.ops)

        name = tracker.track_name(self)
        arg_parts = []
        for a in self.body.args:
            n = tracker.track_name(a)
            if a.type is not None:
                arg_parts.append(f"%{n}: {format_expr(a.type, tracker)}")
            else:
                arg_parts.append(f"%{n}")
        args = ", ".join(arg_parts)
        yield f"%{name} = function ({args}) -> {format_expr(self.type.result, tracker)}:"
        for op in self.body.ops:
            if _is_sugar_op(op):
                continue
            yield from asm.indent(op_asm(op, tracker))


def _walk_all_ops(op: Op) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies."""
    yield op
    for _, block in op.blocks:
        for child in block.ops:
            yield from _walk_all_ops(child)


def _collect_dialects(func: FuncOp, dialects: set[Dialect]) -> None:
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
        _check_type(getattr(op, "type", None))
    for arg in func.body.args:
        _check_type(arg.type)
    _check_type(func.type.result)


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        # Collect non-builtin dialects used (from ops and types) in one pass
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
