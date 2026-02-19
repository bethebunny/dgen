"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields, field
from typing import (
    ClassVar,
    NewType,
    Protocol,
    get_type_hints,
)

StaticString = NewType('StaticString', str)

from toy_python import asm
from toy_python.dialect import Dialect


class Type(Protocol):
    """Any dialect type."""

    @property
    def asm(self) -> str: ...


@dataclass(eq=False, kw_only=True)
class Value:
    """Base class for SSA values. An op or block argument."""

    name: str | None = None

    @property
    def operands(self) -> list[Value]:
        return []

    @property
    def blocks(self) -> dict[str, Block]:
        return {}


@dataclass(eq=False, kw_only=True)
class BlockArg(Value):
    """A block argument (function parameter)."""

    type: Type = None  # type: ignore[assignment]


@dataclass(eq=False, kw_only=True)
class Op(Value):
    """Base class for all dialect operations."""

    _asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]

    @property
    def operands(self) -> list[Value]:
        """All Value-typed fields (auto-introspected)."""
        result: list[Value] = []
        for f in fields(self):
            if f.name == "name":
                continue
            val = getattr(self, f.name)
            if isinstance(val, Value):
                result.append(val)
            elif isinstance(val, list):
                result.extend(v for v in val if isinstance(v, Value))
        return result

    @property
    def blocks(self) -> dict[str, Block]:
        """All Block-typed fields as a name->block dict (auto-introspected)."""
        hints = get_type_hints(type(self), include_extras=True)
        result: dict[str, Block] = {}
        for fname, hint in hints.items():
            if hint is Block:
                result[fname] = getattr(self, fname)
        return result

    @property
    def asm(self) -> Iterable[str]:
        from toy_python.asm.formatting import op_asm

        return op_asm(self)


@dataclass
class Block:
    ops: list[Op]
    args: list[BlockArg] = field(default_factory=list)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm


# ===----------------------------------------------------------------------=== #
# Builtin ReturnOp
# ===----------------------------------------------------------------------=== #

builtin = Dialect("builtin")


@dataclass
class IndexType:
    from toy_python.layout import INT

    __layout__ = INT

    @property
    def asm(self) -> str:
        return "index"


@builtin.type("index")
def _parse_index_type(_parser):
    return IndexType()


@dataclass
class F64Type:
    from toy_python.layout import FLOAT64

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
    from toy_python.layout import FatPointer, BYTE

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
    value: float | int | list[float]
    type: Type


@builtin.op("return")
@dataclass(eq=False, kw_only=True)
class ReturnOp(Op):
    value: Value | None = None


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


@builtin.op("function")
@dataclass(eq=False, kw_only=True)
class FuncOp(Op):
    body: Block = None  # type: ignore[assignment]
    func_type: Function = None  # type: ignore[assignment]

    @property
    def asm(self) -> Iterable[str]:
        from toy_python.asm.formatting import SlotTracker, op_asm

        tracker = SlotTracker()
        # Pre-register block args and all ops in this function
        for arg in self.body.args:
            tracker.get_name(arg)
        _register_ops(tracker, self.body.ops)

        name = tracker.get_name(self)
        args = ", ".join(
            f"%{tracker.get_name(a)}: {a.type.asm}" for a in self.body.args
        )
        yield f"%{name} = function ({args}) -> {self.func_type.result.asm}:"
        for op in self.body.ops:
            yield from asm.indent(op_asm(op, tracker))


def _register_ops(tracker, ops: list[Op]):
    """Pre-register all ops in a tracker so slot numbers are stable."""
    for op in ops:
        tracker.get_name(op)
        for block in op.blocks.values():
            for arg in block.args:
                tracker.get_name(arg)
            _register_ops(tracker, block.ops)


def _walk_all_ops(op: Op) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies."""
    yield op
    for block in op.blocks.values():
        for child in block.ops:
            yield from _walk_all_ops(child)


def _collect_type_dialects(func: FuncOp, dialects: set):
    """Collect non-builtin dialects referenced by types in a function."""
    def _check(t):
        d = getattr(t, '_dialect', None)
        if d is not None and d.name != "builtin":
            dialects.add(d)

    for op in _walk_all_ops(func):
        _check(getattr(op, 'type', None))
    for arg in func.body.args:
        _check(arg.type)
    _check(func.func_type.result)


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

        for d in sorted(dialects):
            yield f"import {d.name}"
        if dialects:
            yield ""

        for function in self.functions:
            yield from function.asm
            yield ""
