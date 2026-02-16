"""Builtin structure types shared across all dialects."""

from __future__ import annotations

import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from toy_python import asm
from toy_python.dialect import Dialect


class Type(Protocol):
    """Any dialect type."""

    @property
    def asm(self) -> str: ...


def _unwrap_optional(hint):
    """If hint is X | None, return X; otherwise return None."""
    origin = get_origin(hint)
    if origin is Union or isinstance(hint, types.UnionType):
        args = get_args(hint)
        if len(args) == 2 and type(None) in args:
            return args[0] if args[1] is type(None) else args[1]
    return None


def _is_value_hint(hint) -> bool:
    """Check if a type hint refers to Value (including Optional[Value], list[Value])."""
    inner = _unwrap_optional(hint)
    effective = inner if inner is not None else hint
    if isinstance(effective, type) and issubclass(effective, Value):
        return True
    if (
        get_origin(effective) is list
        and get_args(effective)
        and isinstance(get_args(effective)[0], type)
        and issubclass(get_args(effective)[0], Value)
    ):
        return True
    return False


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
        hints = get_type_hints(type(self), include_extras=True)
        result: list[Value] = []
        for fname, hint in hints.items():
            if fname == "name":
                continue
            inner = _unwrap_optional(hint)
            effective = inner if inner is not None else hint
            if isinstance(effective, type) and issubclass(effective, Value):
                value = getattr(self, fname)
                if value is None:
                    continue
                result.append(value)
            elif (
                get_origin(effective) is list
                and get_args(effective)
                and isinstance(get_args(effective)[0], type)
                and issubclass(get_args(effective)[0], Value)
            ):
                value = getattr(self, fname)
                if value is not None:
                    result.extend(value)
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
        from toy_python.asm.formatting import SlotTracker

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
        yield from asm.indent(_emit_block_asm(tracker, self.body))


def _register_ops(tracker, ops: list[Op]):
    """Pre-register all ops in a tracker so slot numbers are stable."""
    for op in ops:
        tracker.get_name(op)
        for block in op.blocks.values():
            for arg in block.args:
                tracker.get_name(arg)
            _register_ops(tracker, block.ops)


def _emit_block_asm(tracker, block: Block) -> Iterable[str]:
    """Emit asm for a block using the given slot tracker."""
    from toy_python.asm.formatting import op_asm

    for op in block.ops:
        yield from op_asm(op, tracker)


def _walk_all_ops(op: Op) -> Iterable[Op]:
    """Recursively yield all ops, descending into op bodies."""
    yield op
    for block in op.blocks.values():
        for child in block.ops:
            yield from _walk_all_ops(child)


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        # Collect non-builtin dialects used
        dialects: set[Dialect] = set()
        for func in self.functions:
            for op in _walk_all_ops(func):
                if op.dialect.name != "builtin":
                    dialects.add(op.dialect)

        for d in sorted(dialects):
            yield f"import {d.name}"
        if dialects:
            yield ""

        for function in self.functions:
            yield from function.asm
            yield ""
