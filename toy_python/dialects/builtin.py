"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import types
from typing import Annotated, ClassVar, Protocol, Union, get_type_hints, get_origin, get_args

from toy_python import asm
from toy_python.dialect import Dialect

Ssa = Annotated[str, "ssa"]  # %name
String = Annotated[str, "string"]  # name (as-is)
StringList = Annotated[list[str], "string"]  # [a, b]


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


@dataclass
class Op:
    """Base class for all dialect operations."""

    result: Ssa

    _asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]

    @property
    def operands(self) -> list[str]:
        """All Ssa-annotated fields except result (auto-introspected)."""
        hints = get_type_hints(type(self), include_extras=True)
        result: list[str] = []
        for name, hint in hints.items():
            if name == "result":
                continue
            inner = _unwrap_optional(hint)
            effective = inner if inner is not None else hint
            if get_origin(effective) is Annotated and get_args(effective)[1] == "ssa":
                value = getattr(self, name)
                if value is None:
                    continue
                if isinstance(value, list):
                    result.extend(value)
                else:
                    result.append(value)
        return result

    @property
    def blocks(self) -> dict[str, Block]:
        """All Block-typed fields as a name->block dict (auto-introspected)."""
        hints = get_type_hints(type(self), include_extras=True)
        result: dict[str, Block] = {}
        for name, hint in hints.items():
            if hint is Block:
                result[name] = getattr(self, name)
        return result

    @property
    def asm(self) -> Iterable[str]:
        from toy_python.asm.formatting import op_asm

        return op_asm(self)


@dataclass
class Value:
    """A block argument (function parameter)."""

    name: str
    type: Type


@dataclass
class Block:
    ops: list[Op]
    args: list[Value] = field(default_factory=list)

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


@builtin.op("return")
class ReturnOp(Op):
    value: Ssa | None


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


@builtin.op("function")
class FuncOp(Op):
    body: Block
    func_type: Function

    @property
    def asm(self) -> Iterable[str]:
        args = ", ".join(f"%{a.name}: {a.type.asm}" for a in self.body.args)
        yield f"%{self.result} = function ({args}) -> {self.func_type.result.asm}:"
        yield from asm.indent(self.body.asm)


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
