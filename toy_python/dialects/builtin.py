"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Protocol

from toy_python import asm
from toy_python.dialect import Dialect

Ssa = Annotated[str, "ssa"]  # %name
String = Annotated[str, "string"]  # name (as-is)
StringList = Annotated[list[str], "string"]  # [a, b]


class Type(Protocol):
    """Any dialect type."""

    @property
    def asm(self) -> str: ...


class Op(Protocol):
    """Any dialect operation (has an asm property)."""

    @property
    def asm(self) -> Iterable[str]: ...


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
class ReturnOp:
    result: Ssa
    value: Ssa | None


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


@builtin.op("function")
@dataclass
class FuncOp:
    result: Ssa
    body: Block
    func_type: Function

    @property
    def asm(self) -> Iterable[str]:
        args = ", ".join(f"%{a.name}: {a.type.asm}" for a in self.body.args)
        yield f"%{self.result} = function ({args}) -> {self.func_type.result.asm}:"
        yield from asm.indent(self.body.asm)


def _walk_all_ops(ops) -> Iterable:
    """Recursively yield all ops, descending into op bodies."""
    for op in ops:
        yield op
        body = getattr(op, "body", None)
        if isinstance(body, list):
            yield from _walk_all_ops(body)


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        # Collect non-builtin dialects used
        dialects: set[str] = set()
        for func in self.functions:
            for op in _walk_all_ops(func.body.ops):
                dialect_name = getattr(type(op), "_dialect_name", "builtin")
                if dialect_name != "builtin":
                    dialects.add(dialect_name)

        for d in sorted(dialects):
            yield f"import {d}"
        if dialects:
            yield ""

        for function in self.functions:
            yield from function.asm
            yield ""
