"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from toy_python import asm


class Type(Protocol):
    """Any dialect type."""

    @property
    def asm(self) -> str: ...


class Nil:
    """Represents a void/empty return type."""

    @property
    def asm(self) -> str:
        return "()"


@dataclass
class FuncType:
    """A function signature."""

    result: Type


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


@dataclass
class FuncOp:
    name: str
    body: Block
    func_type: FuncType

    @property
    def asm(self) -> Iterable[str]:
        args = ", ".join(f"%{a.name}: {a.type.asm}" for a in self.body.args)
        yield f"%{self.name} = function ({args}) -> {self.func_type.result.asm}:"
        yield from asm.indent(self.body.asm)


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        for i, function in enumerate(self.functions):
            yield from function.asm
            yield ""
