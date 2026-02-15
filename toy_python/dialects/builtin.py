"""Builtin structure types shared across all dialects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Protocol

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


# ===----------------------------------------------------------------------=== #
# Builtin ReturnOp
# ===----------------------------------------------------------------------=== #

_Ssa = Annotated[str, "ssa"]


@dataclass
class ReturnOp:
    value: _Ssa | None
    _asm_name = "return"
    _dialect_name = "builtin"
    _builtin = True

    @property
    def asm(self):
        from toy_python.ir_format import op_asm

        return op_asm(self)


KEYWORD_TABLE = {"return": ReturnOp}
OP_TABLE: dict = {}
TYPE_TABLE: dict = {}


# ===----------------------------------------------------------------------=== #
# Function and Module
# ===----------------------------------------------------------------------=== #


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
        # Collect dialect info from all ops
        dialects: set[str] = set()
        builtin_names: set[str] = {"function"}
        for func in self.functions:
            for op in _walk_all_ops(func.body.ops):
                dialect_name = getattr(type(op), "_dialect_name", "builtin")
                asm_name = getattr(type(op), "_asm_name", "")
                if dialect_name == "builtin":
                    builtin_names.add(asm_name)
                else:
                    dialects.add(dialect_name)

        yield f"from builtin import {', '.join(sorted(builtin_names))}"
        for d in sorted(dialects):
            yield f"import {d}"
        yield ""

        for function in self.functions:
            yield from function.asm
            yield ""
