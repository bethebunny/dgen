"""Dialect-agnostic IR text deserialization.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.  Dialect knowledge is discovered from import headers
at the top of the IR text.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import re
from typing import Any

from dgen import Block, Constant, Dialect, Op, Type, TypeType, Value
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module

_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_QUALIFIED = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
_STRING = re.compile(r'"[^"]*"')
_SSA = re.compile(r"%[a-zA-Z0-9_]+")
_FLOAT = re.compile(r"-?\d+\.\d*|-?\d*\.\d+")
_INT = re.compile(r"\d+")
_NUMBER = re.compile(r"-?\d+\.\d*|-?\d*\.\d+|-?\d+")


class Token(enum.Enum):
    IDENTIFIER = r"[a-zA-Z_][a-zA-Z0-9_]*"


def parse_module(asm: str) -> Module:
    module = Module()
    namespace = {}

    parser = ASMParser(asm)
    while not parser.done:
        if (imports := parser.try_read(ImportStatement)) is not None:
            namespace.update(imports.namespace)
        else:
            module.functions.append(parser.read(OpStatement).op)

    return module


class ImportStatement:
    @classmethod
    def read(cls, parser: ASMParser) -> ImportStatement:
        pass

    @property
    def namespace(self) -> dict[str, Op]: ...


class OpStatement:
    @classmethod
    def read(cls, parser: ASMParser) -> OpStatement:
        name = parser.read(SSAName)
        type = parser.read(TypeExpression) if parser.try_read(":") else None
        parser.read("=")
        op = parser.read(OpExpression)
        return OpStatement(op, type, name)

    @property
    def op(self) -> Op: ...


class LiteralExpression:
    @classmethod
    def read(cls, parser: ASMParser) -> LiteralExpression:
        pass


class TypeExpression:
    @classmethod
    def read(cls, parser: ASMParser) -> TypeExpression:
        if (literal := parser.try_read(DictLiteral)) is not None:
            # TODO: probably turn this into a type immediately
            return literal
        name = parser.read(QualifiedName)
        type = namespace.lookup(name.name)
        parameters = []
        if type.__params__:
            parser.read("<")
            parameters = parser.read_list((TypeExpression, LiteralExpression))
            parser.read(">")
        return TypeExpression(type, parameters)

    @property
    def type(self) -> Type: ...


class OpExpression:
    @classmethod
    def read(cls, parser: ASMParser) -> OpExpression:
        pass
