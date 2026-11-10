"""IR text deserialization.

read/try_read dispatch to reader functions — like lisp reader macros.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import dgen.type
from dgen import Block, Constant, Op, Type, Value
from dgen.block import BlockArgument, BlockParameter
from dgen.dialect import Dialect
from dgen.dialects import builtin
from dgen.module import ConstantOp, PackOp
from dgen.type import literal_type
from dgen.verify import verify_closed_blocks, verify_constraints, verify_dag

verify_parsed: ContextVar[bool] = ContextVar("dgen.verify_parsed", default=False)


class ParseError(RuntimeError):
    """A failure to read text at offset pos. try_read() catches it to backtrack."""

    def __init__(self, message: str, *, text: str, pos: int) -> None:
        super().__init__(message)
        self.message = message
        self.text = text
        self.pos = pos

    @property
    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    @property
    def column(self) -> int:
        return self.pos - self.text.rfind("\n", 0, self.pos)

    @property
    def source_line(self) -> str:
        start = self.text.rfind("\n", 0, self.pos) + 1
        end = self.text.find("\n", self.pos)
        return self.text[start : len(self.text) if end == -1 else end]

    def __str__(self) -> str:
        caret = " " * (self.column - 1) + "^"
        return f"line {self.line}, column {self.column}: {self.message}\n    {self.source_line}\n    {caret}"


class FatalParseError(ParseError):
    """A failure no other reading of the text avoids, so try_read() doesn't backtrack past it."""


_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_QUALIFIED = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_SSA = re.compile(r"%[a-zA-Z0-9_]+")
_NUMBER = re.compile(r"-?\d+\.\d*|-?\d*\.\d+|-?\d+")
_LITERAL_START = set('-0123456789{["(')


class Scope(dict[str, "type[Op | Type] | Scope"]):
    """Name-resolution scope for ASM parsing.

    A flat dict of unqualified names (ops and types).  ``import_dialect``
    nests a dialect's contents under its name so ``dialect.Foo`` resolves
    via ``scope.lookup("dialect.Foo")``.
    """

    @classmethod
    def from_dialect(cls, dialect: Dialect) -> Scope:
        return cls(**dialect.ops, **dialect.types)

    def import_dialect(self, dialect: Dialect) -> None:
        self[dialect.name] = Scope.from_dialect(dialect)

    def lookup(self, qualified_name: str) -> type[Op | Type]:
        """The class a dotted name names. Raises KeyError on a miss."""
        result = reduce(
            lambda scope, key: scope[key] if isinstance(scope, Scope) else {}[key],
            qualified_name.split("."),
            self,
        )
        if isinstance(result, Scope):
            raise KeyError(qualified_name)
        return result


@dataclass
class BlockScope:
    """The SSA names one block body sees: its inputs, captures and own statements."""

    label: str
    names: dict[str, Value] = field(default_factory=dict)


def parse(text: str) -> Value:
    """Parse IR text — imports and one or more statements, interleavable — and return the last value.

    Statements and imports may appear in any order. Each statement defines an
    SSA value that subsequent statements can reference by name; the last value
    defined becomes the root of the returned use-def graph (all earlier ones
    reachable as transitive dependencies).
    """
    parser = ASMParser(text)
    value: Value | None = None
    while not parser.done:
        if parser.try_read(_import_line) is None:
            value = parser.read(op_statement)
    if value is None:
        raise parser.error("No statements in input")
    if verify_parsed.get():
        verify_dag(value)
        verify_closed_blocks(value)
        verify_constraints(value)
    return value


class ASMParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos: int = 0
        self.token_start: int = 0  # where the last token read began, for errors
        self.scopes: list[BlockScope] = [BlockScope("the top level")]
        self.scope: Scope = Scope.from_dialect(Dialect.get("builtin"))
        self.block_indent: int = 0  # indent level of the current block's ops

    @property
    def done(self) -> bool:
        self._skip_all()
        return self.pos >= len(self.text)

    def peek(self) -> str:
        self._skip_ws()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def read(self, grammar: str | Callable[..., Any]) -> Any:  # noqa: ANN401
        if isinstance(grammar, str):
            self._skip_ws()
            self._expect(grammar)
            return grammar
        return grammar(self)

    def try_read(self, grammar: str | Callable[..., Any]) -> Any | None:  # noqa: ANN401
        saved = self.pos
        try:
            return self.read(grammar)
        except FatalParseError:
            raise
        except ParseError:
            self.pos = saved
            return None

    def parse_token(self, regex: re.Pattern[str]) -> str | None:
        self._skip_ws()
        if match := regex.match(self.text, self.pos):
            self.token_start = self.pos
            self.pos = match.end()
            return match.group()
        return None

    def expect_token(self, regex: re.Pattern[str], name: str) -> str:
        if (token := self.parse_token(regex)) is not None:
            return token
        raise self.error(f"Expected {name}")

    def read_list(self, reader: Callable[..., Any]) -> list[Any]:
        """Read comma-separated items until reader fails."""
        items: list[Any] = []
        if (first := self.try_read(reader)) is not None:
            items.append(first)
            while self.try_read(",") is not None:
                items.append(self.read(reader))
        return items

    def error(self, message: str, *, at: int | None = None) -> ParseError:
        return ParseError(message, text=self.text, pos=self.pos if at is None else at)

    def fatal(self, message: str, *, at: int) -> FatalParseError:
        return FatalParseError(message, text=self.text, pos=at)

    def resolve(self, name: str) -> Value:
        if (value := self.scopes[-1].names.get(name)) is not None:
            return value
        if any(name in scope.names for scope in self.scopes):
            raise self.fatal(
                f"%{name} is not in scope in {self.scopes[-1].label}: capture it",
                at=self.token_start,
            )
        raise self.fatal(f"Undefined reference: %{name}", at=self.token_start)

    def lookup(self, name: str) -> type[Op | Type]:
        try:
            return self.scope.lookup(name)
        except KeyError:
            raise self.fatal(f"Unknown name: {name}", at=self.token_start) from None

    def define(self, name: str, value: Value) -> None:
        self.scopes[-1].names[name] = value

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def _skip_all(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _expect(self, string: str) -> None:
        for ch in string:
            if self.pos >= len(self.text) or self.text[self.pos] != ch:
                raise self.error(f"Expected '{string}'")
            self.pos += 1


# -- Reader functions -------------------------------------------------------


def ssa_name(parser: ASMParser) -> str:
    return parser.expect_token(_SSA, "SSA name")[1:]


def qualified_name(parser: ASMParser) -> str:
    return parser.expect_token(_QUALIFIED, "qualified name")


def _import_line(parser: ASMParser) -> str:
    """An import line, with its dialect brought into scope. Returns the dialect's name."""
    parser._skip_all()
    if parser.expect_token(_IDENT, "keyword") != "import":
        raise parser.error("Expected 'import'", at=parser.token_start)
    dialect_name = parser.expect_token(_QUALIFIED, "dialect name")
    try:
        parser.scope.import_dialect(Dialect.get(dialect_name))
    except KeyError:
        raise parser.fatal(
            f"Unknown dialect: {dialect_name}", at=parser.token_start
        ) from None
    newline(parser)
    return dialect_name


def newline(parser: ASMParser) -> int:
    """Advance past current line and blank lines. Return indent of next content line."""
    # Skip to end of current line if not already at line start
    if (
        parser.pos > 0
        and parser.pos < len(parser.text)
        and parser.text[parser.pos - 1] != "\n"
    ):
        end = parser.text.find("\n", parser.pos)
        parser.pos = len(parser.text) if end == -1 else end + 1
    # Skip blank lines, return indent of first content line
    while parser.pos < len(parser.text):
        indent = 0
        while (
            parser.pos + indent < len(parser.text)
            and parser.text[parser.pos + indent] in " \t"
        ):
            indent += 1
        if (
            parser.pos + indent >= len(parser.text)
            or parser.text[parser.pos + indent] == "\n"
        ):
            parser.pos += indent + (1 if parser.pos + indent < len(parser.text) else 0)
            continue
        return indent
    return 0


def _dict_entry(parser: ASMParser) -> tuple[str, object]:
    key = json.loads(parser.expect_token(_STRING, "key"))
    parser.read(":")
    return key, value_expression(parser)


def value_expression(parser: ASMParser) -> object:
    """Universal value reader: SSA ref, Nil, list, dict, string, number, or named type."""
    if (name := parser.try_read(ssa_name)) is not None:
        return parser.resolve(name)
    if parser.try_read("()") is not None:
        return builtin.Nil()
    if parser.try_read("[") is not None:
        items = parser.read_list(value_expression)
        parser.read("]")
        return items
    if parser.try_read("{") is not None:
        entries = parser.read_list(_dict_entry)
        parser.read("}")
        return dict(entries)
    if (string := parser.parse_token(_STRING)) is not None:
        return json.loads(string)
    if (number := parser.parse_token(_NUMBER)) is not None:
        return float(number) if "." in number else int(number)
    type_val = _named_type(parser)
    if parser.try_read("(") is not None:
        raw = value_expression(parser)
        parser.read(")")
        return type_val.constant(raw)
    return type_val


def _named_type(parser: ASMParser) -> Type:
    name = parser.read(qualified_name)
    start = parser.token_start
    if not issubclass(type_cls := parser.lookup(name), Type):
        raise parser.fatal(f"Expected a type, got op {name}", at=start)
    if not type_cls.__params__ or parser.try_read("<") is None:
        return type_cls()
    values = parser.read_list(value_expression)
    parser.read(">")
    _check_count(parser, name, "parameters", type_cls.__params__, values, at=start)
    try:
        return type_cls(
            **{
                param_name: _coerce(parser, value, param_type)
                for (param_name, param_type), value in zip(type_cls.__params__, values)
            }
        )
    except TypeError as error:
        raise parser.fatal(f"{name}: {error}", at=start) from error


def _check_count(
    parser: ASMParser,
    name: str,
    kind: str,
    fields: dgen.type.Fields,
    given: list[object],
    *,
    at: int,
) -> None:
    if len(given) > len(fields):
        raise parser.fatal(
            f"{name} takes {len(fields)} {kind}, got {len(given)}", at=at
        )


def op_expression(
    parser: ASMParser, name: str
) -> tuple[type[Op], list[object], list[object], list[Block]]:
    op_name = parser.read(qualified_name)
    start = parser.token_start
    if not issubclass(op_cls := parser.lookup(op_name), Op):
        raise parser.fatal(f"Expected an op, got type {op_name}", at=start)
    parameters: list[object] = []
    if op_cls.__params__:
        parser.read("<")
        parameters = parser.read_list(value_expression)
        parser.read(">")
    parser.read("(")
    operands = parser.read_list(value_expression)
    parser.read(")")
    blocks: list[Block] = []
    for block_name in op_cls.__blocks__:
        saved = parser.pos
        parser._skip_all()
        if parser.parse_token(_IDENT) != block_name:
            parser.pos = saved
            break
        blocks.append(_read_block_body(parser, f"the {block_name} of %{name}"))
    return op_cls, parameters, operands, blocks


def op_statement(parser: ASMParser) -> Op:
    name = parser.read(ssa_name)
    start = parser.token_start
    pre_type = value_expression(parser) if parser.try_read(":") is not None else None
    parser.read("=")
    if pre_type is not None and parser.peek() in _LITERAL_START:
        op = ConstantOp(name=name, value=value_expression(parser), type=pre_type)
        parser.define(name, op)
        return op
    op_cls, parameters, operands, blocks = op_expression(parser, name)
    if issubclass(op_cls, ConstantOp):
        assert len(operands) == 1
        op = ConstantOp(name=name, value=operands[0], type=pre_type)
        parser.define(name, op)
        return op
    op_name = op_cls.dialect.qualified_name(op_cls.asm_name)
    _check_count(parser, op_name, "parameters", op_cls.__params__, parameters, at=start)
    _check_count(parser, op_name, "operands", op_cls.__operands__, operands, at=start)
    kwargs: dict[str, object] = {"name": name}
    if pre_type is not None:
        kwargs["type"] = pre_type
    annotation = pre_type if isinstance(pre_type, Value) else None
    try:
        for (param_name, param_type), value in zip(op_cls.__params__, parameters):
            kwargs[param_name] = _coerce(parser, value, param_type)
        for (field_name, field_type), value in zip(op_cls.__operands__, operands):
            kwargs[field_name] = _coerce(
                parser, value, literal_type(field_type, annotation)
            )
        for block_name, block in zip(op_cls.__blocks__, blocks):
            kwargs[block_name] = block
        op = op_cls(**kwargs)
    except TypeError as error:
        raise parser.fatal(f"%{name}: {error}", at=start) from error
    parser.define(name, op)
    return op


def _block_argument(parser: ASMParser) -> BlockArgument:
    name = parser.read(ssa_name)
    parser.read(":")
    arg = BlockArgument(name=name, type=value_expression(parser))
    parser.define(name, arg)
    return arg


def _block_parameter(parser: ASMParser) -> BlockParameter:
    name = parser.read(ssa_name)
    parser.read(":")
    param = BlockParameter(name=name, type=value_expression(parser))
    parser.define(name, param)
    return param


def block_arguments(parser: ASMParser) -> list[BlockArgument]:
    parser.read("(")
    args = parser.read_list(_block_argument)
    parser.read(")")
    return args


# -- Coercion ---------------------------------------------------------------


def _as_constant(field_type: type[Type], raw: object) -> Constant:
    """Wrap a raw literal as a Constant of *field_type*.

    Raises if *field_type* is parameterized (can't infer params from a literal).
    """
    if field_type.__params__:
        raise TypeError(
            f"Cannot use a bare literal for parameterized type {field_type.asm_name}; "
            f"use {field_type.asm_name}<...>({raw!r}) to specify type parameters explicitly"
        )
    return field_type().constant(raw)


def _coerce(
    parser: ASMParser,
    value: object,
    field_type: type[Type],
) -> object:
    """Coerce a parsed value to match a declared field type.

    - ``Value`` → pass through
    - ``list`` → ``PackOp`` (elements coerced individually)
    - other scalars → ``Constant`` via ``_as_constant``
    """
    if isinstance(value, Value):
        return value
    if isinstance(value, list):
        return _pack_list(parser, value, field_type)
    return _as_constant(field_type, value)


def _pack_list(
    parser: ASMParser, elems: list[object], field_type: type[Type]
) -> PackOp:
    if field_type is builtin.Span or issubclass(field_type, builtin.Span):
        # Span field: infer element type from first Value element.
        element_type: Type | None = None
        for elem in elems:
            if isinstance(elem, Value):
                element_type = elem.type
                break
        if element_type is None:
            element_type = dgen.type.TypeType()
    elif field_type.__params__:
        # Parameterized non-Span type — can't pack bare literals.
        _as_constant(field_type, elems)  # always raises
        raise AssertionError("unreachable")
    else:
        element_type = field_type()
    values: list[Value] = []
    for elem in elems:
        if isinstance(elem, Value):
            values.append(elem)
        else:
            values.append(ConstantOp(value=elem, type=element_type))
    return PackOp(values=values, type=builtin.Span(pointee=element_type))


# -- Block parsing ----------------------------------------------------------


def _read_block_body(parser: ASMParser, label: str) -> Block:
    # The header sees the enclosing scope and the inputs it has declared so far.
    outer = parser.scopes[-1]
    parser.scopes.append(BlockScope(outer.label, dict(outer.names)))
    block_params: list[BlockParameter] = []
    if parser.try_read("<") is not None:
        block_params = parser.read_list(_block_parameter)
        parser.read(">")
    args = block_arguments(parser)
    # Optional captures: captures(%name, ...)
    captured: dict[str, Value] = {}
    if parser.try_read("captures") is not None:
        parser.read("(")
        while parser.try_read(")") is None:
            if captured:
                parser.read(",")
            name = parser.expect_token(_SSA, "capture reference")[1:]
            captured[name] = parser.resolve(name)
    parser.read(":")
    parser.scopes[-1] = BlockScope(
        label,
        {
            value.name: value
            for value in [*block_params, *args]
            if value.name is not None
        }
        | captured,
    )
    block_indent = newline(parser)
    if block_indent <= parser.block_indent:
        raise parser.error("Empty block body")
    saved_indent = parser.block_indent
    parser.block_indent = block_indent
    last_op: Op | None = None
    while parser.pos < len(parser.text):
        indent = newline(parser)
        if indent < block_indent:
            break
        parser.pos += indent
        last_op = op_statement(parser)
    parser.block_indent = saved_indent
    parser.scopes.pop()
    assert last_op is not None
    return Block(
        result=last_op,
        args=args,
        parameters=block_params,
        captures=list(captured.values()),
    )
