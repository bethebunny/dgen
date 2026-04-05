"""IR text deserialization.

read/try_read dispatch to reader functions — like lisp reader macros.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import reduce
from typing import Any

import dgen.type
from dgen import Block, Constant, Op, Type, Value
from dgen.dialect import Dialect
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import builtin
from dgen.module import ConstantOp, PackOp


class ParseError(RuntimeError):
    """Recoverable parse failure — caught by try_read for backtracking."""


_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_QUALIFIED = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
_STRING = re.compile(r'"[^"]*"')
_SSA = re.compile(r"%[a-zA-Z0-9_]+")
_NUMBER = re.compile(r"-?\d+\.\d*|-?\d*\.\d+|-?\d+")
_LITERAL_START = set('-0123456789{["(')


class Scope(dict[str, "Scope | type[Op] | type[Type]"]):
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

    def lookup(self, qualified_name: str) -> type[Op] | type[Type]:
        try:
            result = reduce(
                lambda scope, key: scope[key], qualified_name.split("."), self
            )
        except KeyError:
            raise ParseError(f"Unknown name: {qualified_name}") from None
        assert isinstance(result, type)
        return result


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
        if (name := parser.try_read(_import_line)) is not None:
            parser.scope.import_dialect(Dialect.get(name))
        else:
            value = parser.read(op_statement)
    if value is None:
        raise ParseError("parse: no statements in input")
    return value


class ASMParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos: int = 0
        self.name_table: dict[str, Value] = {}
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
        except ParseError:
            self.pos = saved
            return None

    def parse_token(self, regex: re.Pattern[str]) -> str | None:
        self._skip_ws()
        if match := regex.match(self.text, self.pos):
            self.pos = match.end()
            return match.group()
        return None

    def expect_token(self, regex: re.Pattern[str], name: str) -> str:
        if (token := self.parse_token(regex)) is not None:
            return token
        raise ParseError(f"Expected {name} at {self.pos}")

    def read_list(self, reader: Callable[..., Any]) -> list[Any]:
        """Read comma-separated items until reader fails."""
        items: list[Any] = []
        if (first := self.try_read(reader)) is not None:
            items.append(first)
            while self.try_read(",") is not None:
                items.append(self.read(reader))
        return items

    def resolve(self, name: str) -> Value:
        if name not in self.name_table:
            raise ParseError(f"Undefined reference: %{name}")
        return self.name_table[name]

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def _skip_all(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _expect(self, string: str) -> None:
        for ch in string:
            if self.pos >= len(self.text) or self.text[self.pos] != ch:
                raise ParseError(f"Expected '{string}' at {self.pos}")
            self.pos += 1


# -- Reader functions -------------------------------------------------------


def ssa_name(parser: ASMParser) -> str:
    return parser.expect_token(_SSA, "SSA name")[1:]


def qualified_name(parser: ASMParser) -> str:
    return parser.expect_token(_QUALIFIED, "qualified name")


def _import_line(parser: ASMParser) -> str:
    """Parse an import line. Returns dialect name (may be dotted)."""
    parser._skip_all()
    keyword = parser.expect_token(_IDENT, "keyword")
    if keyword != "import":
        raise ParseError(f"Expected 'import', got '{keyword}'")
    dialect_name = parser.expect_token(_QUALIFIED, "dialect name")
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
    key = parser.expect_token(_STRING, "key")[1:-1]
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
        return string[1:-1]
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
    type_cls = parser.scope.lookup(name)
    assert issubclass(type_cls, Type)
    if not type_cls.__params__ or parser.try_read("<") is None:
        return type_cls()
    values = parser.read_list(value_expression)
    parser.read(">")
    kwargs = {
        param_name: _coerce(parser, value, param_type)
        for (param_name, param_type), value in zip(type_cls.__params__, values)
    }
    return type_cls(**kwargs)


def op_expression(
    parser: ASMParser,
) -> tuple[type[Op], list[object], list[object], list[Block]]:
    name = parser.read(qualified_name)
    op_cls = parser.scope.lookup(name)
    assert issubclass(op_cls, Op)
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
        blocks.append(_read_block_body(parser))
    return op_cls, parameters, operands, blocks


def op_statement(parser: ASMParser) -> Op:
    name = parser.read(ssa_name)
    pre_type = value_expression(parser) if parser.try_read(":") is not None else None
    parser.read("=")
    if pre_type is not None and parser.peek() in _LITERAL_START:
        op = ConstantOp(name=name, value=value_expression(parser), type=pre_type)
        parser.name_table[name] = op
        return op
    op_cls, parameters, operands, blocks = op_expression(parser)
    if issubclass(op_cls, ConstantOp):
        assert len(operands) == 1
        op = ConstantOp(name=name, value=operands[0], type=pre_type)
        parser.name_table[name] = op
        return op
    kwargs: dict[str, object] = {"name": name}
    if pre_type is not None:
        kwargs["type"] = pre_type
    for (param_name, param_type), value in zip(op_cls.__params__, parameters):
        kwargs[param_name] = _coerce(parser, value, param_type)
    for (field_name, field_type), value in zip(op_cls.__operands__, operands):
        # For polymorphic ops (field_type is base Type), infer concrete type
        # from the explicit result type annotation when available.
        effective_type = field_type
        if field_type is Type and isinstance(pre_type, Type):
            effective_type = type(pre_type)
        kwargs[field_name] = _coerce(parser, value, effective_type)
    for block_name, block in zip(op_cls.__blocks__, blocks):
        kwargs[block_name] = block
    op = op_cls(**kwargs)
    parser.name_table[name] = op
    return op


def _block_argument(parser: ASMParser) -> BlockArgument:
    name = parser.read(ssa_name)
    parser.read(":")
    arg = BlockArgument(name=name, type=value_expression(parser))
    parser.name_table[name] = arg
    return arg


def _block_parameter(parser: ASMParser) -> BlockParameter:
    name = parser.read(ssa_name)
    parser.read(":")
    param = BlockParameter(name=name, type=value_expression(parser))
    parser.name_table[name] = param
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
        raise RuntimeError(
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


def _read_block_body(parser: ASMParser) -> Block:
    # Optional block parameters: <%name: Type, ...>
    block_params: list[BlockParameter] = []
    if parser.try_read("<") is not None:
        block_params = parser.read_list(_block_parameter)
        parser.read(">")
    args = block_arguments(parser)
    # Optional captures: captures(%name, ...)
    captures: list[dgen.Value] = []
    if parser.try_read("captures") is not None:
        parser.read("(")
        while parser.try_read(")") is None:
            if captures:
                parser.read(",")
            ssa = parser.expect_token(_SSA, "capture reference")
            captures.append(parser.resolve(ssa[1:]))  # strip leading %
    parser.read(":")
    block_indent = newline(parser)
    if block_indent <= parser.block_indent:
        raise ParseError("Empty block body")
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
    assert last_op is not None
    return Block(result=last_op, args=args, parameters=block_params, captures=captures)
