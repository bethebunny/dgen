"""IR text deserialization.

read/try_read dispatch to reader functions — like lisp reader macros.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import dgen.type
from dgen import Block, Constant, Dialect, Op, Type, Value
from dgen.block import BlockArgument
from dgen.graph import walk_ops
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module, PackOp


class ParseError(RuntimeError):
    """Recoverable parse failure — caught by try_read for backtracking."""


_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_QUALIFIED = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
_STRING = re.compile(r'"[^"]*"')
_SSA = re.compile(r"%[a-zA-Z0-9_]+")
_NUMBER = re.compile(r"-?\d+\.\d*|-?\d*\.\d+|-?\d+")
_LITERAL_START = set('-0123456789{["(')


def parse_module(text: str) -> Module:
    parser = ASMParser(text)
    while parser.try_read(_import_line) is not None:
        pass
    functions: list[builtin.FunctionOp] = []
    while not parser.done:
        functions.append(parser.read(op_statement))
    return Module(functions=functions)


class ASMParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos: int = 0
        self.name_table: dict[str, Value] = {}
        self.pending_ops: list[Op] = []

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
            self.name_table[name] = Value(name=name, type=builtin.Nil())
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


# -- Lookup ----------------------------------------------------------------


def _lookup_op(name: str) -> type[Op]:
    """Look up an op by name, splitting on '.' for qualified names."""
    dialect, local = _resolve_name(name)
    op_cls = dialect.ops.get(local)
    if op_cls is None:
        raise ParseError(f"Unknown op: {name}")
    return op_cls


def _lookup_type(name: str) -> type[Type]:
    """Look up a type by name, splitting on '.' for qualified names."""
    dialect, local = _resolve_name(name)
    type_cls = dialect.types.get(local)
    if type_cls is None:
        raise ParseError(f"Unknown type: {name}")
    return type_cls


def _resolve_name(name: str) -> tuple[Dialect, str]:
    """Split 'dialect.name' into (Dialect, local_name). Unqualified → builtin."""
    if "." in name:
        dialect_name, local = name.split(".", 1)
        try:
            return Dialect.get(dialect_name), local
        except KeyError:
            raise ParseError(f"Unknown dialect: {dialect_name}") from None
    return Dialect.get("builtin"), name


# -- Reader functions -------------------------------------------------------


def ssa_name(parser: ASMParser) -> str:
    return parser.expect_token(_SSA, "SSA name")[1:]


def qualified_name(parser: ASMParser) -> str:
    return parser.expect_token(_QUALIFIED, "qualified name")


def _import_line(parser: ASMParser) -> str:
    """Skip an import line (dialects are registered externally). Returns dialect name."""
    parser._skip_all()
    keyword = parser.expect_token(_IDENT, "keyword")
    if keyword != "import":
        raise ParseError(f"Expected 'import', got '{keyword}'")
    dialect_name = parser.expect_token(_IDENT, "dialect name")
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
    type_cls = _lookup_type(name)
    if not type_cls.__params__ or parser.try_read("<") is None:
        return type_cls()
    values = parser.read_list(value_expression)
    parser.read(">")
    kwargs = {}
    for (param_name, param_type), value in zip(type_cls.__params__, values):
        kwargs[param_name] = _coerce_param(value, param_type)
    return type_cls(**kwargs)


def op_expression(
    parser: ASMParser,
) -> tuple[type[Op], list[object], list[object], list[Block]]:
    name = parser.read(qualified_name)
    op_cls = _lookup_op(name)
    parameters: list[object] = []
    if op_cls.__params__:
        parser.read("<")
        parameters = parser.read_list(value_expression)
        parser.read(">")
    parser.read("(")
    operands = parser.read_list(value_expression)
    parser.read(")")
    blocks: list[Block] = []
    for index, block_name in enumerate(op_cls.__blocks__):
        if index:
            saved = parser.pos
            parser._skip_all()
            if parser.parse_token(_IDENT) != block_name.removesuffix("_body"):
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
    kwargs: dict[str, object] = {"name": name}
    if pre_type is not None:
        kwargs["type"] = pre_type
    for (param_name, param_type), value in zip(op_cls.__params__, parameters):
        kwargs[param_name] = _coerce_param(value, param_type)
    for (field_name, field_type), value in zip(op_cls.__operands__, operands):
        kwargs[field_name] = _coerce_operand(parser, value, field_type, op_cls)
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


def block_arguments(parser: ASMParser) -> list[BlockArgument]:
    parser.read("(")
    args = parser.read_list(_block_argument)
    parser.read(")")
    return args


# -- Coercion ---------------------------------------------------------------


def _coerce_param(value: object, field_type: type[Type]) -> object:
    """Wrap a parsed value to match an expected param type."""
    if isinstance(value, Value):
        return value
    if isinstance(value, list) and (
        not field_type.__params__ or any(isinstance(v, Value) for v in value)
    ):
        return [_coerce_param(v, field_type) for v in value]
    return _wrap_constant(field_type, value)


def _coerce_operand(
    parser: ASMParser, value: object, field_type: type[Type], op_cls: type[Op]
) -> object:
    """Wrap a parsed operand: PackOp for mixed lists, Constant for raw scalars."""
    if isinstance(value, Value):
        return value
    if isinstance(value, list):
        if any(isinstance(v, Value) for v in value) or issubclass(
            field_type, builtin.List
        ):
            return _pack_list(parser, value, field_type)
        return value
    if issubclass(op_cls, ConstantOp):
        return value
    return _wrap_constant(field_type, value)


def _wrap_constant(field_type: type[Type], raw: object) -> Constant:
    if field_type.__params__:
        raise RuntimeError(
            f"Cannot use a bare literal for parameterized type {field_type.asm_name}; "
            f"use {field_type.asm_name}<...>({raw!r}) to specify type parameters explicitly"
        )
    return field_type().constant(raw)


def _pack_list(
    parser: ASMParser, elems: list[object], field_type: type[Type]
) -> PackOp:
    if field_type is builtin.List or field_type.__params__:
        # List is parameterized; infer element type from first Value element
        element_type: Type | None = None
        for elem in elems:
            if isinstance(elem, Value):
                element_type = elem.type
                break
        if element_type is None:
            element_type = dgen.type.TypeType()
    else:
        element_type = field_type()
    values: list[Value] = []
    for elem in elems:
        if isinstance(elem, Value):
            values.append(elem)
        else:
            op = ConstantOp(value=elem, type=element_type)
            parser.pending_ops.append(op)
            values.append(op)
    pack = PackOp(values=values, type=builtin.List(element_type=element_type))
    parser.pending_ops.append(pack)
    return pack


# -- Block parsing ----------------------------------------------------------


def _read_block_body(parser: ASMParser) -> Block:
    args = block_arguments(parser)
    parser.read(":")
    block_indent = newline(parser)
    if block_indent == 0:
        from dgen.dialects.builtin import Nil

        return Block(result=dgen.Value(type=Nil()), args=args)
    ops: list[Op] = []
    while parser.pos < len(parser.text):
        indent = newline(parser)
        if indent < block_indent:
            break
        parser.pos += indent
        op = op_statement(parser)
        ops.extend(parser.pending_ops)
        parser.pending_ops.clear()
        ops.append(op)
    block = Block(result=ops[-1], args=args)
    live = set(walk_ops(block.result))
    dead = [op for op in ops if op not in live]
    if dead:
        names = [op.name or type(op).__name__ for op in dead]
        raise ParseError(f"Dead ops in block (not reachable from result): {names}")
    return block
