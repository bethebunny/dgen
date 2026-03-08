"""Dialect-agnostic IR text deserialization.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.  Dialect knowledge is discovered from import headers
at the top of the IR text.
"""

from __future__ import annotations

import dataclasses
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


def _wrap_constant(field_type: type[Type], raw_value: object) -> Constant:
    """Wrap a raw Python value as a Constant of the given field type.

    For non-parameterized types, constructs field_type() directly.
    For parameterized types, derives parameters from the raw value.
    """
    if field_type.__params__:
        kwargs: dict[str, object] = {}
        for param_name, param_type in field_type.__params__:
            assert isinstance(raw_value, list)
            kwargs[param_name] = param_type().constant(len(raw_value))
        return field_type(**kwargs).constant(raw_value)
    return field_type().constant(raw_value)


def parse_module(text: str) -> Module:
    parser = ASMParser(text)
    while parser.try_read(ImportStatement) is not None:
        pass
    functions: list[builtin.FunctionOp] = []
    while not parser.done:
        op = parser.read(OpStatement)
        assert isinstance(op, builtin.FunctionOp)
        functions.append(op)
    return Module(functions=functions)


def parse_value(text: str, type: Type) -> Value:
    """Parse a typed value from ASM text. Used by Memory.from_asm."""
    parser = ASMParser(text)
    return _read_typed_value(parser, type.__class__)


# ===----------------------------------------------------------------------=== #
# New ASMParser with read/try_read protocol
# ===----------------------------------------------------------------------=== #


class Namespace:
    """Holds op/type registries populated from dialects."""

    def __init__(self) -> None:
        self.ops: dict[str, type[Op]] = {}
        self.types: dict[str, type[Type]] = {}

        # Implicit: builtin dialect (unqualified names)
        builtin_dialect = Dialect.get("builtin")
        self.ops.update(builtin_dialect.ops)
        self.types.update(builtin_dialect.types)

    def import_dialect(self, name: str) -> None:
        """Import a dialect module and register its ops/types with qualified names."""
        for prefix in ("dgen.dialects", "toy.dialects"):
            try:
                importlib.import_module(f"{prefix}.{name}")
                break
            except ModuleNotFoundError:
                continue

        try:
            dialect = Dialect.get(name)
        except KeyError:
            raise RuntimeError(f"Unknown dialect: {name}") from None
        for op_name, cls in dialect.ops.items():
            self.ops[f"{name}.{op_name}"] = cls
        for type_name, tcls in dialect.types.items():
            self.types[f"{name}.{type_name}"] = tcls


class ASMParser:
    """Low-level tokenizer with grammar-class dispatch via read/try_read."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos: int = 0
        self.namespace: Namespace = Namespace()
        self.name_table: dict[str, Value] = {}

    @property
    def done(self) -> bool:
        """Skip whitespace and newlines, return True if at end."""
        self._skip_whitespace_and_newlines()
        return self.pos >= len(self.text)

    def peek(self) -> str:
        """Return current char or empty string if at end."""
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def read(self, grammar: str | type) -> Any:  # noqa: ANN401
        """Read a grammar element.

        If grammar is a str, consume it as punctuation (skip whitespace first).
        If grammar is a class, call grammar.read(self).
        """
        if isinstance(grammar, str):
            return self._read_punct(grammar)
        return grammar.read(self)

    def try_read(self, grammar: str | type) -> Any | None:  # noqa: ANN401
        """Try to read a grammar element; restore position on failure."""
        saved = self.pos
        try:
            return self.read(grammar)
        except RuntimeError:
            self.pos = saved
            return None

    # ===------------------------------------------------------------------=== #
    # Low-level token methods
    # ===------------------------------------------------------------------=== #

    def _skip_whitespace(self) -> None:
        """Skip spaces and tabs (not newlines)."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def _skip_whitespace_and_newlines(self) -> None:
        """Skip spaces, tabs, and newlines."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _skip_line(self) -> None:
        """Skip to the next line."""
        while self.pos < len(self.text) and self.text[self.pos] != "\n":
            self.pos += 1
        if self.pos < len(self.text):
            self.pos += 1  # skip \n

    def _expect(self, expected: str) -> None:
        """Consume exact characters without skipping whitespace."""
        for ch in expected:
            if self.pos >= len(self.text) or self.text[self.pos] != ch:
                raise RuntimeError(f"Expected '{expected}' at position {self.pos}")
            self.pos += 1

    def _read_punct(self, punct: str) -> str:
        """Skip whitespace, then match punctuation string. Raises on failure."""
        self._skip_whitespace()
        for ch in punct:
            if self.pos >= len(self.text) or self.text[self.pos] != ch:
                raise RuntimeError(f"Expected '{punct}' at position {self.pos}")
            self.pos += 1
        return punct

    def parse_token(self, regex: re.Pattern[str]) -> str | None:
        """Try to match a regex token after skipping whitespace. Returns None on failure."""
        self._skip_whitespace()
        if match := regex.match(self.text, pos=self.pos):
            self.pos = match.end()
            return match.group()
        return None

    def expect_token(self, regex: re.Pattern[str], token_name: str) -> str:
        """Match a regex token after skipping whitespace. Raises on failure."""
        token = self.parse_token(regex)
        if token is None:
            raise RuntimeError(f"Expected {token_name} at position {self.pos}")
        return token

    def _parse_string_literal(self) -> str:
        """Parse a double-quoted string literal, return the contents."""
        self._skip_whitespace()
        match = _STRING.match(self.text, pos=self.pos)
        if match is None:
            raise RuntimeError(f"Expected string literal at position {self.pos}")
        self.pos = match.end()
        return match.group()[1:-1]

    def _parse_identifier(self) -> str:
        """Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*"""
        return self.expect_token(_IDENT, "identifier")


# ===----------------------------------------------------------------------=== #
# Grammar classes for ASMParser
# ===----------------------------------------------------------------------=== #


class PartialOp:
    """Intermediate result from OpExpression — carries parsed kwargs before name/type."""

    def __init__(self, cls: type[Op], kwargs: dict[str, object]) -> None:
        self.cls = cls
        self.kwargs = kwargs

    def build(self, name: str | None, type: Value[TypeType] | None) -> Op:
        """Finalize the op by injecting name and pre-annotated type."""
        self.kwargs["name"] = name
        if type is not None:
            self.kwargs["type"] = type
        return self.cls(**self.kwargs)


class SSAName:
    """Parse %foo, return 'foo'."""

    @classmethod
    def read(cls, parser: ASMParser) -> str:
        token = parser.expect_token(_SSA, "SSA name")
        return token[1:]


class QualifiedName:
    """Parse 'toy.transpose' or 'Index', return str."""

    @classmethod
    def read(cls, parser: ASMParser) -> str:
        return parser.expect_token(_QUALIFIED, "qualified name")


class ImportStatement:
    """Parse 'import toy\\n' or 'from builtin import ...\\n'.

    Side-effects: calls parser.namespace.import_dialect(name).
    """

    @classmethod
    def read(cls, parser: ASMParser) -> str:
        parser._skip_whitespace_and_newlines()
        word = parser._parse_identifier()

        if word == "from":
            mod_name = parser._parse_identifier()
            if mod_name != "builtin":
                raise RuntimeError(f"Expected 'builtin' after 'from', got '{mod_name}'")
            import_kw = parser._parse_identifier()
            if import_kw != "import":
                raise RuntimeError(
                    f"Expected 'import' after 'from builtin', got '{import_kw}'"
                )
            parser._skip_line()
            return "builtin"

        if word == "import":
            dialect_name = parser._parse_identifier()
            parser._skip_line()
            parser.namespace.import_dialect(dialect_name)
            return dialect_name

        raise RuntimeError(f"Expected 'import' or 'from', got '{word}'")


def _parse_raw_expr(parser: ASMParser) -> object:
    """Parse a single expression, dispatching on syntax.

    Returns raw Python values (int, float, str, list, dict), Type instances,
    or Value (SSA ref). This is the new-parser equivalent of parse_expr.
    """
    parser._skip_whitespace()
    c = parser.peek()

    if c == "(":
        parser._expect("()")
        return builtin.Nil()

    if c == "[":
        parser.read("[")
        items: list[object] = []
        if parser.try_read("]") is None:
            items.append(_parse_raw_expr(parser))
            while parser.try_read(",") is not None:
                items.append(_parse_raw_expr(parser))
            parser.read("]")
        return items

    if c == "{":
        parser.read("{")
        result: dict[str, object] = {}
        if parser.try_read("}") is None:
            key = _parse_raw_expr(parser)
            parser.read(":")
            val = _parse_raw_expr(parser)
            assert isinstance(key, str)
            result[key] = val
            while parser.try_read(",") is not None:
                key = _parse_raw_expr(parser)
                parser.read(":")
                val = _parse_raw_expr(parser)
                assert isinstance(key, str)
                result[key] = val
            parser.read("}")
        return result

    if c == "%":
        ssa_name = parser.read(SSAName)
        return _resolve_or_create_new(parser, ssa_name)

    if c == '"':
        return parser._parse_string_literal()

    if c in "-0123456789":
        token = parser.expect_token(_NUMBER, "number")
        return float(token) if "." in token else int(token)

    # Identifier -> type reference or qualified name
    name = parser.read(QualifiedName)
    type_cls = parser.namespace.types.get(name)
    if type_cls is None:
        raise RuntimeError(f"Unknown name: {name}")
    if dataclasses.is_dataclass(type_cls):
        fields = dataclasses.fields(type_cls)
    else:
        fields = ()
    if not fields:
        return type_cls()
    all_have_defaults = all(
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
        for f in fields
    )
    if all_have_defaults and parser.peek() != "<":
        return type_cls()
    parser.read("<")
    kwargs: dict[str, object] = {}
    for i, (f_name, f_type) in enumerate(type_cls.__params__):
        if i > 0:
            parser.read(",")
        raw_value = _parse_raw_expr(parser)
        if not isinstance(raw_value, Value):
            raw_value = _wrap_constant(f_type, raw_value)
        kwargs[f_name] = raw_value
    parser.read(">")
    return type_cls(**kwargs)


def _resolve_or_create_new(parser: ASMParser, ssa_name: str) -> Value:
    """Resolve an SSA name to a Value, creating a forward reference if needed."""
    if ssa_name not in parser.name_table:
        val = Value(name=ssa_name, type=builtin.Nil())
        parser.name_table[ssa_name] = val
        return val
    return parser.name_table[ssa_name]


class TypeExpression:
    """Parse type expressions, return Type | Value[TypeType].

    Handles: %t, {dict}, (), Index, Array<Index, 4>, toy.Tensor<[2, 3], F64>, etc.
    """

    @classmethod
    def read(cls, parser: ASMParser) -> Type | Value[TypeType]:
        result = _parse_raw_expr(parser)
        if isinstance(result, Type):
            return result
        if isinstance(result, Value) and isinstance(result.type, TypeType):
            return result
        # Dict in type position → raw dict for TypeType constants
        if isinstance(result, dict):
            return result  # type: ignore[return-value]
        raise RuntimeError(f"Expected type expression, got {result}")


def _read_typed_value(parser: ASMParser, expected_type: type[Type]) -> object:
    """Read a value when expected type is known.

    Returns Value, raw Python value (for ConstantOp), or list.
    """
    parser._skip_whitespace()
    c = parser.peek()

    if c == "%":
        ssa_name = parser.read(SSAName)
        return _resolve_or_create_new(parser, ssa_name)

    if c == "(":
        parser._expect("()")
        return builtin.Nil()

    if c == "[":
        return _parse_raw_expr(parser)

    if c == '"':
        return parser._parse_string_literal()

    if c == "{":
        return _parse_raw_expr(parser)

    if c in "-0123456789":
        token = parser.expect_token(_NUMBER, "number")
        return float(token) if "." in token else int(token)

    # Identifier → type expression
    return _parse_raw_expr(parser)


def _expand_list_sugar_new(
    parser: ASMParser,
    elements: list[object],
    element_type_cls: type[Type],
    pending_ops: list[Op],
) -> Value:
    """Expand [expr, expr, ...] into a PackOp.

    Non-Value elements (raw ints, floats) are wrapped as ConstantOps.
    """
    element_type = element_type_cls()
    list_type = builtin.List(element_type=element_type)
    values: list[Value] = []
    for v in elements:
        if isinstance(v, Value):
            values.append(v)
        else:
            const_op = ConstantOp(value=v, type=element_type)
            pending_ops.append(const_op)
            values.append(const_op)
    pack_op = builtin.PackOp(values=values, type=list_type)
    pending_ops.append(pack_op)
    return pack_op


class BlockArgs:
    """Parse (%x: F64, %y: Index), return list[BlockArgument]."""

    @classmethod
    def read(cls, parser: ASMParser) -> list[BlockArgument]:
        parser.read("(")
        args: list[BlockArgument] = []
        if parser.try_read(")") is not None:
            return args
        args.append(cls._parse_one(parser))
        while parser.try_read(",") is not None:
            args.append(cls._parse_one(parser))
        parser.read(")")
        return args

    @classmethod
    def _parse_one(cls, parser: ASMParser) -> BlockArgument:
        param_name = parser.read(SSAName)
        parser.read(":")
        type_ = TypeExpression.read(parser)
        arg = BlockArgument(name=param_name, type=type_)
        parser.name_table[param_name] = arg
        return arg


class OpExpression:
    """Parse dialect.op<params>(operands) blocks, return PartialOp."""

    @classmethod
    def read(cls, parser: ASMParser) -> PartialOp:
        return cls._read_with_pending(parser)[0]

    @classmethod
    def _read_with_pending(cls, parser: ASMParser) -> tuple[PartialOp, list[Op]]:
        op_name = parser.read(QualifiedName)
        op_cls = parser.namespace.ops.get(op_name)
        if op_cls is None:
            raise RuntimeError(f"Unknown op: {op_name}")

        kwargs: dict[str, object] = {}
        pending_ops: list[Op] = []

        # Parse constant params in <...>
        if op_cls.__params__:
            parser.read("<")
            for i, (f_name, f_type) in enumerate(op_cls.__params__):
                if i > 0:
                    parser.read(",")
                raw_value = _parse_raw_expr(parser)
                if not isinstance(raw_value, Value):
                    if isinstance(raw_value, list):
                        raw_value = [
                            _wrap_constant(f_type, v) if not isinstance(v, Value) else v
                            for v in raw_value
                        ]
                    else:
                        raw_value = _wrap_constant(f_type, raw_value)
                kwargs[f_name] = raw_value
            parser.read(">")

        # Parse runtime operand fields in (...)
        parser.read("(")
        for i, (f_name, f_type) in enumerate(op_cls.__operands__):
            if i > 0:
                parser.read(",")
            raw_value = _parse_raw_expr(parser)
            if isinstance(raw_value, list) and any(
                isinstance(v, Value) for v in raw_value
            ):
                raw_value = _expand_list_sugar_new(
                    parser, list(raw_value), f_type, pending_ops
                )
            elif not isinstance(raw_value, (Value, list)) and not issubclass(
                op_cls, ConstantOp
            ):
                raw_value = _wrap_constant(f_type, raw_value)
            kwargs[f_name] = raw_value
        parser.read(")")

        # Parse blocks if the op has them
        if op_cls.__blocks__:
            for block_idx, block_name in enumerate(op_cls.__blocks__):
                if block_idx > 0:
                    # Subsequent blocks: expect keyword at parent indent
                    keyword = block_name.removesuffix("_body")
                    parser._skip_whitespace_and_newlines()
                    saved_pos = parser.pos
                    try:
                        word = parser._parse_identifier()
                    except RuntimeError:
                        parser.pos = saved_pos
                        break
                    if word != keyword:
                        parser.pos = saved_pos
                        break
                block_args = BlockArgs.read(parser)
                parser.read(":")
                ops = _parse_indented_block(parser)
                kwargs[block_name] = Block(ops=ops, args=block_args)

        return PartialOp(op_cls, kwargs), pending_ops


def _parse_indented_block(parser: ASMParser) -> list[Op]:
    """Parse an indented block after ':' (for ops with body)."""
    parser._skip_line()
    # Determine indent of first line
    start = parser.pos
    indent = 0
    while parser.pos < len(parser.text) and parser.text[parser.pos] in " \t":
        if parser.text[parser.pos] == " ":
            indent += 1
        else:
            indent += 4
        parser.pos += 1
    parser.pos = start
    if indent == 0:
        return []
    return _parse_block(parser, min_indent=indent)


def _parse_block(parser: ASMParser, min_indent: int) -> list[Op]:
    """Parse a block of indented ops."""
    ops: list[Op] = []
    while parser.pos < len(parser.text):
        line_start = parser.pos
        indent = 0
        while parser.pos < len(parser.text) and parser.text[parser.pos] in " \t":
            if parser.text[parser.pos] == " ":
                indent += 1
            else:
                indent += 4
            parser.pos += 1

        # Empty line
        if parser.pos >= len(parser.text) or parser.peek() == "\n":
            if parser.pos < len(parser.text):
                parser.pos += 1  # skip \n
            continue

        # Not enough indent -> end of block
        if indent < min_indent:
            parser.pos = line_start
            break

        # Find the newline ending this line so we can detect body-bearing ops
        eol = parser.text.find("\n", parser.pos)
        if eol == -1:
            eol = len(parser.text)

        op, pending = _parse_op_statement(parser)
        # Drain pending ops (e.g. list sugar ConstantOps)
        ops.extend(pending)
        ops.append(op)

        # If parse consumed past original newline (body-bearing op), don't skip
        if parser.pos <= eol:
            parser._skip_line()
    return ops


def _parse_op_statement(parser: ASMParser) -> tuple[Op, list[Op]]:
    """Parse %name [: Type] = expr, return (Op, pending_ops)."""
    op_name_str = parser.read(SSAName)
    pre_type: Value[TypeType] | None = None
    if parser.try_read(":") is not None:
        pre_type = TypeExpression.read(parser)
    parser.read("=")
    parser._skip_whitespace()

    # Implicit constant: value starts with '[', '{', digit/minus
    if parser.peek() in "{[-0123456789":
        value = _parse_raw_expr(parser)
        if pre_type is None:
            raise RuntimeError(f"constant %{op_name_str} missing type annotation")
        op = ConstantOp(
            name=op_name_str,
            value=value,
            type=pre_type,
        )
        parser.name_table[op_name_str] = op
        return op, []

    # Op expression
    partial, pending_ops = OpExpression._read_with_pending(parser)
    op = partial.build(name=op_name_str, type=pre_type)
    parser.name_table[op_name_str] = op
    return op, pending_ops


class OpStatement:
    """Parse %name [: Type] = expr, return Op."""

    @classmethod
    def read(cls, parser: ASMParser) -> Op:
        op, _pending = _parse_op_statement(parser)
        return op
