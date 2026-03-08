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


def _resolve_or_create(parser: IRParser, ssa_name: str) -> Value:
    """Resolve an SSA name to a Value, creating a forward reference if needed."""
    if ssa_name not in parser.name_table:
        val = Value(name=ssa_name, type=builtin.Nil())
        parser.name_table[ssa_name] = val
        return val
    return parser.name_table[ssa_name]


def parse_expr(parser: IRParser) -> object:
    """Parse a single expression, dispatching on syntax."""
    parser.skip_whitespace()
    c = parser.peek()

    if c == "(":
        # Nil value literal: ()
        parser.expect("()")
        return builtin.Nil()

    if c == "[":
        # List: [expr, expr, ...]
        parser.expect("[")
        items = []
        if not parser.parse_punct("]"):
            items.append(parse_expr(parser))
            while parser.parse_punct(","):
                items.append(parse_expr(parser))
            parser.expect_punct("]")
        return items

    if c == "{":
        # Dict: {key: value, key: value, ...}
        parser.expect("{")
        result: dict[str, object] = {}
        if not parser.parse_punct("}"):
            key = parse_expr(parser)
            parser.expect_punct(":")
            val = parse_expr(parser)
            assert isinstance(key, str)
            result[key] = val
            while parser.parse_punct(","):
                key = parse_expr(parser)
                parser.expect_punct(":")
                val = parse_expr(parser)
                assert isinstance(key, str)
                result[key] = val
            parser.expect_punct("}")
        return result

    if c == "%":
        # SSA reference
        return _resolve_or_create(parser, parser.parse_ssa_name())

    if c == '"':
        # String literal
        return parser.parse_string_literal()

    if c in "-0123456789":
        # Unified number parse: float if '.' present, else int
        token = parser.expect_token(_NUMBER, "number")
        return float(token) if "." in token else int(token)

    # Identifier -> type reference or qualified name
    name = parser.parse_qualified_name()
    cls = parser._types.get(name)
    if cls is None:
        raise RuntimeError(f"Unknown name: {name}")
    if dataclasses.is_dataclass(cls):
        fields = dataclasses.fields(cls)
    else:
        fields = ()
    if not fields:
        return cls()
    # If all fields have defaults and no '<' follows, use defaults
    all_have_defaults = all(
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
        for f in fields
    )
    if all_have_defaults and parser.peek() != "<":
        return cls()
    # Parameterized type: Name<expr, expr, ...>
    parser.expect_punct("<")
    kwargs = _parse_fields_from_exprs(parser, cls)
    parser.expect_punct(">")
    return cls(**kwargs)


def _wrap_constant(field_type: type[Type], raw_value: object) -> Constant:
    """Wrap a raw Python value as a Constant of the given field type.

    For non-parameterized types, constructs field_type() directly.
    For parameterized types, derives parameters from the raw value.
    """
    if field_type.__params__:
        # Build kwargs by wrapping each param as a constant from raw_value
        kwargs: dict[str, object] = {}
        for param_name, param_type in field_type.__params__:
            # Infer param value: for "rank"-like params on array-valued types,
            # the param is len(raw_value)
            assert isinstance(raw_value, list)
            kwargs[param_name] = param_type().constant(len(raw_value))
        return field_type(**kwargs).constant(raw_value)
    return field_type().constant(raw_value)


def _expand_list_sugar(
    parser: IRParser, elements: list[object], element_type_cls: type[Type]
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
            parser.pending_ops.append(const_op)
            values.append(const_op)
    pack_op = builtin.PackOp(values=values, type=list_type)
    parser.pending_ops.append(pack_op)
    return pack_op


def _parse_fields_from_exprs(parser: IRParser, cls: type[Type]) -> dict[str, object]:
    """Parse comma-separated exprs and map them to the type's declared fields.

    Iterates __params__. Raw Python values are wrapped via
    field_type().constant(); values that are already Value or Type
    instances are kept as-is.
    """
    kwargs = {}
    all_fields = cls.__params__
    for i, (name, field_type) in enumerate(all_fields):
        if i > 0:
            parser.expect_punct(",")
        raw_value = parse_expr(parser)
        if not isinstance(raw_value, Value):
            raw_value = _wrap_constant(field_type, raw_value)
        kwargs[name] = raw_value
    return kwargs


def parse_op_fields(
    parser: IRParser,
    cls: type[Op],
    name: str | None = None,
    pre_type: Value[TypeType] | None = None,
) -> Op:
    """Generic op field parser driven by field declarations."""
    kwargs: dict[str, Any] = {"name": name}  # noqa: ANN401

    # Parse constant fields in <...>
    if cls.__params__:
        parser.expect_punct("<")
        for i, (f_name, f_type) in enumerate(cls.__params__):
            if i > 0:
                parser.expect_punct(",")
            raw_value = parse_expr(parser)
            if not isinstance(raw_value, Value):
                if isinstance(raw_value, list):
                    raw_value = [
                        _wrap_constant(f_type, v) if not isinstance(v, Value) else v
                        for v in raw_value
                    ]
                else:
                    raw_value = _wrap_constant(f_type, raw_value)
            kwargs[f_name] = raw_value
        parser.expect_punct(">")

    # Parse runtime operand fields in (...)
    parser.expect_punct("(")
    for i, (f_name, f_type) in enumerate(cls.__operands__):
        if i > 0:
            parser.expect_punct(",")
        raw_value = parse_expr(parser)
        if isinstance(raw_value, list) and any(isinstance(v, Value) for v in raw_value):
            raw_value = _expand_list_sugar(parser, list(raw_value), f_type)
        elif not isinstance(raw_value, (Value, list)) and not issubclass(
            cls, ConstantOp
        ):
            raw_value = _wrap_constant(f_type, raw_value)
        kwargs[f_name] = raw_value
    parser.expect_punct(")")

    # Type annotation (already parsed before '=' if present)
    if pre_type is not None:
        kwargs["type"] = pre_type

    # Body (indented blocks)
    if cls.__blocks__:
        for block_idx, block_name in enumerate(cls.__blocks__):
            if block_idx > 0:
                # Subsequent blocks: expect keyword at parent indent level
                keyword = block_name.removesuffix("_body")
                parser.skip_whitespace_and_newlines()
                saved_pos = parser.pos
                try:
                    word = parser.parse_identifier()
                except RuntimeError:
                    parser.pos = saved_pos
                    break
                if word != keyword:
                    parser.pos = saved_pos
                    break
            args = []
            if parser.parse_punct("("):
                if not parser.parse_punct(")"):
                    args.append(parser._parse_param())
                    while parser.parse_punct(","):
                        args.append(parser._parse_param())
                    parser.expect_punct(")")
            parser.expect_punct(":")
            ops = parser.parse_indented_block()
            kwargs[block_name] = Block(ops=ops, args=args)

    op = cls(**kwargs)

    # Register op in name table if it has a name
    if name is not None:
        parser.name_table[name] = op

    return op


class IRParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self._ops: dict[str, type[Op]] = {}
        self._types: dict = {}
        self.name_table: dict[str, Value] = {}
        self.pending_ops: list[Op] = []

        # Implicit: from builtin import *
        builtin_dialect = Dialect.get("builtin")
        self._ops.update(builtin_dialect.ops)
        self._types.update(builtin_dialect.types)

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def peek(self) -> str:
        if self.at_end():
            return ""
        return self.text[self.pos]

    def advance(self) -> str:
        c = self.text[self.pos]
        self.pos += 1
        return c

    def skip_whitespace(self) -> None:
        while not self.at_end() and self.text[self.pos] in " \t":
            self.pos += 1

    def skip_whitespace_and_newlines(self) -> None:
        while not self.at_end() and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def expect(self, expected: str) -> None:
        for ch in expected:
            if self.at_end() or self.text[self.pos] != ch:
                raise RuntimeError(f"Expected '{expected}' at position {self.pos}")
            self.pos += 1

    def parse_punct(self, char: str) -> bool:
        """Skip whitespace, then consume char if present."""
        self.skip_whitespace()
        if not self.at_end() and self.text[self.pos] == char:
            self.pos += 1
            return True
        return False

    def expect_punct(self, char: str) -> None:
        """Skip whitespace, then expect a punctuation character."""
        self.skip_whitespace()
        if self.at_end() or self.text[self.pos] != char:
            raise RuntimeError(f"Expected '{char}' at position {self.pos}")
        self.pos += 1

    def parse_token(self, regex: re.Pattern[str]) -> str | None:
        self.skip_whitespace()
        if match := regex.match(self.text, pos=self.pos):
            self.pos = match.end()
            return match.group()
        return None

    def expect_token(self, expected: re.Pattern[str], token_name: str) -> str:
        token = self.parse_token(expected)
        if token is None:
            raise RuntimeError(
                f"Expected {token_name} ({expected!r}) at position {self.pos}"
            )
        return token

    def parse_identifier(self) -> str:
        """Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*"""
        return self.expect_token(_IDENT, "identifier")

    def parse_qualified_name(self) -> str:
        """Parse a possibly-qualified name: ident or ident.ident"""
        return self.expect_token(_QUALIFIED, "qualified name")

    def parse_string_literal(self) -> str:
        """Parse a double-quoted string literal, return the contents."""
        token = self.expect_token(_STRING, "string literal")
        return token[1:-1]

    def parse_ssa_name(self) -> str:
        """Parse %name, return the name part."""
        token = self.expect_token(_SSA, "SSA name")
        return token[1:]

    def parse_number(self) -> float:
        """Parse a floating point number."""
        return float(self.expect_token(_FLOAT, "number"))

    def parse_int(self) -> int:
        """Parse a non-negative integer."""
        return int(self.expect_token(_INT, "integer"))

    def parse_type(self) -> Type | Value:
        """Parse a type, or an SSA ref to a TypeType value."""
        result = parse_expr(self)
        if isinstance(result, Type):
            return result
        if isinstance(result, Value) and isinstance(result.type, TypeType):
            return result
        raise RuntimeError(f"Expected type, got {result}")

    def skip_line(self) -> None:
        """Skip to the next line."""
        while not self.at_end() and self.peek() != "\n":
            self.pos += 1
        if not self.at_end():
            self.pos += 1  # skip \n

    # ===------------------------------------------------------------------=== #
    # Import header parsing
    # ===------------------------------------------------------------------=== #

    def _parse_imports(self) -> None:
        """Parse import headers at the top of the module."""
        while not self.at_end():
            self.skip_whitespace_and_newlines()
            if self.at_end():
                break
            # Quick check: imports start with 'f' (from) or 'i' (import)
            if self.peek() not in ("f", "i"):
                break
            saved = self.pos
            word = self.parse_identifier()

            if word == "from":
                # from builtin import ... — no-op (builtin is implicit)
                mod_name = self.parse_identifier()
                if mod_name != "builtin":
                    raise RuntimeError(
                        f"Expected 'builtin' after 'from', got '{mod_name}'"
                    )
                import_kw = self.parse_identifier()
                if import_kw != "import":
                    raise RuntimeError(
                        f"Expected 'import' after 'from builtin', got '{import_kw}'"
                    )
                # Skip the rest of the line (name list)
                self.skip_line()

            elif word == "import":
                dialect_name = self.parse_identifier()
                self.skip_line()
                for _pfx in ("toy.dialects", "dgen.dialects"):
                    try:
                        importlib.import_module(f"{_pfx}.{dialect_name}")
                        break
                    except ModuleNotFoundError:
                        continue
                d = Dialect.get(dialect_name)
                for op_name, cls in d.ops.items():
                    self._ops[f"{dialect_name}.{op_name}"] = cls
                for tname, tcls in d.types.items():
                    self._types[f"{dialect_name}.{tname}"] = tcls
            else:
                # Not an import line — rewind
                self.pos = saved
                break

    # ===------------------------------------------------------------------=== #
    # Module / function / block parsing
    # ===------------------------------------------------------------------=== #

    def parse_module(self) -> Module:
        """Parse a complete module from IR text."""
        self._parse_imports()
        self.skip_whitespace_and_newlines()

        functions: list[builtin.FunctionOp] = []
        while not self.at_end():
            self.skip_whitespace_and_newlines()
            if self.at_end():
                break
            op = self.parse_op()
            assert isinstance(op, builtin.FunctionOp)
            functions.append(op)

        return Module(functions=functions)

    def _parse_param(self) -> BlockArgument:
        """Parse %name: Type"""
        param_name = self.parse_ssa_name()
        self.expect_punct(":")
        type_ = self.parse_type()
        arg = BlockArgument(name=param_name, type=type_)
        self.name_table[param_name] = arg
        return arg

    def _parse_block(self, min_indent: int) -> list[Op]:
        """Parse a block of indented ops."""
        ops: list[Op] = []
        while not self.at_end():
            line_start = self.pos
            indent = 0
            while not self.at_end() and self.text[self.pos] in " \t":
                if self.text[self.pos] == " ":
                    indent += 1
                else:
                    indent += 4
                self.pos += 1

            # Empty line
            if self.at_end() or self.peek() == "\n":
                if not self.at_end():
                    self.pos += 1  # skip \n
                continue

            # Not enough indent -> end of block
            if indent < min_indent:
                self.pos = line_start
                break

            # Find the newline ending this line (before parsing) so we can
            # detect whether parse_op consumed past it (body-bearing ops).
            eol = self.text.find("\n", self.pos)
            if eol == -1:
                eol = len(self.text)

            op = self.parse_op()
            # Drain pending ops created during parsing (e.g. list sugar ConstantOps)
            # — these must appear before the op that references them
            ops.extend(self.pending_ops)
            self.pending_ops.clear()
            ops.append(op)

            # If parse_op consumed past the original newline (body-bearing op),
            # the cursor is already at the next line's start — don't skip.
            if self.pos <= eol:
                self.skip_line()
        return ops

    def parse_indented_block(self) -> list[Op]:
        """Parse an indented block after a ':' (for ops with body)."""
        self.skip_line()
        # Determine indent of first line
        start = self.pos
        indent = 0
        while not self.at_end() and self.text[self.pos] in " \t":
            if self.text[self.pos] == " ":
                indent += 1
            else:
                indent += 4
            self.pos += 1
        self.pos = start
        if indent == 0:
            return []
        return self._parse_block(min_indent=indent)

    def parse_op(self) -> Op:
        """Parse a single operation: %result [: type] = [dialect.]op(...)"""
        op_name_str = self.parse_ssa_name()
        pre_type = None
        if self.parse_punct(":"):
            pre_type = self.parse_type()
        self.expect_punct("=")
        self.skip_whitespace()
        # Implicit constant: value starts with '[' or digit/minus
        if self.peek() in "{[-0123456789":
            value = parse_expr(self)
            if pre_type is None:
                raise RuntimeError(f"constant %{op_name_str} missing type annotation")
            op = ConstantOp(
                name=op_name_str,
                value=value,
                type=pre_type,
            )
            self.name_table[op_name_str] = op
            return op
        name = self.parse_qualified_name()
        cls = self._ops.get(name)
        if cls is None:
            raise RuntimeError(f"Unknown op: {name}")
        return parse_op_fields(self, cls, name=op_name_str, pre_type=pre_type)


def parse_module(text: str) -> Module:
    parser = IRParser(text)
    return parser.parse_module()


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
        else:
            raise RuntimeError(f"Unknown dialect: {name}")

        dialect = Dialect.get(name)
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

    def read_token(self) -> str:
        """Read an identifier token (skips whitespace first). Raises if none found."""
        self._skip_whitespace()
        match = _IDENT.match(self.text, pos=self.pos)
        if match is None:
            raise RuntimeError(f"Expected token at position {self.pos}")
        self.pos = match.end()
        return match.group()

    def expect_token(self, expected: str) -> str:
        """Read a token and verify it matches expected. Raises on mismatch."""
        token = self.read_token()
        if token != expected:
            raise RuntimeError(
                f"Expected '{expected}' at position {self.pos}, got '{token}'"
            )
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
        return self.read_token()
