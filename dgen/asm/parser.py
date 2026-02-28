"""Dialect-agnostic IR text deserialization.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.  Dialect knowledge is discovered from import headers
at the top of the IR text.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from dgen import Block, Dialect, Op, Type, Value
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.type import Memory


def _resolve_or_create(parser: IRParser, ssa_name: str) -> Value:
    """Resolve an SSA name to a Value, creating a forward reference if needed."""
    if ssa_name not in parser.name_table:
        val = Value(name=ssa_name, type=builtin.Nil())
        parser.name_table[ssa_name] = val
        return val
    return parser.name_table[ssa_name]


def parse_expr(parser: IRParser) -> object:
    """Parse a single expression, dispatching on syntax."""
    c = parser.peek()

    if c == "(":
        # Nil literal: ()
        from dgen.dialects.builtin import Nil

        parser.expect("()")
        return Nil()

    if c == "[":
        # List: [expr, expr, ...]
        parser.expect("[")
        parser.skip_whitespace()
        items = []
        if parser.peek() != "]":
            items.append(parse_expr(parser))
            parser.skip_whitespace()
            while parser.peek() == ",":
                parser.expect(",")
                parser.skip_whitespace()
                items.append(parse_expr(parser))
                parser.skip_whitespace()
        parser.expect("]")
        return items

    if c == "%":
        # SSA reference
        return _resolve_or_create(parser, parser.parse_ssa_name())

    if c == '"':
        # String literal
        return parser.parse_string_literal()

    if c in "-0123456789":
        # Number: peek for "." to distinguish int vs float
        p = parser.pos
        if p < len(parser.text) and parser.text[p] == "-":
            p += 1
        while p < len(parser.text) and parser.text[p].isdigit():
            p += 1
        if p < len(parser.text) and parser.text[p] == ".":
            return parser.parse_number()
        return parser.parse_int()

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
    # Parameterized type: Name<expr, expr, ...>
    parser.expect("<")
    parser.skip_whitespace()
    kwargs = _parse_fields_from_exprs(parser, cls)
    parser.expect(">")
    return cls(**kwargs)


def _expand_list_sugar(
    parser: IRParser, elements: list[object], element_type_cls: type[Type]
) -> Value:
    """Expand [expr, expr, ...] into a PackOp."""
    from dgen.dialects.builtin import IndexType, List, PackOp

    element_type = element_type_cls()
    list_type = List(
        element_type=element_type,
        count=IndexType().constant(len(elements)),
    )
    values = [v for v in elements if isinstance(v, Value)]
    pack_op = PackOp(values=values, type=list_type)
    parser.pending_ops.append(pack_op)
    return pack_op


def _parse_fields_from_exprs(parser: IRParser, cls: type[Type]) -> dict[str, object]:
    """Parse comma-separated exprs and map them to the type's declared fields.

    Iterates __params__. Raw Python values are wrapped via
    field_type.for_value().constant(); values that are already Value or Type
    instances are kept as-is.
    """
    kwargs = {}
    all_fields = cls.__params__
    for i, (name, field_type) in enumerate(all_fields):
        if i > 0:
            parser.expect(",")
            parser.skip_whitespace()
        raw_value = parse_expr(parser)
        if not isinstance(raw_value, (Value, Type)):
            raw_value = field_type.for_value(raw_value).constant(raw_value)
        kwargs[name] = raw_value
        parser.skip_whitespace()
    return kwargs


def parse_op_fields(
    parser: IRParser,
    cls: type[Op],
    name: str | None = None,
    pre_type: Type | None = None,
) -> Op:
    """Generic op field parser driven by field declarations."""
    kwargs: dict[str, Any] = {"name": name}  # noqa: ANN401

    # Parse constant fields in <...>
    if cls.__params__:
        parser.expect("<")
        parser.skip_whitespace()
        for i, (f_name, f_type) in enumerate(cls.__params__):
            if i > 0:
                parser.expect(",")
                parser.skip_whitespace()
            raw_value = parse_expr(parser)
            if not isinstance(raw_value, (Value, Type)):
                if isinstance(raw_value, list):
                    raw_value = [
                        f_type.for_value(v).constant(v)
                        if not isinstance(v, (Value, Type))
                        else v
                        for v in raw_value
                    ]
                else:
                    raw_value = f_type.for_value(raw_value).constant(raw_value)
            kwargs[f_name] = raw_value
            parser.skip_whitespace()
        parser.expect(">")

    # Parse runtime operand fields in (...)
    parser.expect("(")
    parser.skip_whitespace()
    for i, (f_name, f_type) in enumerate(cls.__operands__):
        if i > 0:
            parser.expect(",")
            parser.skip_whitespace()
        raw_value = parse_expr(parser)
        if isinstance(raw_value, list) and all(isinstance(v, Value) for v in raw_value):
            # [%ref, %ref, ...] sugar → list_new + list_set chain
            raw_value = _expand_list_sugar(parser, list(raw_value), f_type)
        kwargs[f_name] = raw_value
        parser.skip_whitespace()
    parser.expect(")")

    # Type annotation (already parsed before '=' if present)
    if pre_type is not None:
        kwargs["type"] = pre_type

    # Body (indented block)
    if cls.__blocks__:
        parser.skip_whitespace()
        # Parse optional block args: (%name: type, ...)
        args = []
        if parser.peek() == "(":
            parser.expect("(")
            parser.skip_whitespace()
            if parser.peek() != ")":
                args.append(parser._parse_param())
                parser.skip_whitespace()
                while parser.peek() == ",":
                    parser.expect(",")
                    parser.skip_whitespace()
                    args.append(parser._parse_param())
                    parser.skip_whitespace()
            parser.expect(")")
        parser.skip_whitespace()
        parser.expect(":")
        ops = parser.parse_indented_block()
        kwargs[cls.__blocks__[0]] = Block(ops=ops, args=args)

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

    def parse_identifier(self) -> str:
        """Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*"""
        start = self.pos
        if self.at_end():
            raise RuntimeError(f"Expected identifier at position {self.pos}")
        c = self.peek()
        if not (c.isalpha() or c == "_"):
            raise RuntimeError(f"Expected identifier at position {self.pos}")
        self.pos += 1
        while not self.at_end():
            c = self.peek()
            if c.isalnum() or c == "_":
                self.pos += 1
            else:
                break
        return self.text[start : self.pos]

    def parse_qualified_name(self) -> str:
        """Parse a possibly-qualified name: ident or ident.ident"""
        name = self.parse_identifier()
        if not self.at_end() and self.peek() == ".":
            self.advance()
            name += "." + self.parse_identifier()
        return name

    def parse_string_literal(self) -> str:
        """Parse a double-quoted string literal, return the contents."""
        self.expect('"')
        start = self.pos
        while not self.at_end() and self.peek() != '"':
            self.pos += 1
        result = self.text[start : self.pos]
        self.expect('"')
        return result

    def parse_ssa_name(self) -> str:
        """Parse %name, return the name part."""
        self.expect("%")
        return self._parse_name_chars()

    def _parse_name_chars(self) -> str:
        """Parse [a-zA-Z0-9_]+"""
        start = self.pos
        while not self.at_end():
            c = self.peek()
            if c.isalnum() or c == "_":
                self.pos += 1
            else:
                break
        if self.pos == start:
            raise RuntimeError(f"Expected name at position {self.pos}")
        return self.text[start : self.pos]

    def parse_number(self) -> float:
        """Parse a floating point number."""
        start = self.pos
        if not self.at_end() and self.peek() == "-":
            self.pos += 1
        while not self.at_end() and self.peek().isdigit():
            self.pos += 1
        if not self.at_end() and self.peek() == ".":
            self.pos += 1
            while not self.at_end() and self.peek().isdigit():
                self.pos += 1
        if self.pos == start:
            raise RuntimeError(f"Expected number at position {self.pos}")
        return float(self.text[start : self.pos])

    def parse_int(self) -> int:
        """Parse a non-negative integer."""
        start = self.pos
        while not self.at_end() and self.peek().isdigit():
            self.pos += 1
        if self.pos == start:
            raise RuntimeError(f"Expected integer at position {self.pos}")
        return int(self.text[start : self.pos])

    def parse_type(self) -> Type:
        """Parse a type via the registered type table, or () for Nil."""
        if self.peek() == "(":
            self.expect("()")
            return builtin.Nil()
        result = parse_expr(self)
        assert isinstance(result, Type)
        return result

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
                self.skip_whitespace()
                mod_name = self.parse_identifier()
                if mod_name != "builtin":
                    raise RuntimeError(
                        f"Expected 'builtin' after 'from', got '{mod_name}'"
                    )
                self.skip_whitespace()
                import_kw = self.parse_identifier()
                if import_kw != "import":
                    raise RuntimeError(
                        f"Expected 'import' after 'from builtin', got '{import_kw}'"
                    )
                # Skip the rest of the line (name list)
                self.skip_line()

            elif word == "import":
                self.skip_whitespace()
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

    def parse_module(self) -> builtin.Module:
        """Parse a complete module from IR text."""
        self._parse_imports()
        self.skip_whitespace_and_newlines()

        functions: list[builtin.FuncOp] = []
        while not self.at_end():
            self.skip_whitespace_and_newlines()
            if self.at_end():
                break
            functions.append(self.parse_func())

        return builtin.Module(functions=functions)

    def parse_func(self) -> builtin.FuncOp:
        """Parse a function definition."""
        func_name = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        self.expect("function")
        self.skip_whitespace()
        self.expect("(")

        # Parse parameters
        args: list[BlockArgument] = []
        self.skip_whitespace()
        if self.peek() != ")":
            arg = self._parse_param()
            args.append(arg)
            self.skip_whitespace()
            while self.peek() == ",":
                self.expect(",")
                self.skip_whitespace()
                arg = self._parse_param()
                args.append(arg)
                self.skip_whitespace()
        self.expect(")")

        # Return type
        self.skip_whitespace()
        self.expect("->")
        self.skip_whitespace()
        result_type: Type = self.parse_type()

        self.expect(":")
        self.skip_line()

        # Parse body (indented lines)
        ops = self._parse_block(min_indent=1)

        func_type = builtin.Function(result=result_type)
        func_op = builtin.FuncOp(
            name=func_name, type=func_type, body=Block(ops=ops, args=args)
        )
        self.name_table[func_name] = func_op
        return func_op

    def _parse_param(self) -> BlockArgument:
        """Parse %name or %name: Type"""
        param_name = self.parse_ssa_name()
        self.skip_whitespace()
        type_ = None
        if not self.at_end() and self.peek() == ":":
            self.expect(":")
            self.skip_whitespace()
            type_ = self.parse_type()
        if type_ is None:
            raise RuntimeError(f"parameter %{param_name} missing type annotation")
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

            # Drain any pending ops (from list sugar expansion)
            ops.extend(self.pending_ops)
            self.pending_ops.clear()
            ops.append(self.parse_op())

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
        self.skip_whitespace()
        # Optional type annotation before '='
        pre_type = None
        if self.peek() == ":":
            self.expect(":")
            self.skip_whitespace()
            pre_type = self.parse_type()
            self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        # Implicit constant: value starts with '[' or digit/minus
        if self.peek() in "[-0123456789":
            value = parse_expr(self)
            if pre_type is None:
                raise RuntimeError(f"constant %{op_name_str} missing type annotation")
            op = builtin.ConstantOp(
                name=op_name_str,
                value=Memory.from_value(pre_type, value),
                type=pre_type,
            )
            self.name_table[op_name_str] = op
            return op
        name = self.parse_qualified_name()
        cls = self._ops.get(name)
        if cls is None:
            raise RuntimeError(f"Unknown op: {name}")
        return parse_op_fields(self, cls, name=op_name_str, pre_type=pre_type)


def parse_module(text: str) -> builtin.Module:
    parser = IRParser(text)
    return parser.parse_module()
