"""Dialect-agnostic IR text deserialization.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.  Dialect knowledge (ops, keyword ops, types) is
supplied via lookup tables passed by the caller.
"""

from toy_python.dialects import builtin
from toy_python.ir_format import parse_op_fields


class IRParser:
    def __init__(self, text: str, ops: dict, keywords: dict, types: dict):
        self.text = text
        self.pos = 0
        self._ops = ops
        self._keywords = keywords
        self._types = types

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

    def skip_whitespace(self):
        while not self.at_end() and self.text[self.pos] in " \t":
            self.pos += 1

    def skip_whitespace_and_newlines(self):
        while not self.at_end() and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def expect(self, expected: str):
        for ch in expected:
            if self.at_end() or self.text[self.pos] != ch:
                raise RuntimeError(
                    f"Expected '{expected}' at position {self.pos}"
                )
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

    def parse_type(self) -> builtin.Type:
        """Parse a type via the registered type table, or () for Nil."""
        if self.peek() == "(":
            self.expect("()")
            return builtin.Nil()
        name = self.parse_identifier()
        if name not in self._types:
            raise RuntimeError(f"Unknown type: {name}")
        return self._types[name](self)

    def skip_line(self):
        """Skip to the next line."""
        while not self.at_end() and self.peek() != "\n":
            self.pos += 1
        if not self.at_end():
            self.pos += 1  # skip \n

    def parse_module(self) -> builtin.Module:
        """Parse a complete module from IR text."""
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
        name = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        self.expect("function")
        self.skip_whitespace()
        self.expect("(")

        # Parse parameters
        args: list[builtin.Value] = []
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
        result: builtin.Type = self.parse_type()

        self.expect(":")
        self.skip_line()

        # Parse body (indented lines)
        ops = self._parse_block(min_indent=1)

        func_type = builtin.FuncType(result=result)
        return builtin.FuncOp(
            name=name,
            func_type=func_type,
            body=builtin.Block(ops=ops, args=args),
        )

    def _parse_param(self) -> builtin.Value:
        """Parse %name: Type"""
        name = self.parse_ssa_name()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return builtin.Value(name=name, type=type_)

    def _parse_block(self, min_indent: int) -> list[builtin.Op]:
        """Parse a block of indented ops."""
        ops: list[builtin.Op] = []
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

            ops.append(self.parse_op())

            # If parse_op consumed past the original newline (body-bearing op),
            # the cursor is already at the next line's start — don't skip.
            if self.pos <= eol:
                self.skip_line()
        return ops

    def parse_indented_block(self) -> list[builtin.Op]:
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

    def parse_op(self) -> builtin.Op:
        """Parse a single operation via registered tables."""
        if self.peek() != "%":
            name = self.parse_identifier()
            if name not in self._keywords:
                raise RuntimeError(f"Unknown keyword op: {name}")
            cls = self._keywords[name]
            return parse_op_fields(self, cls)

        result = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        op_name = self.parse_identifier()
        if op_name not in self._ops:
            raise RuntimeError(f"Unknown op: {op_name}")
        cls = self._ops[op_name]
        return parse_op_fields(self, cls, result=result)


def parse_module(text: str, ops: dict, keywords: dict, types: dict) -> builtin.Module:
    parser = IRParser(text, ops, keywords, types)
    return parser.parse_module()
