"""Dialect-agnostic IR text deserialization.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.  Dialect knowledge is discovered from import headers
at the top of the IR text.
"""

import importlib

from toy_python.dialects import builtin
from toy_python.ir_format import parse_op_fields


class IRParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self._dialect_ops: dict[str, dict] = {}
        self._dialect_keywords: dict[str, dict] = {}
        self._unqualified_ops: dict[str, type] = {}
        self._unqualified_keywords: dict[str, type] = {}
        self._types: dict = {}

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

    # ===------------------------------------------------------------------=== #
    # Import header parsing
    # ===------------------------------------------------------------------=== #

    def _parse_imports(self):
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
                self.skip_whitespace()
                names = [self.parse_identifier()]
                self.skip_whitespace()
                while not self.at_end() and self.peek() == ",":
                    self.advance()
                    self.skip_whitespace()
                    names.append(self.parse_identifier())
                    self.skip_whitespace()
                self.skip_line()
                for name in names:
                    if name == "function":
                        continue  # handled by parse_func
                    if name in builtin.KEYWORD_TABLE:
                        self._unqualified_keywords[name] = builtin.KEYWORD_TABLE[name]

            elif word == "import":
                self.skip_whitespace()
                dialect_name = self.parse_identifier()
                self.skip_line()
                mod = importlib.import_module(
                    f"toy_python.dialects.{dialect_name}"
                )
                self._dialect_ops[dialect_name] = getattr(mod, "OP_TABLE", {})
                self._dialect_keywords[dialect_name] = getattr(
                    mod, "KEYWORD_TABLE", {}
                )
                self._types.update(getattr(mod, "TYPE_TABLE", {}))
                # Register builtin ops from this dialect in unqualified tables
                for name, cls in getattr(mod, "OP_TABLE", {}).items():
                    if getattr(cls, "_builtin", False):
                        self._unqualified_ops[name] = cls
                for name, cls in getattr(mod, "KEYWORD_TABLE", {}).items():
                    if getattr(cls, "_builtin", False):
                        self._unqualified_keywords[name] = cls
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
        """Parse a single operation, handling dialect-qualified names."""
        if self.peek() != "%":
            # Keyword op (no result) — may be qualified: dialect.op(...)
            name = self.parse_identifier()
            if not self.at_end() and self.peek() == ".":
                self.advance()  # skip '.'
                op_name = self.parse_identifier()
                cls = self._dialect_keywords.get(name, {}).get(op_name)
                if cls is None:
                    cls = self._dialect_ops.get(name, {}).get(op_name)
                if cls is None:
                    raise RuntimeError(f"Unknown op: {name}.{op_name}")
                return parse_op_fields(self, cls)
            if name not in self._unqualified_keywords:
                raise RuntimeError(f"Unknown keyword op: {name}")
            cls = self._unqualified_keywords[name]
            return parse_op_fields(self, cls)

        # Result op: %result = [dialect.]op(...)
        result = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        name = self.parse_identifier()
        if not self.at_end() and self.peek() == ".":
            self.advance()  # skip '.'
            op_name = self.parse_identifier()
            cls = self._dialect_ops.get(name, {}).get(op_name)
            if cls is None:
                cls = self._dialect_keywords.get(name, {}).get(op_name)
            if cls is None:
                raise RuntimeError(f"Unknown op: {name}.{op_name}")
            return parse_op_fields(self, cls, result=result)
        if name not in self._unqualified_ops:
            raise RuntimeError(f"Unknown op: {name}")
        cls = self._unqualified_ops[name]
        return parse_op_fields(self, cls, result=result)


def parse_module(text: str) -> builtin.Module:
    parser = IRParser(text)
    return parser.parse_module()
