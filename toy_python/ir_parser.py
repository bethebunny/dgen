"""IR text deserialization for the Toy dialect.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.
"""

from toy_python.dialects.toy_ops import (
    Module,
    FuncOp,
    Block,
    ToyValue,
    AnyToyOp,
    AnyToyType,
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    GenericCallOp,
    PrintOp,
    ReturnOp,
    UnrankedTensorType,
    RankedTensorType,
    FunctionType,
)


class IRParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

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

    def parse_type(self) -> AnyToyType:
        """Parse tensor<*xf64> or tensor<NxMxf64>"""
        self.expect("tensor<")
        if self.peek() == "*":
            self.expect("*xf64>")
            return UnrankedTensorType()
        # Ranked: parse dimensions
        shape = [self.parse_int()]
        while self.peek() == "x":
            self.expect("x")
            if self.peek() == "f":
                break
            shape.append(self.parse_int())
        self.expect("f64>")
        return RankedTensorType(shape=shape)

    def skip_line(self):
        """Skip to the next line."""
        while not self.at_end() and self.peek() != "\n":
            self.pos += 1
        if not self.at_end():
            self.pos += 1  # skip \n

    def parse_module(self) -> Module:
        """Parse a complete module from IR text."""
        # Skip header line: "from toy use *"
        self.skip_whitespace_and_newlines()
        if not self.at_end() and self.peek() == "f":
            self.expect("from toy use *")
            self.skip_line()

        functions: list[FuncOp] = []
        while not self.at_end():
            self.skip_whitespace_and_newlines()
            if self.at_end():
                break
            functions.append(self.parse_func())

        return Module(functions=functions)

    def parse_func(self) -> FuncOp:
        """Parse a function definition."""
        # %name = function (...) [-> Type]:
        name = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        self.expect("function")
        self.skip_whitespace()
        self.expect("(")

        # Parse parameters
        args: list[ToyValue] = []
        input_types: list[AnyToyType] = []
        self.skip_whitespace()
        if self.peek() != ")":
            arg = self._parse_param()
            input_types.append(arg.type)
            args.append(arg)
            self.skip_whitespace()
            while self.peek() == ",":
                self.expect(",")
                self.skip_whitespace()
                arg = self._parse_param()
                input_types.append(arg.type)
                args.append(arg)
                self.skip_whitespace()
        self.expect(")")

        # Optional return type
        self.skip_whitespace()
        result: AnyToyType | None = None
        if not self.at_end() and self.peek() == "-":
            self.expect("->")
            self.skip_whitespace()
            result = self.parse_type()

        self.expect(":")
        self.skip_line()

        # Parse body (indented lines)
        ops: list[AnyToyOp] = []
        while not self.at_end():
            # Check if next line is indented (body) or not (next function)
            if self.peek() not in (" ", "\t"):
                break
            self.skip_whitespace()
            if self.at_end() or self.peek() == "\n":
                self.skip_line()
                continue
            ops.append(self.parse_op())
            self.skip_line()

        func_type = FunctionType(inputs=input_types, result=result)
        return FuncOp(
            name=name,
            func_type=func_type,
            body=Block(args=args, ops=ops),
        )

    def _parse_param(self) -> ToyValue:
        """Parse %name: Type"""
        name = self.parse_ssa_name()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return ToyValue(name=name, type=type_)

    def parse_op(self) -> AnyToyOp:
        """Parse a single operation."""
        # Check for bare ops first: Print(...), return ...
        if self.peek() == "P":
            return self._parse_print_op()
        if self.peek() == "r":
            return self._parse_return_op()

        # %result = OpName(...)
        result = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        op_name = self.parse_identifier()

        if op_name == "Constant":
            return self._parse_constant_body(result)
        if op_name == "Transpose":
            return self._parse_transpose_body(result)
        if op_name == "Reshape":
            return self._parse_reshape_body(result)
        if op_name == "Mul":
            return self._parse_mul_body(result)
        if op_name == "Add":
            return self._parse_add_body(result)
        if op_name == "GenericCall":
            return self._parse_generic_call_body(result)

        raise RuntimeError(f"Unknown op: {op_name}")

    def _parse_print_op(self) -> AnyToyOp:
        self.expect("Print(")
        input_ = self.parse_ssa_name()
        self.expect(")")
        return PrintOp(input=input_)

    def _parse_return_op(self) -> AnyToyOp:
        self.expect("return")
        self.skip_whitespace()
        if self.at_end() or self.peek() == "\n":
            return ReturnOp(value=None)
        val = self.parse_ssa_name()
        return ReturnOp(value=val)

    def _parse_constant_body(self, result: str) -> AnyToyOp:
        """Parse (<shape> [values]) : Type"""
        self.expect("(")
        # Parse shape: <NxM> or <N>
        self.expect("<")
        shape = [self.parse_int()]
        while self.peek() == "x":
            self.expect("x")
            shape.append(self.parse_int())
        self.expect(">")
        self.skip_whitespace()

        # Parse values: [v1, v2, ...]
        self.expect("[")
        values: list[float] = []
        self.skip_whitespace()
        values.append(self.parse_number())
        while self.peek() == ",":
            self.expect(",")
            self.skip_whitespace()
            values.append(self.parse_number())
        self.expect("]")
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()

        return ConstantOp(result=result, value=values, shape=shape, type=type_)

    def _parse_transpose_body(self, result: str) -> AnyToyOp:
        self.expect("(")
        input_ = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return TransposeOp(result=result, input=input_, type=type_)

    def _parse_reshape_body(self, result: str) -> AnyToyOp:
        self.expect("(")
        input_ = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return ReshapeOp(result=result, input=input_, type=type_)

    def _parse_mul_body(self, result: str) -> AnyToyOp:
        self.expect("(")
        lhs = self.parse_ssa_name()
        self.expect(",")
        self.skip_whitespace()
        rhs = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return MulOp(result=result, lhs=lhs, rhs=rhs, type=type_)

    def _parse_add_body(self, result: str) -> AnyToyOp:
        self.expect("(")
        lhs = self.parse_ssa_name()
        self.expect(",")
        self.skip_whitespace()
        rhs = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()
        return AddOp(result=result, lhs=lhs, rhs=rhs, type=type_)

    def _parse_generic_call_body(self, result: str) -> AnyToyOp:
        """Parse @callee(%a, %b) : Type"""
        self.skip_whitespace()
        self.expect("@")
        callee = self.parse_identifier()
        self.expect("(")
        args: list[str] = []
        self.skip_whitespace()
        if self.peek() != ")":
            args.append(self.parse_ssa_name())
            while self.peek() == ",":
                self.expect(",")
                self.skip_whitespace()
                args.append(self.parse_ssa_name())
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        type_ = self.parse_type()

        return GenericCallOp(
            result=result, callee=callee, args=args, type=type_
        )


def parse_module(text: str) -> Module:
    parser = IRParser(text)
    return parser.parse_module()
