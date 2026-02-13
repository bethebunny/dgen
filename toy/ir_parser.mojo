"""IR text deserialization for the Toy dialect.

Line-oriented recursive descent parser that reads IR text format back into
Module data structures.
"""

from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
)
from collections import Optional


@fieldwise_init
struct IRParser(Movable):
    var text: String
    var pos: Int

    fn __init__(out self, text: String):
        self.text = text
        self.pos = 0

    fn at_end(self) -> Bool:
        return self.pos >= len(self.text)

    fn peek(self) -> String:
        if self.at_end():
            return ""
        return String(self.text[byte=self.pos])

    fn advance(mut self) -> String:
        var c = String(self.text[byte=self.pos])
        self.pos += 1
        return c

    fn skip_whitespace(mut self):
        while not self.at_end() and (self.peek() == " " or self.peek() == "\t"):
            self.pos += 1

    fn skip_whitespace_and_newlines(mut self):
        while not self.at_end() and (
            self.peek() == " " or self.peek() == "\t" or self.peek() == "\n" or self.peek() == "\r"
        ):
            self.pos += 1

    fn expect(mut self, expected: String) raises:
        for i in range(len(expected)):
            if self.at_end() or String(self.text[byte=self.pos]) != String(expected[byte=i]):
                raise Error(
                    "Expected '" + expected + "' at position " + String(self.pos)
                )
            self.pos += 1

    fn parse_identifier(mut self) raises -> String:
        """Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*"""
        var start = self.pos
        if self.at_end():
            raise Error("Expected identifier at position " + String(self.pos))
        var c = self.peek()
        if not (_is_alpha(c) or c == "_"):
            raise Error("Expected identifier at position " + String(self.pos))
        self.pos += 1
        while not self.at_end():
            c = self.peek()
            if _is_alpha(c) or _is_digit(c) or c == "_":
                self.pos += 1
            else:
                break
        return String(self.text[start:self.pos])

    fn parse_ssa_name(mut self) raises -> String:
        """Parse %name, return the name part."""
        self.expect("%")
        return self._parse_name_chars()

    fn _parse_name_chars(mut self) raises -> String:
        """Parse [a-zA-Z0-9_]+"""
        var start = self.pos
        while not self.at_end():
            var c = self.peek()
            if _is_alpha(c) or _is_digit(c) or c == "_":
                self.pos += 1
            else:
                break
        if self.pos == start:
            raise Error("Expected name at position " + String(self.pos))
        return String(self.text[start:self.pos])

    fn parse_number(mut self) raises -> Float64:
        """Parse a floating point number."""
        var start = self.pos
        if not self.at_end() and self.peek() == "-":
            self.pos += 1
        while not self.at_end() and _is_digit(self.peek()):
            self.pos += 1
        if not self.at_end() and self.peek() == ".":
            self.pos += 1
            while not self.at_end() and _is_digit(self.peek()):
                self.pos += 1
        if self.pos == start:
            raise Error("Expected number at position " + String(self.pos))
        return Float64(atof(String(self.text[start:self.pos])))

    fn parse_int(mut self) raises -> Int:
        """Parse a non-negative integer."""
        var start = self.pos
        while not self.at_end() and _is_digit(self.peek()):
            self.pos += 1
        if self.pos == start:
            raise Error("Expected integer at position " + String(self.pos))
        return Int(String(self.text[start:self.pos]))

    fn parse_type(mut self) raises -> AnyToyType:
        """Parse tensor<*xf64> or tensor<NxMxf64>"""
        self.expect("tensor<")
        if self.peek() == "*":
            self.expect("*xf64>")
            return AnyToyType(UnrankedTensorType())
        # Ranked: parse dimensions
        var shape = List[Int]()
        shape.append(self.parse_int())
        while self.peek() == "x":
            self.expect("x")
            if self.peek() == "f":
                break
            shape.append(self.parse_int())
        self.expect("f64>")
        return AnyToyType(RankedTensorType(shape=shape^))

    fn skip_line(mut self):
        """Skip to the next line."""
        while not self.at_end() and self.peek() != "\n":
            self.pos += 1
        if not self.at_end():
            self.pos += 1  # skip \n

    fn parse_module(mut self) raises -> Module:
        """Parse a complete module from IR text."""
        # Skip header line: "from toy use *"
        self.skip_whitespace_and_newlines()
        if not self.at_end() and self.peek() == "f":
            self.expect("from toy use *")
            self.skip_line()

        var functions = List[FuncOp]()
        while not self.at_end():
            self.skip_whitespace_and_newlines()
            if self.at_end():
                break
            functions.append(self.parse_func())

        return Module(functions=functions^)

    fn parse_func(mut self) raises -> FuncOp:
        """Parse a function definition."""
        # %name = function (...) [-> Type]:
        var name = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        self.expect("function")
        self.skip_whitespace()
        self.expect("(")

        # Parse parameters
        var args = List[ToyValue]()
        var input_types = List[AnyToyType]()
        self.skip_whitespace()
        if self.peek() != ")":
            var arg = self._parse_param()
            input_types.append(arg.type)
            args.append(arg^)
            self.skip_whitespace()
            while self.peek() == ",":
                self.expect(",")
                self.skip_whitespace()
                arg = self._parse_param()
                input_types.append(arg.type)
                args.append(arg^)
                self.skip_whitespace()
        self.expect(")")

        # Optional return type
        self.skip_whitespace()
        var result = Optional[AnyToyType]()
        if not self.at_end() and self.peek() == "-":
            self.expect("->")
            self.skip_whitespace()
            result = self.parse_type()

        self.expect(":")
        self.skip_line()

        # Parse body (indented lines)
        var ops = List[AnyToyOp]()
        while not self.at_end():
            # Check if next line is indented (body) or not (next function)
            if self.peek() != " " and self.peek() != "\t":
                break
            self.skip_whitespace()
            if self.at_end() or self.peek() == "\n":
                self.skip_line()
                continue
            ops.append(self.parse_op())
            self.skip_line()

        var func_type = FunctionType(inputs=input_types^, result=result^)
        return FuncOp(
            name=name^,
            func_type=func_type^,
            body=Block(args=args^, ops=ops^),
        )

    fn _parse_param(mut self) raises -> ToyValue:
        """Parse %name: Type"""
        var name = self.parse_ssa_name()
        self.expect(":")
        self.skip_whitespace()
        var type = self.parse_type()
        return ToyValue(name=name^, type=type^)

    fn parse_op(mut self) raises -> AnyToyOp:
        """Parse a single operation."""
        # Check for bare ops first: Print(...), return ...
        if self.peek() == "P":
            return self._parse_print_op()
        if self.peek() == "r":
            return self._parse_return_op()

        # %result = OpName(...)
        var result = self.parse_ssa_name()
        self.skip_whitespace()
        self.expect("=")
        self.skip_whitespace()
        var op_name = self.parse_identifier()

        if op_name == "Constant":
            return self._parse_constant_body(result^)
        if op_name == "Transpose":
            return self._parse_transpose_body(result^)
        if op_name == "Reshape":
            return self._parse_reshape_body(result^)
        if op_name == "Mul":
            return self._parse_mul_body(result^)
        if op_name == "Add":
            return self._parse_add_body(result^)
        if op_name == "GenericCall":
            return self._parse_generic_call_body(result^)

        raise Error("Unknown op: " + op_name)

    fn _parse_print_op(mut self) raises -> AnyToyOp:
        self.expect("Print(")
        var input = self.parse_ssa_name()
        self.expect(")")
        return AnyToyOp(PrintOp(input=input^))

    fn _parse_return_op(mut self) raises -> AnyToyOp:
        self.expect("return")
        self.skip_whitespace()
        if self.at_end() or self.peek() == "\n":
            return AnyToyOp(ReturnOp(value=Optional[String]()))
        var val = self.parse_ssa_name()
        return AnyToyOp(ReturnOp(value=val^))

    fn _parse_constant_body(mut self, var result: String) raises -> AnyToyOp:
        """Parse (<shape> [values]) : Type"""
        self.expect("(")
        # Parse shape: <NxM> or <N>
        self.expect("<")
        var shape = List[Int]()
        shape.append(self.parse_int())
        while self.peek() == "x":
            self.expect("x")
            shape.append(self.parse_int())
        self.expect(">")
        self.skip_whitespace()

        # Parse values: [v1, v2, ...]
        self.expect("[")
        var values = List[Float64]()
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
        var type = self.parse_type()

        return AnyToyOp(ConstantOp(
            result=result^, value=values^, shape=shape^, type=type^,
        ))

    fn _parse_transpose_body(mut self, var result: String) raises -> AnyToyOp:
        self.expect("(")
        var input = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        var type = self.parse_type()
        return AnyToyOp(TransposeOp(result=result^, input=input^, type=type^))

    fn _parse_reshape_body(mut self, var result: String) raises -> AnyToyOp:
        self.expect("(")
        var input = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        var type = self.parse_type()
        return AnyToyOp(ReshapeOp(result=result^, input=input^, type=type^))

    fn _parse_mul_body(mut self, var result: String) raises -> AnyToyOp:
        self.expect("(")
        var lhs = self.parse_ssa_name()
        self.expect(",")
        self.skip_whitespace()
        var rhs = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        var type = self.parse_type()
        return AnyToyOp(MulOp(result=result^, lhs=lhs^, rhs=rhs^, type=type^))

    fn _parse_add_body(mut self, var result: String) raises -> AnyToyOp:
        self.expect("(")
        var lhs = self.parse_ssa_name()
        self.expect(",")
        self.skip_whitespace()
        var rhs = self.parse_ssa_name()
        self.expect(")")
        self.skip_whitespace()
        self.expect(":")
        self.skip_whitespace()
        var type = self.parse_type()
        return AnyToyOp(AddOp(result=result^, lhs=lhs^, rhs=rhs^, type=type^))

    fn _parse_generic_call_body(mut self, var result: String) raises -> AnyToyOp:
        """Parse @callee(%a, %b) : Type"""
        self.skip_whitespace()
        self.expect("@")
        var callee = self.parse_identifier()
        self.expect("(")
        var args = List[String]()
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
        var type = self.parse_type()

        return AnyToyOp(GenericCallOp(
            result=result^, callee=callee^, args=args^, type=type^,
        ))


fn _is_alpha(c: String) -> Bool:
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z")


fn _is_digit(c: String) -> Bool:
    return c >= "0" and c <= "9"


fn parse_module(text: String) raises -> Module:
    var parser = IRParser(text)
    return parser.parse_module()
