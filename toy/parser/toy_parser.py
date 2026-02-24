"""Toy source language parser (recursive descent)."""

from __future__ import annotations

from toy.parser.ast import (
    BinaryOp,
    CallExpr,
    Expression,
    ExprStmt,
    Function,
    NumberLiteral,
    PrintExpr,
    Prototype,
    ReturnStmt,
    Statement,
    TensorLiteral,
    ToyModule,
    VarDecl,
    VarRef,
)
from toy.parser.lexer import Lexer, Token


class ToyParser:
    def __init__(self, text: str) -> None:
        self.lexer = Lexer(text)
        self.current = self.lexer.next_token()

    def advance(self) -> None:
        self.current = self.lexer.next_token()

    def expect(self, kind: str) -> Token:
        if self.current.kind != kind:
            raise RuntimeError(
                f"Expected '{kind}' but got '{self.current.kind}' "
                f"('{self.current.text}')"
            )
        tok = self.current
        self.advance()
        return tok

    def parse_module(self) -> ToyModule:
        functions: list[Function] = []
        while self.current.kind != "EOF":
            functions.append(self.parse_function())
        return ToyModule(functions=functions)

    def parse_function(self) -> Function:
        """Parse: def name(params) { stmts }"""
        self.expect("def")
        name = self.expect("IDENT").text
        self.expect("(")
        params: list[str] = []
        if self.current.kind != ")":
            params.append(self.expect("IDENT").text)
            while self.current.kind == ",":
                self.advance()
                params.append(self.expect("IDENT").text)
        self.expect(")")
        self.expect("{")
        body: list[Statement] = []
        while self.current.kind != "}":
            body.append(self.parse_statement())
        self.expect("}")
        return Function(proto=Prototype(name=name, params=params), body=body)

    def parse_statement(self) -> Statement:
        if self.current.kind == "var":
            return self._parse_var_decl()
        if self.current.kind == "return":
            return self._parse_return()
        return self._parse_expr_stmt()

    def _parse_var_decl(self) -> VarDecl:
        """Parse: var name [<shape>] = expr ;"""
        self.expect("var")
        name = self.expect("IDENT").text

        # Optional shape annotation: <N, M>
        shape: list[int] | None = None
        if self.current.kind == "<":
            self.advance()
            dims = [int(self.expect("NUMBER").text)]
            while self.current.kind == ",":
                self.advance()
                dims.append(int(self.expect("NUMBER").text))
            self.expect(">")
            shape = dims

        self.expect("=")
        value = self.parse_expression()
        self.expect(";")
        return VarDecl(name=name, shape=shape, value=value)

    def _parse_return(self) -> ReturnStmt:
        """Parse: return [expr] ;"""
        self.expect("return")
        if self.current.kind == ";":
            self.advance()
            return ReturnStmt(value=None)
        value = self.parse_expression()
        self.expect(";")
        return ReturnStmt(value=value)

    def _parse_expr_stmt(self) -> ExprStmt:
        """Parse: expr ;"""
        expr = self.parse_expression()
        self.expect(";")
        return ExprStmt(expr=expr)

    def parse_expression(self) -> Expression:
        """Parse: primary (('*' | '+') primary)*"""
        lhs = self.parse_primary()
        while self.current.kind in ("*", "+"):
            op = self.current.text
            self.advance()
            rhs = self.parse_primary()
            lhs = BinaryOp(op=op, lhs=lhs, rhs=rhs)
        return lhs

    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        if self.current.kind == "NUMBER":
            val = float(self.current.text)
            self.advance()
            return NumberLiteral(value=val)

        if self.current.kind == "[":
            return self._parse_tensor_literal()

        if self.current.kind == "IDENT":
            name = self.current.text
            self.advance()
            if self.current.kind == "(":
                return self._parse_call(name)
            return VarRef(name=name)

        if self.current.kind == "(":
            self.advance()
            expr = self.parse_expression()
            self.expect(")")
            return expr

        raise RuntimeError(
            f"Unexpected token: {self.current.kind} '{self.current.text}'"
        )

    def _parse_call(self, callee: str) -> Expression:
        """Parse the argument list of a call: name(args)."""
        self.expect("(")
        # Special handling for print
        if callee == "print":
            arg = self.parse_expression()
            self.expect(")")
            return PrintExpr(arg=arg)
        args: list[Expression] = []
        if self.current.kind != ")":
            args.append(self.parse_expression())
            while self.current.kind == ",":
                self.advance()
                args.append(self.parse_expression())
        self.expect(")")
        return CallExpr(callee=callee, args=args)

    def _parse_tensor_literal(self) -> TensorLiteral:
        """Parse nested bracket tensor literal, flatten values, infer shape."""
        values: list[float] = []
        shape: list[int] = []
        self._parse_tensor_level(values, shape, 0)
        return TensorLiteral(values=values, shape=shape)

    def _parse_tensor_level(
        self, values: list[float], shape: list[int], depth: int
    ) -> None:
        """Recursively parse one level of [...] nesting."""
        self.expect("[")
        count = 0
        if self.current.kind == "[":
            # Nested level
            self._parse_tensor_level(values, shape, depth + 1)
            count += 1
            while self.current.kind == ",":
                self.advance()
                self._parse_tensor_level(values, shape, depth + 1)
                count += 1
        else:
            # Leaf level: numbers
            values.append(float(self.current.text))
            self.expect("NUMBER")
            count += 1
            while self.current.kind == ",":
                self.advance()
                values.append(float(self.current.text))
                self.expect("NUMBER")
                count += 1
        self.expect("]")
        while len(shape) <= depth:
            shape.append(0)
        shape[depth] = count


def parse_toy(text: str) -> ToyModule:
    parser = ToyParser(text)
    return parser.parse_module()
