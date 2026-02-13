"""Toy source language parser (recursive descent)."""

from collections import Optional

from toy.parser.ast import (
    AnyExpr, AnyStmt, NumberLiteral, TensorLiteral, VarRef,
    BinaryOp, CallExpr, PrintExpr, VarDecl, ReturnStmt, ExprStmt,
    Function, Prototype, ToyModule, ExprArena,
)
from toy.parser.lexer import Lexer, Token


struct ToyParser(Movable):
    var lexer: Lexer
    var current: Token
    var arena: ExprArena

    fn __init__(out self, text: String):
        self.lexer = Lexer(text)
        self.current = self.lexer.next_token()
        self.arena = ExprArena()

    fn advance(mut self):
        self.current = self.lexer.next_token()

    fn expect(mut self, kind: String) raises -> Token:
        if self.current.kind != kind:
            raise Error(
                "Expected '" + kind + "' but got '" + self.current.kind
                + "' ('" + self.current.text + "')"
            )
        var tok = self.current.copy()
        self.advance()
        return tok^

    fn parse_module(mut self) raises -> ToyModule:
        var functions = List[Function]()
        while self.current.kind != "EOF":
            functions.append(self.parse_function())
        var arena = self.arena^
        self.arena = ExprArena()
        return ToyModule(functions=functions^, arena=arena^)

    fn parse_function(mut self) raises -> Function:
        """Parse: def name(params) { stmts }"""
        _ = self.expect("def")
        var name = self.expect("IDENT").text
        _ = self.expect("(")
        var params = List[String]()
        if self.current.kind != ")":
            params.append(self.expect("IDENT").text)
            while self.current.kind == ",":
                self.advance()
                params.append(self.expect("IDENT").text)
        _ = self.expect(")")
        _ = self.expect("{")
        var body = List[AnyStmt]()
        while self.current.kind != "}":
            body.append(self.parse_statement())
        _ = self.expect("}")
        return Function(
            proto=Prototype(name=name^, params=params^),
            body=body^,
        )

    fn parse_statement(mut self) raises -> AnyStmt:
        if self.current.kind == "var":
            return self._parse_var_decl()
        if self.current.kind == "return":
            return self._parse_return()
        return self._parse_expr_stmt()

    fn _parse_var_decl(mut self) raises -> AnyStmt:
        """Parse: var name [<shape>] = expr ;"""
        _ = self.expect("var")
        var name = self.expect("IDENT").text

        # Optional shape annotation: <N, M>
        var shape = Optional[List[Int]]()
        if self.current.kind == "<":
            self.advance()
            var dims = List[Int]()
            dims.append(Int(self.expect("NUMBER").text))
            while self.current.kind == ",":
                self.advance()
                dims.append(Int(self.expect("NUMBER").text))
            _ = self.expect(">")
            shape = dims^

        _ = self.expect("=")
        var value_idx = self.parse_expression()
        _ = self.expect(";")
        return AnyStmt(VarDecl(name=name^, shape=shape^, value=value_idx))

    fn _parse_return(mut self) raises -> AnyStmt:
        """Parse: return [expr] ;"""
        _ = self.expect("return")
        if self.current.kind == ";":
            self.advance()
            return AnyStmt(ReturnStmt(value=Optional[Int]()))
        var value_idx = self.parse_expression()
        _ = self.expect(";")
        return AnyStmt(ReturnStmt(value=value_idx))

    fn _parse_expr_stmt(mut self) raises -> AnyStmt:
        """Parse: expr ;"""
        var expr_idx = self.parse_expression()
        _ = self.expect(";")
        return AnyStmt(ExprStmt(expr=expr_idx))

    fn parse_expression(mut self) raises -> Int:
        """Parse: primary (('*' | '+') primary)*. Returns arena index."""
        var lhs_idx = self.parse_primary()
        while self.current.kind == "*" or self.current.kind == "+":
            var op = self.current.text
            self.advance()
            var rhs_idx = self.parse_primary()
            lhs_idx = self.arena.add(AnyExpr(BinaryOp(
                op=op^, lhs=lhs_idx, rhs=rhs_idx,
            )))
        return lhs_idx

    fn parse_primary(mut self) raises -> Int:
        """Parse primary expression. Returns arena index."""
        if self.current.kind == "NUMBER":
            var val = Float64(atof(self.current.text))
            self.advance()
            return self.arena.add(AnyExpr(NumberLiteral(value=val)))

        if self.current.kind == "[":
            return self._parse_tensor_literal()

        if self.current.kind == "IDENT":
            var name = self.current.text
            self.advance()
            if self.current.kind == "(":
                return self._parse_call(name^)
            return self.arena.add(AnyExpr(VarRef(name=name^)))

        if self.current.kind == "(":
            self.advance()
            var idx = self.parse_expression()
            _ = self.expect(")")
            return idx

        raise Error("Unexpected token: " + self.current.kind + " '" + self.current.text + "'")

    fn _parse_call(mut self, var callee: String) raises -> Int:
        """Parse the argument list of a call: name(args). Returns arena index."""
        _ = self.expect("(")
        # Special handling for print
        if callee == "print":
            var arg_idx = self.parse_expression()
            _ = self.expect(")")
            return self.arena.add(AnyExpr(PrintExpr(arg=arg_idx)))
        var args = List[Int]()
        if self.current.kind != ")":
            args.append(self.parse_expression())
            while self.current.kind == ",":
                self.advance()
                args.append(self.parse_expression())
        _ = self.expect(")")
        return self.arena.add(AnyExpr(CallExpr(callee=callee^, args=args^)))

    fn _parse_tensor_literal(mut self) raises -> Int:
        """Parse nested bracket tensor literal, flatten values, infer shape."""
        var values = List[Float64]()
        var shape = List[Int]()
        self._parse_tensor_level(values, shape, 0)
        return self.arena.add(AnyExpr(TensorLiteral(values=values^, shape=shape^)))

    fn _parse_tensor_level(
        mut self,
        mut values: List[Float64],
        mut shape: List[Int],
        depth: Int,
    ) raises:
        """Recursively parse one level of [...] nesting."""
        _ = self.expect("[")
        var count = 0
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
            values.append(Float64(atof(self.current.text)))
            _ = self.expect("NUMBER")
            count += 1
            while self.current.kind == ",":
                self.advance()
                values.append(Float64(atof(self.current.text)))
                _ = self.expect("NUMBER")
                count += 1
        _ = self.expect("]")
        # Record dimension at this depth (extend shape list if needed,
        # then set at the correct position)
        while len(shape) <= depth:
            shape.append(0)
        shape[depth] = count


fn parse_toy(text: String) raises -> ToyModule:
    var parser = ToyParser(text)
    return parser.parse_module()
