"""AST to IR lowering for the Toy dialect."""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin
from toy.dialects import toy
from toy.parser.ast import (
    BinaryOp,
    CallExpr,
    Expression,
    ExprStmt,
    Function,
    NumberLiteral,
    PrintExpr,
    ReturnStmt,
    Statement,
    TensorLiteral,
    ToyModule,
    VarDecl,
    VarRef,
)


def _inferred() -> dgen.Type:
    return toy.InferredShapeTensor()


def _ranked(shape: list[int]) -> dgen.Type:
    return toy.TensorType(shape=shape)


class Lowering:
    def __init__(self):
        self.scope: dict[str, dgen.Value] = {}

    def lower_module(self, tm: ToyModule) -> builtin.Module:
        functions = [self.lower_function(f) for f in tm.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: Function) -> builtin.FuncOp:
        self.scope = {}

        # Create block args for function params
        args: list[BlockArgument] = []
        input_types: list[dgen.Type] = []
        for param_name in f.proto.params:
            arg = BlockArgument(name=param_name, type=_inferred())
            self.scope[param_name] = arg
            args.append(arg)
            input_types.append(_inferred())

        # Lower body statements
        ops = []
        for stmt in f.body:
            ops.extend(self._lower_statement(stmt))

        # Determine return type from ops
        result: dgen.Type = builtin.Nil()
        if ops:
            last_op = ops[-1]
            if isinstance(last_op, builtin.ReturnOp) and last_op.value is not None:
                result = _inferred()

        func_type = toy.FunctionType(inputs=input_types, result=result)
        return builtin.FuncOp(
            name=f.proto.name,
            type=func_type,
            body=dgen.Block(ops=ops, args=args),
        )

    def _lower_statement(self, stmt: Statement):
        if isinstance(stmt, VarDecl):
            yield from self._lower_var_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            yield from self._lower_return(stmt)
        elif isinstance(stmt, ExprStmt):
            yield from self.lower_expr(stmt.expr)

    def _lower_var_decl(self, decl: VarDecl):
        expr_val = yield from self.lower_expr(decl.value)

        # If explicit shape, add a Reshape
        if decl.shape is not None:
            op = toy.ReshapeOp(
                input=expr_val,
                type=_ranked(list(decl.shape)),
            )
            yield op
            self.scope[decl.name] = op
        else:
            self.scope[decl.name] = expr_val

    def _lower_return(self, ret: ReturnStmt):
        if ret.value is not None:
            val = yield from self.lower_expr(ret.value)
            yield builtin.ReturnOp(value=val)
        else:
            yield builtin.ReturnOp()

    def lower_expr(self, expr: Expression):
        """Lower an expression, yielding ops and returning the result Value."""
        if isinstance(expr, NumberLiteral):
            op = builtin.ConstantOp(
                value=[expr.value],
                type=_ranked([1]),
            )
            yield op
            return op
        if isinstance(expr, TensorLiteral):
            op = builtin.ConstantOp(
                value=list(expr.values),
                type=_ranked(list(expr.shape)),
            )
            yield op
            return op
        if isinstance(expr, VarRef):
            if expr.name not in self.scope:
                raise RuntimeError(f"Undefined variable: {expr.name}")
            return self.scope[expr.name]
        if isinstance(expr, BinaryOp):
            lhs = yield from self.lower_expr(expr.lhs)
            rhs = yield from self.lower_expr(expr.rhs)
            if expr.op == "*":
                result_op = toy.MulOp(lhs=lhs, rhs=rhs, type=_inferred())
            elif expr.op == "+":
                result_op = toy.AddOp(lhs=lhs, rhs=rhs, type=_inferred())
            else:
                raise RuntimeError(f"Unknown binary operator: {expr.op}")
            yield result_op
            return result_op
        if isinstance(expr, CallExpr):
            return (yield from self._lower_call(expr))
        if isinstance(expr, PrintExpr):
            return (yield from self._lower_print(expr))
        raise RuntimeError("Unknown expression type")

    def _lower_call(self, call: CallExpr):
        # Builtin: transpose
        if call.callee == "transpose":
            if len(call.args) != 1:
                raise RuntimeError("transpose takes exactly 1 argument")
            arg = yield from self.lower_expr(call.args[0])
            op = toy.TransposeOp(input=arg, type=_inferred())
            yield op
            return op

        # Generic call
        args = []
        for a in call.args:
            args.append((yield from self.lower_expr(a)))
        op = toy.GenericCallOp(
            callee=call.callee,
            args=args,
            type=_inferred(),
        )
        yield op
        return op

    def _lower_print(self, p: PrintExpr):
        arg = yield from self.lower_expr(p.arg)
        yield toy.PrintOp(input=arg)
        return arg


def lower(tm: ToyModule) -> builtin.Module:
    lowering = Lowering()
    return lowering.lower_module(tm)
