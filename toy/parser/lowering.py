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
        self.ops: list[dgen.Op] = []
        self.scope: dict[str, dgen.Value] = {}

    def lower_module(self, tm: ToyModule) -> builtin.Module:
        functions = [self.lower_function(f) for f in tm.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: Function) -> builtin.FuncOp:
        # Reset per-function state
        self.ops = []
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
        for stmt in f.body:
            self.lower_statement(stmt)

        # Determine return type from ops
        result: dgen.Type = builtin.Nil()
        if self.ops:
            last_op = self.ops[-1]
            if isinstance(last_op, builtin.ReturnOp) and last_op.value is not None:
                result = _inferred()

        ops = self.ops
        self.ops = []
        func_type = toy.FunctionType(inputs=input_types, result=result)
        return builtin.FuncOp(
            name=f.proto.name,
            type=func_type,
            body=dgen.Block(ops=ops, args=args),
        )

    def lower_statement(self, stmt: Statement):
        if isinstance(stmt, VarDecl):
            self._lower_var_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._lower_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self.lower_expr(stmt.expr)

    def _lower_var_decl(self, decl: VarDecl):
        expr_val = self.lower_expr(decl.value)

        # If explicit shape, add a Reshape
        if decl.shape is not None:
            op = toy.ReshapeOp(
                input=expr_val,
                type=_ranked(list(decl.shape)),
            )
            self.ops.append(op)
            self.scope[decl.name] = op
        else:
            self.scope[decl.name] = expr_val

    def _lower_return(self, ret: ReturnStmt):
        if ret.value is not None:
            val = self.lower_expr(ret.value)
            self.ops.append(builtin.ReturnOp(value=val))
        else:
            self.ops.append(builtin.ReturnOp())

    def lower_expr(self, expr: Expression) -> dgen.Value:
        """Lower an expression, return the Value of the result."""
        if isinstance(expr, NumberLiteral):
            return self._lower_number(expr)
        if isinstance(expr, TensorLiteral):
            return self._lower_tensor(expr)
        if isinstance(expr, VarRef):
            return self._lower_varref(expr)
        if isinstance(expr, BinaryOp):
            return self._lower_binop(expr)
        if isinstance(expr, CallExpr):
            return self._lower_call(expr)
        if isinstance(expr, PrintExpr):
            return self._lower_print(expr)
        raise RuntimeError("Unknown expression type")

    def _lower_number(self, num: NumberLiteral) -> dgen.Value:
        op = builtin.ConstantOp(
            value=[num.value],
            type=_ranked([1]),
        )
        self.ops.append(op)
        return op

    def _lower_tensor(self, tensor: TensorLiteral) -> dgen.Value:
        op = builtin.ConstantOp(
            value=list(tensor.values),
            type=_ranked(list(tensor.shape)),
        )
        self.ops.append(op)
        return op

    def _lower_varref(self, vr: VarRef) -> dgen.Value:
        if vr.name not in self.scope:
            raise RuntimeError(f"Undefined variable: {vr.name}")
        return self.scope[vr.name]

    def _lower_binop(self, op: BinaryOp) -> dgen.Value:
        lhs = self.lower_expr(op.lhs)
        rhs = self.lower_expr(op.rhs)
        if op.op == "*":
            result_op = toy.MulOp(lhs=lhs, rhs=rhs, type=_inferred())
        elif op.op == "+":
            result_op = toy.AddOp(lhs=lhs, rhs=rhs, type=_inferred())
        else:
            raise RuntimeError(f"Unknown binary operator: {op.op}")
        self.ops.append(result_op)
        return result_op

    def _lower_call(self, call: CallExpr) -> dgen.Value:
        # Builtin: transpose
        if call.callee == "transpose":
            if len(call.args) != 1:
                raise RuntimeError("transpose takes exactly 1 argument")
            arg = self.lower_expr(call.args[0])
            op = toy.TransposeOp(input=arg, type=_inferred())
            self.ops.append(op)
            return op

        # Generic call
        args = [self.lower_expr(a) for a in call.args]
        op = toy.GenericCallOp(
            callee=call.callee,
            args=args,
            type=_inferred(),
        )
        self.ops.append(op)
        return op

    def _lower_print(self, p: PrintExpr) -> dgen.Value:
        arg = self.lower_expr(p.arg)
        self.ops.append(toy.PrintOp(input=arg))
        return arg


def lower(tm: ToyModule) -> builtin.Module:
    lowering = Lowering()
    return lowering.lower_module(tm)
