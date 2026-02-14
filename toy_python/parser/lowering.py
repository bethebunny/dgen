"""AST to IR lowering for the Toy dialect."""

from __future__ import annotations

from toy_python.parser.ast import (
    Expression,
    Statement,
    NumberLiteral,
    TensorLiteral,
    VarRef,
    BinaryOp,
    CallExpr,
    PrintExpr,
    VarDecl,
    ReturnStmt,
    ExprStmt,
    Function,
    ToyModule,
)
from toy_python.dialects.toy import (
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


def _unranked() -> AnyToyType:
    return UnrankedTensorType()


def _ranked(shape: list[int]) -> AnyToyType:
    return RankedTensorType(shape=shape)


class Lowering:
    def __init__(self):
        self.counter = 0
        self.ops: list[AnyToyOp] = []
        self.scope: dict[str, str] = {}

    def fresh(self) -> str:
        name = str(self.counter)
        self.counter += 1
        return name

    def lower_module(self, tm: ToyModule) -> Module:
        functions = [self.lower_function(f) for f in tm.functions]
        return Module(functions=functions)

    def lower_function(self, f: Function) -> FuncOp:
        # Reset per-function state
        self.counter = 0
        self.ops = []
        self.scope = {}

        # Create block args for function params
        args: list[ToyValue] = []
        input_types: list[AnyToyType] = []
        for param_name in f.proto.params:
            self.scope[param_name] = param_name
            args.append(ToyValue(name=param_name, type=_unranked()))
            input_types.append(_unranked())

        # Lower body statements
        for stmt in f.body:
            self.lower_statement(stmt)

        # Determine return type from ops
        result: AnyToyType | None = None
        if self.ops:
            last_op = self.ops[-1]
            if isinstance(last_op, ReturnOp) and last_op.value is not None:
                result = _unranked()

        ops = self.ops
        self.ops = []
        func_type = FunctionType(inputs=input_types, result=result)
        return FuncOp(
            name=f.proto.name,
            func_type=func_type,
            body=Block(args=args, ops=ops),
        )

    def lower_statement(self, stmt: Statement):
        if isinstance(stmt, VarDecl):
            self._lower_var_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._lower_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self.lower_expr(stmt.expr)

    def _lower_var_decl(self, decl: VarDecl):
        expr_name = self.lower_expr(decl.value)

        # If explicit shape, add a Reshape
        if decl.shape is not None:
            result = self.fresh()
            self.ops.append(
                ReshapeOp(
                    result=result,
                    input=expr_name,
                    type=_ranked(list(decl.shape)),
                )
            )
            self.scope[decl.name] = result
        else:
            self.scope[decl.name] = expr_name

    def _lower_return(self, ret: ReturnStmt):
        if ret.value is not None:
            name = self.lower_expr(ret.value)
            self.ops.append(ReturnOp(value=name))
        else:
            self.ops.append(ReturnOp(value=None))

    def lower_expr(self, expr: Expression) -> str:
        """Lower an expression, return the SSA name of the result."""
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

    def _lower_number(self, num: NumberLiteral) -> str:
        result = self.fresh()
        self.ops.append(
            ConstantOp(
                result=result,
                value=[num.value],
                shape=[1],
                type=_ranked([1]),
            )
        )
        return result

    def _lower_tensor(self, tensor: TensorLiteral) -> str:
        result = self.fresh()
        self.ops.append(
            ConstantOp(
                result=result,
                value=list(tensor.values),
                shape=list(tensor.shape),
                type=_ranked(list(tensor.shape)),
            )
        )
        return result

    def _lower_varref(self, vr: VarRef) -> str:
        if vr.name not in self.scope:
            raise RuntimeError(f"Undefined variable: {vr.name}")
        return self.scope[vr.name]

    def _lower_binop(self, op: BinaryOp) -> str:
        lhs = self.lower_expr(op.lhs)
        rhs = self.lower_expr(op.rhs)
        result = self.fresh()
        if op.op == "*":
            self.ops.append(
                MulOp(result=result, lhs=lhs, rhs=rhs, type=_unranked())
            )
        elif op.op == "+":
            self.ops.append(
                AddOp(result=result, lhs=lhs, rhs=rhs, type=_unranked())
            )
        else:
            raise RuntimeError(f"Unknown binary operator: {op.op}")
        return result

    def _lower_call(self, call: CallExpr) -> str:
        # Builtin: transpose
        if call.callee == "transpose":
            if len(call.args) != 1:
                raise RuntimeError("transpose takes exactly 1 argument")
            arg = self.lower_expr(call.args[0])
            result = self.fresh()
            self.ops.append(
                TransposeOp(result=result, input=arg, type=_unranked())
            )
            return result

        # Generic call
        args = [self.lower_expr(a) for a in call.args]
        result = self.fresh()
        self.ops.append(
            GenericCallOp(
                result=result,
                callee=call.callee,
                args=args,
                type=_unranked(),
            )
        )
        return result

    def _lower_print(self, p: PrintExpr) -> str:
        arg = self.lower_expr(p.arg)
        self.ops.append(PrintOp(input=arg))
        return arg


def lower(tm: ToyModule) -> Module:
    lowering = Lowering()
    return lowering.lower_module(tm)
