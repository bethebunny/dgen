"""AST to IR lowering for the Toy dialect."""

from __future__ import annotations

from collections.abc import Generator, Iterator

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, function, index
from dgen.dialects.function import Function as FunctionType
from dgen.module import ConstantOp, Module, pack
from toy.dialects import shape_constant, toy
from toy.parser.ast import (
    BinaryOp,
    CallExpr,
    Expression,
    ExprStmt,
    Function,
    NumberLiteral,
    ReturnStmt,
    Statement,
    TensorLiteral,
    ToyModule,
    VarDecl,
    VarRef,
)


class Lowering:
    def __init__(self) -> None:
        self.scope: dict[str, dgen.Value] = {}
        self.last_effect: dgen.Op | None = None
        self.has_value_return: bool = False
        self.return_value: dgen.Value | None = None

    def lower_module(self, tm: ToyModule) -> Module:
        functions = [self.lower_function(f) for f in tm.functions]
        return Module(ops=functions)

    def lower_function(self, f: Function) -> function.FunctionOp:
        self.scope = {}
        self.last_effect = None
        self.has_value_return = False
        self.return_value = None

        # Create block args for function params
        args: list[BlockArgument] = []
        for param_name in f.proto.params:
            arg = BlockArgument(name=param_name, type=toy.InferredShapeTensor())
            self.scope[param_name] = arg
            args.append(arg)

        # Lower body statements
        ops = []
        for stmt in f.body:
            ops.extend(self._lower_statement(stmt))

        # Determine return type from AST-level return
        result: dgen.Type = builtin.Nil()
        if self.has_value_return:
            result = toy.InferredShapeTensor()

        if self.return_value is not None:
            block_result: dgen.Value = self.return_value
        elif ops:
            block_result = ops[-1]
        else:
            block_result = dgen.Value(type=builtin.Nil())
        return function.FunctionOp(
            name=f.proto.name,
            result=result,
            body=dgen.Block(result=block_result, args=args),
            type=FunctionType(result=result),
        )

    def _lower_statement(self, stmt: Statement) -> Iterator[dgen.Op]:
        if isinstance(stmt, VarDecl):
            yield from self._lower_var_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            yield from self._lower_return(stmt)
        elif isinstance(stmt, ExprStmt):
            yield from self.lower_expr(stmt.expr)

    def _lower_var_decl(self, decl: VarDecl) -> Iterator[dgen.Op]:
        expr_val = yield from self.lower_expr(decl.value)

        # If explicit shape, add a Reshape
        if decl.shape is not None:
            op = toy.ReshapeOp(
                input=expr_val,
                type=toy.Tensor(shape=shape_constant(list(decl.shape))),
            )
            yield op
            self.scope[decl.name] = op
        else:
            self.scope[decl.name] = expr_val

    def _lower_return(self, ret: ReturnStmt) -> Generator[dgen.Op, None, None]:
        if ret.value is not None:
            self.has_value_return = True
            val = yield from self.lower_expr(ret.value)
            if self.last_effect is not None:
                chain_op = builtin.ChainOp(lhs=val, rhs=self.last_effect, type=val.type)
                yield chain_op
                self.return_value = chain_op
            else:
                self.return_value = val
        elif self.last_effect is not None:
            self.return_value = self.last_effect

    def lower_expr(self, expr: Expression) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an expression, yielding ops and returning the result Value."""
        if isinstance(expr, NumberLiteral):
            op = ConstantOp(
                value=[expr.value],
                type=toy.Tensor(shape=shape_constant([1])),
            )
            yield op
            return op
        if isinstance(expr, TensorLiteral):
            op = ConstantOp(
                value=list(expr.values),
                type=toy.Tensor(shape=shape_constant(list(expr.shape))),
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
                result_op = toy.MulOp(lhs=lhs, rhs=rhs, type=toy.InferredShapeTensor())
            elif expr.op == "+":
                result_op = toy.AddOp(lhs=lhs, rhs=rhs, type=toy.InferredShapeTensor())
            else:
                raise RuntimeError(f"Unknown binary operator: {expr.op}")
            yield result_op
            return result_op
        if isinstance(expr, CallExpr):
            return (yield from self._lower_call(expr))
        raise RuntimeError("Unknown expression type")

    def _lower_index_expr(
        self, expr: Expression
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an expression that should produce an index value."""
        if isinstance(expr, NumberLiteral):
            op = ConstantOp(value=int(expr.value), type=index.Index())
            yield op
            return op
        return (yield from self.lower_expr(expr))

    # Builtin dispatch table: name → (arity, handler method name)
    _BUILTINS: dict[str, tuple[int, str]] = {
        "print": (1, "_lower_builtin_print"),
        "transpose": (1, "_lower_builtin_transpose"),
        "tile": (2, "_lower_builtin_tile"),
        "nonzero_count": (1, "_lower_builtin_nonzero_count"),
        "concat": (3, "_lower_builtin_concat"),
        "dim_size": (2, "_lower_builtin_dim_size"),
        "add_index": (2, "_lower_builtin_add_index"),
    }

    def _lower_builtin_transpose(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        arg = yield from self.lower_expr(call.args[0])
        op = toy.TransposeOp(input=arg, type=toy.InferredShapeTensor())
        yield op
        return op

    def _lower_builtin_tile(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        input_val = yield from self.lower_expr(call.args[0])
        count_val = yield from self._lower_index_expr(call.args[1])
        op = toy.TileOp(
            input=input_val, count=count_val, type=toy.InferredShapeTensor()
        )
        yield op
        return op

    def _lower_builtin_nonzero_count(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        arg = yield from self.lower_expr(call.args[0])
        op = toy.NonzeroCountOp(input=arg)
        yield op
        return op

    def _lower_builtin_concat(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        lhs = yield from self.lower_expr(call.args[0])
        rhs = yield from self.lower_expr(call.args[1])
        if not isinstance(call.args[2], NumberLiteral):
            raise RuntimeError("concat axis must be a literal")
        axis = index.Index().constant(int(call.args[2].value))
        op = toy.ConcatOp(axis=axis, lhs=lhs, rhs=rhs, type=toy.InferredShapeTensor())
        yield op
        return op

    def _lower_builtin_dim_size(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        input_val = yield from self.lower_expr(call.args[0])
        if not isinstance(call.args[1], NumberLiteral):
            raise RuntimeError("dim_size axis must be a literal")
        axis = index.Index().constant(int(call.args[1].value))
        op = toy.DimSizeOp(axis=axis, input=input_val)
        yield op
        return op

    def _lower_builtin_add_index(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        lhs = yield from self._lower_index_expr(call.args[0])
        rhs = yield from self._lower_index_expr(call.args[1])
        op = algebra.AddOp(left=lhs, right=rhs, type=index.Index())
        yield op
        return op

    def _lower_call(self, call: CallExpr) -> Generator[dgen.Op, None, dgen.Value]:
        entry = self._BUILTINS.get(call.callee)
        if entry is not None:
            arity, method_name = entry
            if len(call.args) != arity:
                raise RuntimeError(f"{call.callee} takes exactly {arity} argument(s)")
            handler = getattr(self, method_name)
            return (yield from handler(call))

        # Generic call
        args = []
        for a in call.args:
            args.append((yield from self.lower_expr(a)))
        callee_ref = dgen.Value(name=call.callee, type=builtin.Nil())
        p = pack(args)
        yield p
        op = function.CallOp(
            callee=callee_ref,
            arguments=p,
            type=toy.InferredShapeTensor(),
        )
        yield op
        return op

    def _lower_builtin_print(
        self, call: CallExpr
    ) -> Generator[dgen.Op, None, dgen.Value]:
        arg = yield from self.lower_expr(call.args[0])
        if self.last_effect is not None:
            chain_op = builtin.ChainOp(lhs=arg, rhs=self.last_effect, type=arg.type)
            yield chain_op
            print_input = chain_op
        else:
            print_input = arg
        print_op = toy.PrintOp(input=print_input)
        yield print_op
        self.last_effect = print_op
        return arg


def lower(tm: ToyModule) -> Module:
    lowering = Lowering()
    return lowering.lower_module(tm)
