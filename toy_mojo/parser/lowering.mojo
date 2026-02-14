"""AST to IR lowering for the Toy dialect."""

from collections import Optional, Dict

from toy.parser.ast import (
    AnyExpr, AnyStmt, NumberLiteral, TensorLiteral, VarRef,
    BinaryOp, CallExpr, PrintExpr, VarDecl, ReturnStmt, ExprStmt,
    Function, ToyModule, ExprArena,
)
from toy.dialects.toy_ops import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
)


fn unranked() -> AnyToyType:
    return AnyToyType(UnrankedTensorType())


fn ranked(var shape: List[Int]) -> AnyToyType:
    return AnyToyType(RankedTensorType(shape=shape^))


struct Lowering(Movable):
    var counter: Int
    var ops: List[AnyToyOp]
    var scope: Dict[String, String]
    var arena: ExprArena

    fn __init__(out self, var arena: ExprArena):
        self.counter = 0
        self.ops = List[AnyToyOp]()
        self.scope = Dict[String, String]()
        self.arena = arena^

    fn fresh(mut self) -> String:
        var name = String(self.counter)
        self.counter += 1
        return name^

    fn lower_module(mut self, tm: ToyModule) raises -> Module:
        var functions = List[FuncOp]()
        for i in range(len(tm.functions)):
            functions.append(self.lower_function(tm.functions[i]))
        return Module(functions=functions^)

    fn lower_function(mut self, f: Function) raises -> FuncOp:
        # Reset per-function state
        self.counter = 0
        self.ops = List[AnyToyOp]()
        self.scope = Dict[String, String]()

        # Create block args for function params
        var args = List[ToyValue]()
        var input_types = List[AnyToyType]()
        for i in range(len(f.proto.params)):
            var param_name = f.proto.params[i]
            self.scope[param_name] = param_name
            args.append(ToyValue(name=param_name, type=unranked()))
            input_types.append(unranked())

        # Lower body statements
        for i in range(len(f.body)):
            self.lower_statement(f.body[i])

        # Determine return type from ops
        var result = Optional[AnyToyType]()
        if len(self.ops) > 0:
            var last_op = self.ops[len(self.ops) - 1]
            if last_op.isa[ReturnOp]():
                if last_op[ReturnOp].value:
                    result = unranked()

        var ops = self.ops^
        self.ops = List[AnyToyOp]()
        var func_type = FunctionType(inputs=input_types^, result=result^)
        return FuncOp(
            name=f.proto.name,
            func_type=func_type^,
            body=Block(args=args^, ops=ops^),
        )

    fn lower_statement(mut self, stmt: AnyStmt) raises:
        if stmt.isa[VarDecl]():
            self._lower_var_decl(stmt[VarDecl])
        elif stmt.isa[ReturnStmt]():
            self._lower_return(stmt[ReturnStmt])
        elif stmt.isa[ExprStmt]():
            self._lower_expr_stmt(stmt[ExprStmt])

    fn _lower_var_decl(mut self, decl: VarDecl) raises:
        var expr_name = self.lower_expr(decl.value)

        # If explicit shape, add a Reshape
        if decl.shape:
            var shape = decl.shape.value().copy()
            var result = self.fresh()
            self.ops.append(AnyToyOp(ReshapeOp(
                result=result,
                input=expr_name,
                type=ranked(shape^),
            )))
            self.scope[decl.name] = result
        else:
            self.scope[decl.name] = expr_name

    fn _lower_return(mut self, ret: ReturnStmt) raises:
        if ret.value:
            var name = self.lower_expr(ret.value.value())
            self.ops.append(AnyToyOp(ReturnOp(value=name^)))
        else:
            self.ops.append(AnyToyOp(ReturnOp(value=Optional[String]())))

    fn _lower_expr_stmt(mut self, stmt: ExprStmt) raises:
        _ = self.lower_expr(stmt.expr)

    fn lower_expr(mut self, expr_idx: Int) raises -> String:
        """Lower an expression by arena index, return the SSA name of the result."""
        var expr = self.arena.get(expr_idx)
        if expr.isa[NumberLiteral]():
            return self._lower_number(expr[NumberLiteral])
        if expr.isa[TensorLiteral]():
            return self._lower_tensor(expr[TensorLiteral])
        if expr.isa[VarRef]():
            return self._lower_varref(expr[VarRef])
        if expr.isa[BinaryOp]():
            return self._lower_binop(expr[BinaryOp])
        if expr.isa[CallExpr]():
            return self._lower_call(expr[CallExpr])
        if expr.isa[PrintExpr]():
            return self._lower_print(expr[PrintExpr])
        raise Error("Unknown expression type")

    fn _lower_number(mut self, num: NumberLiteral) raises -> String:
        var result = self.fresh()
        self.ops.append(AnyToyOp(ConstantOp(
            result=result,
            value=[num.value],
            shape=[1],
            type=ranked([1]),
        )))
        return result^

    fn _lower_tensor(mut self, tensor: TensorLiteral) raises -> String:
        var result = self.fresh()
        self.ops.append(AnyToyOp(ConstantOp(
            result=result,
            value=tensor.values.copy(),
            shape=tensor.shape.copy(),
            type=ranked(tensor.shape.copy()),
        )))
        return result^

    fn _lower_varref(mut self, vr: VarRef) raises -> String:
        if vr.name not in self.scope:
            raise Error("Undefined variable: " + vr.name)
        return self.scope[vr.name]

    fn _lower_binop(mut self, op: BinaryOp) raises -> String:
        var lhs = self.lower_expr(op.lhs)
        var rhs = self.lower_expr(op.rhs)
        var result = self.fresh()
        if op.op == "*":
            self.ops.append(AnyToyOp(MulOp(
                result=result, lhs=lhs^, rhs=rhs^, type=unranked(),
            )))
        elif op.op == "+":
            self.ops.append(AnyToyOp(AddOp(
                result=result, lhs=lhs^, rhs=rhs^, type=unranked(),
            )))
        else:
            raise Error("Unknown binary operator: " + op.op)
        return result^

    fn _lower_call(mut self, call: CallExpr) raises -> String:
        # Builtin: transpose
        if call.callee == "transpose":
            if len(call.args) != 1:
                raise Error("transpose takes exactly 1 argument")
            var arg = self.lower_expr(call.args[0])
            var result = self.fresh()
            self.ops.append(AnyToyOp(TransposeOp(
                result=result, input=arg^, type=unranked(),
            )))
            return result^

        # Generic call
        var args = List[String]()
        for i in range(len(call.args)):
            args.append(self.lower_expr(call.args[i]))
        var result = self.fresh()
        self.ops.append(AnyToyOp(GenericCallOp(
            result=result, callee=call.callee, args=args^, type=unranked(),
        )))
        return result^

    fn _lower_print(mut self, p: PrintExpr) raises -> String:
        var arg = self.lower_expr(p.arg)
        self.ops.append(AnyToyOp(PrintOp(input=arg)))
        return arg


fn lower(tm: ToyModule) raises -> Module:
    var lowering = Lowering(tm.arena.copy())
    return lowering.lower_module(tm)
