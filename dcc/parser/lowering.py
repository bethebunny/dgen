"""Lower pycparser AST to dgen IR using the C dialect.

Thin, mechanical translation. Each pycparser node maps to C-dialect ops.
No memory ops, no type inference beyond what the C type resolver provides.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra
from dgen.dialects.builtin import Nil, String
from dgen.dialects.control_flow import IfOp, WhileOp
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, Module, pack
from dgen.dialects import function

from dcc.dialects import c_int
from dcc.dialects.c import (
    AddressOfOp,
    AssignOp,
    BreakOp,
    CallOp,
    ContinueOp,
    DereferenceOp,
    MemberAccessOp,
    ModuloOp,
    PointerMemberAccessOp,
    PostDecrementOp,
    PostIncrementOp,
    PreDecrementOp,
    PreIncrementOp,
    ReadVariableOp,
    ReturnOp,
    ShiftLeftOp,
    ShiftRightOp,
    SizeofOp,
    SubscriptOp,
    VariableDeclarationOp,
)
from dcc.parser.type_resolver import TypeResolver


class LoweringError(Exception):
    """Raised when the lowering encounters a C construct it cannot translate."""


# ---------------------------------------------------------------------------
# Binary op table
# ---------------------------------------------------------------------------

_ALGEBRA_OPS: dict[str, type[dgen.Op]] = {
    "+": algebra.AddOp,
    "-": algebra.SubtractOp,
    "*": algebra.MultiplyOp,
    "/": algebra.DivideOp,
    "&": algebra.MeetOp,
    "|": algebra.JoinOp,
    "^": algebra.SymmetricDifferenceOp,
    "==": algebra.EqualOp,
    "!=": algebra.NotEqualOp,
    "<": algebra.LessThanOp,
    "<=": algebra.LessEqualOp,
    ">": algebra.GreaterThanOp,
    ">=": algebra.GreaterEqualOp,
    "&&": algebra.MeetOp,
    "||": algebra.JoinOp,
}

_C_OPS: dict[str, type[dgen.Op]] = {
    "%": ModuloOp,
    "<<": ShiftLeftOp,
    ">>": ShiftRightOp,
}

_COMPARISONS: set[str] = {"==", "!=", "<", "<=", ">", ">="}

_COMPOUND_ASSIGN: dict[str, str] = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}


def _make_binop(
    op_cls: type[dgen.Op], left: dgen.Value, right: dgen.Value, result_type: dgen.Type
) -> dgen.Op:
    """Construct a binary op — algebra uses left/right, C ops use lhs/rhs."""
    if "left" in op_cls.__dataclass_fields__:
        return op_cls(left=left, right=right, type=result_type)
    return op_cls(lhs=left, rhs=right, type=result_type)


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------


def _closed_block(
    result: dgen.Value,
    args: list[BlockArgument] | None = None,
    *,
    local_ops: list[dgen.Op] | None = None,
) -> dgen.Block:
    """Build a Block, computing captures from local_ops."""
    if args is None:
        args = []
    if local_ops is None:
        local_ops = []
    local_ids: set[int] = {id(op) for op in local_ops} | {id(a) for a in args}
    captures: list[dgen.Value] = []
    seen: set[int] = set()

    def _maybe_capture(dep: dgen.Value) -> None:
        vid = id(dep)
        if vid in seen or vid in local_ids or isinstance(dep, dgen.Type):
            return
        seen.add(vid)
        captures.append(dep)

    if id(result) not in local_ids and not isinstance(result, dgen.Type):
        _maybe_capture(result)
    for op in local_ops:
        for dep in op.dependencies:
            _maybe_capture(dep)
    return dgen.Block(result=result, args=args, captures=captures)


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


@dataclass
class LoweringStats:
    functions: int = 0
    typedefs: int = 0
    statements: int = 0
    expressions: int = 0
    skipped_functions: int = 0

    def summary(self) -> str:
        return (
            f"Functions: {self.functions}, Typedefs: {self.typedefs}, "
            f"Statements: {self.statements}, Expressions: {self.expressions}, "
            f"Skipped functions: {self.skipped_functions}"
        )


class Lowering:
    """Lower a pycparser FileAST to a dgen Module."""

    def __init__(self) -> None:
        self.types = TypeResolver()
        self.scope: dict[str, dgen.Value] = {}
        self.func_types: dict[str, dgen.Type] = {}
        self.current_return_type: dgen.Type = Nil()
        self.functions: list[function.FunctionOp] = []
        self.stats = LoweringStats()

    # --- Top level ---

    def lower_file(self, ast: c_ast.FileAST) -> Module:
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef):
                self._register_typedef(ext)
            elif isinstance(ext, c_ast.Decl):
                if isinstance(ext.type, c_ast.FuncDecl):
                    self._register_function(ext)
                elif isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                    self.types.resolve(ext.type)

        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                self.stats.functions += 1
                try:
                    self.functions.append(self._lower_function(ext))
                except LoweringError:
                    self.stats.skipped_functions += 1

        return Module(ops=list(self.functions))

    def _register_typedef(self, node: c_ast.Typedef) -> None:
        if node.name is not None:
            self.types.register_typedef(node.name, self.types.resolve(node.type))
            self.stats.typedefs += 1

    def _register_function(self, node: c_ast.Decl) -> None:
        if node.name is not None:
            self.func_types[node.name] = self._return_type(node.type)

    def _return_type(self, node: c_ast.Node) -> dgen.Type:
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return Reference(element_type=self._return_type(node.type))
        return self.types.resolve(node)

    # --- Functions ---

    def _lower_function(self, node: c_ast.FuncDef) -> function.FunctionOp:
        self.scope = {}
        name = node.decl.name
        return_type = self._return_type(node.decl.type)
        self.func_types[name] = return_type
        self.current_return_type = return_type

        args: list[BlockArgument] = []
        if isinstance(node.decl.type, c_ast.FuncDecl) and node.decl.type.args:
            for param in node.decl.type.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl) and param.name:
                    arg = BlockArgument(
                        name=param.name, type=self.types.resolve(param.type)
                    )
                    self.scope[param.name] = arg
                    args.append(arg)

        ops = list(self._compound(node.body)) if node.body else []
        result = ops[-1] if ops else dgen.Value(type=Nil())
        is_void = isinstance(return_type, Nil)

        return function.FunctionOp(
            name=name,
            result=Nil() if is_void else return_type,
            body=dgen.Block(result=result, args=args),
            type=FunctionType(result=Nil() if is_void else return_type),
        )

    # --- Statements ---

    def _compound(self, node: c_ast.Compound) -> Iterator[dgen.Op]:
        if node.block_items is None:
            return
        for item in node.block_items:
            yield from self._statement(item)

    def _statement(self, node: c_ast.Node) -> Iterator[dgen.Op]:
        self.stats.statements += 1

        if isinstance(node, c_ast.Decl):
            yield from self._declaration(node)
        elif isinstance(node, c_ast.Assignment):
            yield from self._assignment(node)
        elif isinstance(node, c_ast.Return):
            yield from self._return(node)
        elif isinstance(node, c_ast.If):
            yield from self._if(node)
        elif isinstance(node, c_ast.While):
            yield from self._while(node)
        elif isinstance(node, c_ast.DoWhile):
            yield from self._do_while(node)
        elif isinstance(node, c_ast.For):
            yield from self._for(node)
        elif isinstance(node, c_ast.Compound):
            yield from self._compound(node)
        elif isinstance(node, c_ast.FuncCall):
            yield from self._expression(node)
        elif isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "p--", "++", "--"):
            yield from self._expression(node)
        elif isinstance(node, c_ast.Goto):
            pass  # C goto: skip (unstructured control flow)
        elif isinstance(node, c_ast.Label):
            if node.stmt is not None:
                yield from self._statement(node.stmt)
        elif isinstance(node, c_ast.Switch):
            yield from self._expression(node.cond)
            if node.stmt is not None:
                yield from self._statement(node.stmt)
        elif isinstance(node, c_ast.Break):
            yield BreakOp()
        elif isinstance(node, c_ast.Continue):
            yield ContinueOp()
        elif isinstance(node, c_ast.Case):
            if node.stmts:
                for s in node.stmts:
                    yield from self._statement(s)
        elif isinstance(node, c_ast.Default):
            if node.stmts:
                for s in node.stmts:
                    yield from self._statement(s)
        elif isinstance(node, c_ast.EmptyStatement):
            pass
        elif isinstance(node, c_ast.Typedef):
            self._register_typedef(node)
        elif isinstance(node, c_ast.Pragma):
            pass
        elif isinstance(node, c_ast.DeclList):
            for decl in node.decls:
                yield from self._declaration(decl)
        else:
            yield from self._expression(node)

    def _declaration(self, node: c_ast.Decl) -> Iterator[dgen.Op]:
        if node.name is None:
            if isinstance(node.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                self.types.resolve(node.type)
            return
        var_type = self.types.resolve(node.type)
        if isinstance(node.type, c_ast.FuncDecl):
            self.func_types[node.name] = self._return_type(node.type)
            return

        if node.init is not None:
            init = yield from self._expression(node.init, target_type=var_type)
        else:
            init = ConstantOp(value=0, type=var_type)
            yield init

        decl = VariableDeclarationOp(
            variable_name=String().constant(node.name),
            variable_type=var_type,
            initializer=init,
            type=var_type,
        )
        yield decl
        self.scope[node.name] = decl

    def _assignment(self, node: c_ast.Assignment) -> Iterator[dgen.Op]:
        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(f"unsupported lvalue: {type(node.lvalue).__name__}")
        name = node.lvalue.name
        if name not in self.scope:
            raise LoweringError(f"undefined variable: {name}")
        target = self.scope[name]
        rhs = yield from self._expression(node.rvalue)

        if node.op != "=":
            base_op = _COMPOUND_ASSIGN.get(node.op)
            if base_op is not None:
                current = ReadVariableOp(
                    variable_name=String().constant(name),
                    source=target,
                    type=target.type,
                )
                yield current
                op_cls = _ALGEBRA_OPS.get(base_op) or _C_OPS.get(base_op)
                if op_cls is not None:
                    rhs = _make_binop(op_cls, current, rhs, current.type)
                    yield rhs

        assign = AssignOp(
            variable_name=String().constant(name),
            target=target,
            value=rhs,
            type=target.type,
        )
        yield assign
        self.scope[name] = assign

    def _return(self, node: c_ast.Return) -> Iterator[dgen.Op]:
        if node.expr is None:
            nil = ConstantOp(value=None, type=Nil())
            yield nil
            yield ReturnOp(value=nil)
        else:
            val = yield from self._expression(
                node.expr, target_type=self.current_return_type
            )
            yield ReturnOp(value=val)

    # --- Control flow ---

    def _if(self, node: c_ast.If) -> Iterator[dgen.Op]:
        cond = yield from self._expression(node.cond)
        then_ops = list(self._statement(node.iftrue))
        then_result = then_ops[-1] if then_ops else dgen.Value(type=Nil())
        else_ops = list(self._statement(node.iffalse)) if node.iffalse else []
        else_result = else_ops[-1] if else_ops else dgen.Value(type=Nil())
        empty = pack([])
        yield empty
        yield IfOp(
            condition=cond,
            then_arguments=empty,
            else_arguments=empty,
            type=Nil(),
            then_body=_closed_block(then_result, local_ops=then_ops),
            else_body=_closed_block(else_result, local_ops=else_ops),
        )

    def _while(self, node: c_ast.While) -> Iterator[dgen.Op]:
        cond_ops = list(self._expression(node.cond))
        cond = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())
        body_ops = list(self._statement(node.stmt))
        body_result = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        p = pack([])
        yield p
        yield WhileOp(
            initial_arguments=p,
            condition=_closed_block(cond, local_ops=cond_ops),
            body=_closed_block(body_result, local_ops=body_ops),
        )

    def _do_while(self, node: c_ast.DoWhile) -> Iterator[dgen.Op]:
        body_ops = list(self._statement(node.stmt))
        body_result = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        cond_ops = list(self._expression(node.cond))
        cond = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())
        from dcc.dialects.c import DoWhileOp

        p = pack([])
        yield p
        yield DoWhileOp(
            initial=p,
            body=_closed_block(body_result, local_ops=body_ops),
            condition=_closed_block(cond, local_ops=cond_ops),
        )

    def _for(self, node: c_ast.For) -> Iterator[dgen.Op]:
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                for decl in node.init.decls:
                    yield from self._declaration(decl)
            else:
                yield from self._statement(node.init)
        if node.cond is not None:
            cond_ops = list(self._expression(node.cond))
            cond = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())
        else:
            cond = ConstantOp(value=1, type=c_int(32))
            cond_ops = [cond]
        body_ops = list(self._statement(node.stmt))
        if node.next is not None:
            body_ops.extend(self._statement(node.next))
        body_result = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        p = pack([])
        yield p
        yield WhileOp(
            initial_arguments=p,
            condition=_closed_block(cond, local_ops=cond_ops),
            body=_closed_block(body_result, local_ops=body_ops),
        )

    # --- Expressions ---

    def _expression(
        self, node: c_ast.Node, target_type: dgen.Type | None = None
    ) -> Iterator[dgen.Op]:
        """Lower an expression. target_type only affects literal constants."""
        self.stats.expressions += 1

        if isinstance(node, c_ast.Constant):
            return (yield from self._constant(node, target_type))
        if isinstance(node, c_ast.ID):
            return (yield from self._identifier(node))
        if isinstance(node, c_ast.BinaryOp):
            return (yield from self._binary(node))
        if isinstance(node, c_ast.UnaryOp):
            return (yield from self._unary(node))
        if isinstance(node, c_ast.FuncCall):
            return (yield from self._call(node))
        if isinstance(node, c_ast.Assignment):
            yield from self._assignment(node)
            name = node.lvalue.name if isinstance(node.lvalue, c_ast.ID) else None
            if name and name in self.scope:
                val = self.scope[name]
                read = ReadVariableOp(
                    variable_name=String().constant(name), source=val, type=val.type
                )
                yield read
                return read
            raise LoweringError("assign-as-expression on non-variable")
        if isinstance(node, c_ast.Cast):
            inner = yield from self._expression(node.expr)
            target = self.types.resolve(node.to_type)
            op = algebra.CastOp(input=inner, type=target)
            yield op
            return op
        if isinstance(node, c_ast.ArrayRef):
            base = yield from self._expression(node.name)
            idx = yield from self._expression(node.subscript)
            pointee = self._pointee_type(base.type)
            op = SubscriptOp(base=base, index=idx, type=pointee)
            yield op
            return op
        if isinstance(node, c_ast.StructRef):
            return (yield from self._struct_access(node))
        if isinstance(node, c_ast.TernaryOp):
            cond = yield from self._expression(node.cond)
            true_ops = list(self._expression(node.iftrue))
            true_val = true_ops[-1] if true_ops else dgen.Value(type=Nil())
            false_ops = list(self._expression(node.iffalse))
            false_val = false_ops[-1] if false_ops else dgen.Value(type=Nil())
            empty = pack([])
            yield empty
            op = IfOp(
                condition=cond,
                then_arguments=empty,
                else_arguments=empty,
                type=true_val.type,
                then_body=_closed_block(true_val, local_ops=true_ops),
                else_body=_closed_block(false_val, local_ops=false_ops),
            )
            yield op
            return op
        if isinstance(node, c_ast.ExprList):
            result = dgen.Value(type=Nil())
            for expr in node.exprs:
                result = yield from self._expression(expr)
            return result
        if isinstance(node, c_ast.CompoundLiteral):
            return (yield from self._expression(node.init))
        if isinstance(node, c_ast.InitList):
            if node.exprs:
                return (yield from self._expression(node.exprs[0]))
            raise LoweringError("empty initializer list")

        raise LoweringError(f"unsupported expression: {type(node).__name__}")

    # --- Expression helpers ---

    def _constant(
        self, node: c_ast.Constant, target_type: dgen.Type | None = None
    ) -> Iterator[dgen.Op]:
        if node.type == "int":
            s = node.value.rstrip("uUlL")
            val = int(s, 0)  # handles 0x, 0o, decimal
            if target_type is not None:
                ty = target_type
            elif any(c in node.value[len(s) :].lower() for c in ("ll", "l")):
                ty = c_int(64, signed="u" not in node.value[len(s) :].lower())
            elif "u" in node.value[len(s) :].lower():
                ty = c_int(32, signed=False)
            else:
                ty = c_int(32)
            op = ConstantOp(value=val, type=ty)
            yield op
            return op
        if node.type in ("float", "double"):
            val = float(node.value.rstrip("fFlL"))
            op = ConstantOp(value=val, type=Float64())
            yield op
            return op
        if node.type == "char":
            ch = node.value[1:-1]
            if ch.startswith("\\"):
                escapes = {
                    "n": 10,
                    "t": 9,
                    "r": 13,
                    "0": 0,
                    "\\": 92,
                    "'": 39,
                    '"': 34,
                    "a": 7,
                    "b": 8,
                    "f": 12,
                }
                val = escapes.get(ch[1], ord(ch[1]))
            else:
                val = ord(ch)
            op = ConstantOp(value=val, type=c_int(8))
            yield op
            return op
        if node.type == "string":
            op = ConstantOp(value=0, type=Reference(element_type=c_int(8)))
            yield op
            return op
        raise LoweringError(f"unsupported constant type: {node.type}")

    def _identifier(self, node: c_ast.ID) -> Iterator[dgen.Op]:
        if node.name in self.types.enum_constants:
            op = ConstantOp(value=self.types.enum_constants[node.name], type=c_int(32))
            yield op
            return op
        if node.name in self.scope:
            val = self.scope[node.name]
            if isinstance(val, (VariableDeclarationOp, AssignOp)):
                read = ReadVariableOp(
                    variable_name=String().constant(node.name),
                    source=val,
                    type=val.type,
                )
                yield read
                return read
            return val
        raise LoweringError(f"undefined identifier: {node.name}")

    def _binary(self, node: c_ast.BinaryOp) -> Iterator[dgen.Op]:
        left = yield from self._expression(node.left)
        right = yield from self._expression(node.right)
        # Null pointer matching: ptr == 0 → ptr == null
        if isinstance(left.type, Reference) and isinstance(right, ConstantOp):
            right = ConstantOp(value=0, type=left.type)
            yield right
        elif isinstance(right.type, Reference) and isinstance(left, ConstantOp):
            left = ConstantOp(value=0, type=right.type)
            yield left
        op_cls = _ALGEBRA_OPS.get(node.op) or _C_OPS.get(node.op)
        if op_cls is None:
            raise LoweringError(f"unsupported binary operator: {node.op}")
        result_type = self._promote(left.type, right.type)
        op = _make_binop(op_cls, left, right, result_type)
        yield op
        if node.op in _COMPARISONS:
            cast = algebra.CastOp(input=op, type=c_int(32))
            yield cast
            return cast
        return op

    def _unary(self, node: c_ast.UnaryOp) -> Iterator[dgen.Op]:
        if node.op == "sizeof":
            if isinstance(node.expr, c_ast.Typename):
                target = self.types.resolve(node.expr)
            else:
                target = c_int(32)  # simplified
            op = SizeofOp(target_type=target, type=c_int(64, signed=False))
            yield op
            return op
        if node.op == "&":
            operand = yield from self._expression(node.expr)
            op = AddressOfOp(operand=operand, type=Reference(element_type=operand.type))
            yield op
            return op
        if node.op == "*":
            inner = yield from self._expression(node.expr)
            op = DereferenceOp(pointer=inner, type=self._pointee_type(inner.type))
            yield op
            return op
        if node.op in ("++", "p++", "--", "p--"):
            if not isinstance(node.expr, c_ast.ID):
                raise LoweringError("increment/decrement on non-variable")
            name = node.expr.name
            if name not in self.scope:
                raise LoweringError(f"undefined variable: {name}")
            target = self.scope[name]
            op_map = {
                "++": PreIncrementOp,
                "p++": PostIncrementOp,
                "--": PreDecrementOp,
                "p--": PostDecrementOp,
            }
            op = op_map[node.op](
                variable_name=String().constant(name), target=target, type=target.type
            )
            yield op
            return op
        inner = yield from self._expression(node.expr)
        if node.op == "-":
            op = algebra.NegateOp(input=inner, type=inner.type)
        elif node.op == "~":
            op = algebra.ComplementOp(input=inner, type=inner.type)
        elif node.op == "!":
            zero = ConstantOp(value=0, type=inner.type)
            yield zero
            eq = algebra.EqualOp(left=inner, right=zero, type=inner.type)
            yield eq
            op = algebra.CastOp(input=eq, type=c_int(32))
        elif node.op == "+":
            return inner
        else:
            raise LoweringError(f"unsupported unary operator: {node.op}")
        yield op
        return op

    def _call(self, node: c_ast.FuncCall) -> Iterator[dgen.Op]:
        if not isinstance(node.name, c_ast.ID):
            raise LoweringError("indirect function calls not yet supported")
        callee = node.name.name
        return_type = self.func_types.get(callee, c_int(32))
        args: list[dgen.Value] = []
        if node.args is not None:
            for arg in node.args.exprs:
                args.append((yield from self._expression(arg)))
        p = pack(args) if args else pack([])
        yield p
        op = CallOp(callee=String().constant(callee), arguments=p, type=return_type)
        yield op
        return op

    def _struct_access(self, node: c_ast.StructRef) -> Iterator[dgen.Op]:
        base = yield from self._expression(node.name)
        field_name = node.field.name
        if node.type == "->":
            field_type = self.types.get_struct_field_type(
                self._pointee_type(base.type), field_name
            )
            op = PointerMemberAccessOp(
                field_name=String().constant(field_name), base=base, type=field_type
            )
        else:
            field_type = self.types.get_struct_field_type(base.type, field_name)
            op = MemberAccessOp(
                field_name=String().constant(field_name), base=base, type=field_type
            )
        yield op
        return op

    # --- Type helpers (minimal) ---

    def _pointee_type(self, ty: dgen.Type) -> dgen.Type:
        if isinstance(ty, Reference):
            elem = ty.element_type
            if isinstance(elem, dgen.Type):
                return elem
        raise LoweringError(f"cannot dereference non-pointer type: {ty}")

    def _promote(self, a: dgen.Type, b: dgen.Type) -> dgen.Type:
        if isinstance(a, Float64) or isinstance(b, Float64):
            return Float64()
        if isinstance(a, Reference):
            return a
        if isinstance(b, Reference):
            return b
        return a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(ast: c_ast.FileAST) -> tuple[Module, LoweringStats]:
    lowering = Lowering()
    module = lowering.lower_file(ast)
    return module, lowering.stats
