"""Parse C AST into dgen IR.

Thin translation from pycparser AST to C-dialect ops. Name resolution
uses a Scope chain threaded through every method — no mutable self state
for scoping. Semantic transformations (type promotion, implicit casts,
compound assignment expansion) belong in passes, not here.
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
    CompoundAssignOp,
    ContinueOp,
    DereferenceOp,
    LogicalNotOp,
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
    """A C construct the parser cannot translate."""


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope:
    """Lexical scope for C name resolution.

    Each compound statement ({}) creates a child scope. Variable
    declarations bind in the current scope. Lookups walk the parent
    chain. Captures track cross-scope references for block construction.
    """

    def __init__(self, parent: Scope | None = None) -> None:
        self._bindings: dict[str, dgen.Value] = {}
        self._parent = parent
        self.captures: list[dgen.Value] = []

    def bind(self, name: str, value: dgen.Value) -> None:
        self._bindings[name] = value

    def lookup(self, name: str) -> dgen.Value:
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            val = self._parent.lookup(name)
            if val not in self.captures:
                self.captures.append(val)
            return val
        raise LoweringError(f"undefined: {name}")

    def has(self, name: str) -> bool:
        if name in self._bindings:
            return True
        return self._parent.has(name) if self._parent is not None else False

    def child(self) -> Scope:
        return Scope(parent=self)


# ---------------------------------------------------------------------------
# Binary op tables
# ---------------------------------------------------------------------------

_ALGEBRA: dict[str, type[dgen.Op]] = {
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

_C_BINOPS: dict[str, type[dgen.Op]] = {
    "%": ModuloOp,
    "<<": ShiftLeftOp,
    ">>": ShiftRightOp,
}

_ALL_BINOPS = {**_ALGEBRA, **_C_BINOPS}

_COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}

_COMPOUND_OPS = {
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

_INCREMENTS = {
    "++": PreIncrementOp,
    "p++": PostIncrementOp,
    "--": PreDecrementOp,
    "p--": PostDecrementOp,
}


def _binop(cls: type[dgen.Op], a: dgen.Value, b: dgen.Value, ty: dgen.Type) -> dgen.Op:
    if "left" in cls.__dataclass_fields__:
        return cls(left=a, right=b, type=ty)
    return cls(lhs=a, rhs=b, type=ty)


# ---------------------------------------------------------------------------
# Stats
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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Translate pycparser AST to C-dialect IR."""

    def __init__(self) -> None:
        self.types = TypeResolver()
        self.file_scope = Scope()
        self.stats = LoweringStats()

    def parse(self, ast: c_ast.FileAST) -> Module:
        # First pass: register types and function declarations
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef):
                if ext.name is not None:
                    self.types.register_typedef(ext.name, self.types.resolve(ext.type))
                    self.stats.typedefs += 1
            elif isinstance(ext, c_ast.Decl):
                if isinstance(ext.type, c_ast.FuncDecl) and ext.name:
                    self.file_scope.bind(
                        ext.name,
                        dgen.Value(name=ext.name, type=self._return_type(ext.type)),
                    )
                elif isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                    self.types.resolve(ext.type)

        # Second pass: lower function definitions
        functions: list[function.FunctionOp] = []
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                self.stats.functions += 1
                try:
                    functions.append(self._function(ext))
                except LoweringError:
                    self.stats.skipped_functions += 1

        return Module(ops=functions)

    def _return_type(self, node: c_ast.Node) -> dgen.Type:
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return Reference(element_type=self._return_type(node.type))
        return self.types.resolve(node)

    # --- Functions ---

    def _function(self, node: c_ast.FuncDef) -> function.FunctionOp:
        scope = self.file_scope.child()
        name = node.decl.name
        ret = self._return_type(node.decl.type)
        self.file_scope.bind(name, dgen.Value(name=name, type=ret))

        args: list[BlockArgument] = []
        if isinstance(node.decl.type, c_ast.FuncDecl) and node.decl.type.args:
            for param in node.decl.type.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl) and param.name:
                    arg = BlockArgument(
                        name=param.name, type=self.types.resolve(param.type)
                    )
                    scope.bind(param.name, arg)
                    args.append(arg)

        ops = list(self._compound(node.body, scope)) if node.body else []
        result = ops[-1] if ops else dgen.Value(type=Nil())
        is_void = isinstance(ret, Nil)

        return function.FunctionOp(
            name=name,
            result=Nil() if is_void else ret,
            body=dgen.Block(result=result, args=args),
            type=FunctionType(result=Nil() if is_void else ret),
        )

    # --- Statements ---

    def _compound(self, node: c_ast.Compound, scope: Scope) -> Iterator[dgen.Op]:
        child = scope.child()
        if node.block_items:
            for item in node.block_items:
                yield from self._stmt(item, child)

    def _stmt(self, node: c_ast.Node, scope: Scope) -> Iterator[dgen.Op]:
        self.stats.statements += 1
        if isinstance(node, c_ast.Decl):
            yield from self._decl(node, scope)
        elif isinstance(node, c_ast.Assignment):
            yield from self._assign(node, scope)
        elif isinstance(node, c_ast.Return):
            yield from self._ret(node, scope)
        elif isinstance(node, c_ast.If):
            yield from self._if(node, scope)
        elif isinstance(node, c_ast.While):
            yield from self._while(node, scope)
        elif isinstance(node, c_ast.DoWhile):
            yield from self._do_while(node, scope)
        elif isinstance(node, c_ast.For):
            yield from self._for(node, scope)
        elif isinstance(node, c_ast.Compound):
            yield from self._compound(node, scope)
        elif isinstance(node, c_ast.FuncCall):
            yield from self._expr(node, scope)
        elif isinstance(node, c_ast.UnaryOp) and node.op in _INCREMENTS:
            yield from self._expr(node, scope)
        elif isinstance(node, c_ast.Goto):
            pass
        elif isinstance(node, c_ast.Label):
            if node.stmt:
                yield from self._stmt(node.stmt, scope)
        elif isinstance(node, c_ast.Switch):
            yield from self._expr(node.cond, scope)
            if node.stmt:
                yield from self._stmt(node.stmt, scope)
        elif isinstance(node, c_ast.Break):
            yield BreakOp()
        elif isinstance(node, c_ast.Continue):
            yield ContinueOp()
        elif isinstance(node, c_ast.Case):
            if node.stmts:
                for s in node.stmts:
                    yield from self._stmt(s, scope)
        elif isinstance(node, c_ast.Default):
            if node.stmts:
                for s in node.stmts:
                    yield from self._stmt(s, scope)
        elif isinstance(node, c_ast.EmptyStatement):
            pass
        elif isinstance(node, c_ast.Typedef):
            if node.name:
                self.types.register_typedef(node.name, self.types.resolve(node.type))
                self.stats.typedefs += 1
        elif isinstance(node, c_ast.Pragma):
            pass
        elif isinstance(node, c_ast.DeclList):
            for decl in node.decls:
                yield from self._decl(decl, scope)
        else:
            yield from self._expr(node, scope)

    def _decl(self, node: c_ast.Decl, scope: Scope) -> Iterator[dgen.Op]:
        if node.name is None:
            if isinstance(node.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                self.types.resolve(node.type)
            return
        if isinstance(node.type, c_ast.FuncDecl):
            self.file_scope.bind(
                node.name, dgen.Value(name=node.name, type=self._return_type(node.type))
            )
            return
        var_type = self.types.resolve(node.type)
        if node.init is not None:
            init = yield from self._expr(node.init, scope, target_type=var_type)
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
        scope.bind(node.name, decl)

    def _assign(self, node: c_ast.Assignment, scope: Scope) -> Iterator[dgen.Op]:
        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(f"unsupported lvalue: {type(node.lvalue).__name__}")
        name = node.lvalue.name
        target = scope.lookup(name)

        if node.op in _COMPOUND_OPS:
            rhs = yield from self._expr(node.rvalue, scope)
            op = CompoundAssignOp(
                variable_name=String().constant(name),
                operator=String().constant(_COMPOUND_OPS[node.op]),
                target=target,
                operand=rhs,
                type=target.type,
            )
            yield op
            scope.bind(name, op)
        else:
            rhs = yield from self._expr(node.rvalue, scope)
            op = AssignOp(
                variable_name=String().constant(name),
                target=target,
                value=rhs,
                type=target.type,
            )
            yield op
            scope.bind(name, op)

    def _ret(self, node: c_ast.Return, scope: Scope) -> Iterator[dgen.Op]:
        if node.expr is None:
            nil = ConstantOp(value=None, type=Nil())
            yield nil
            yield ReturnOp(value=nil)
        else:
            val = yield from self._expr(node.expr, scope)
            yield ReturnOp(value=val)

    # --- Control flow ---

    def _block_from(self, stmts: Iterator[dgen.Op], scope: Scope) -> dgen.Block:
        ops = list(stmts)
        result = ops[-1] if ops else dgen.Value(type=Nil())
        return dgen.Block(result=result, captures=list(scope.captures))

    def _if(self, node: c_ast.If, scope: Scope) -> Iterator[dgen.Op]:
        cond = yield from self._expr(node.cond, scope)
        then_scope = scope.child()
        then_block = self._block_from(self._stmt(node.iftrue, then_scope), then_scope)
        else_scope = scope.child()
        if node.iffalse:
            else_block = self._block_from(
                self._stmt(node.iffalse, else_scope), else_scope
            )
        else:
            else_block = dgen.Block(result=dgen.Value(type=Nil()))
        empty = pack([])
        yield empty
        yield IfOp(
            condition=cond,
            then_arguments=empty,
            else_arguments=empty,
            type=Nil(),
            then_body=then_block,
            else_body=else_block,
        )

    def _while(self, node: c_ast.While, scope: Scope) -> Iterator[dgen.Op]:
        cond_scope = scope.child()
        cond_block = self._block_from(self._expr(node.cond, cond_scope), cond_scope)
        body_scope = scope.child()
        body_block = self._block_from(self._stmt(node.stmt, body_scope), body_scope)
        p = pack([])
        yield p
        yield WhileOp(initial_arguments=p, condition=cond_block, body=body_block)

    def _do_while(self, node: c_ast.DoWhile, scope: Scope) -> Iterator[dgen.Op]:
        from dcc.dialects.c import DoWhileOp

        body_scope = scope.child()
        body_block = self._block_from(self._stmt(node.stmt, body_scope), body_scope)
        cond_scope = scope.child()
        cond_block = self._block_from(self._expr(node.cond, cond_scope), cond_scope)
        p = pack([])
        yield p
        yield DoWhileOp(initial=p, body=body_block, condition=cond_block)

    def _for(self, node: c_ast.For, scope: Scope) -> Iterator[dgen.Op]:
        # init runs in the parent scope
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                for decl in node.init.decls:
                    yield from self._decl(decl, scope)
            else:
                yield from self._stmt(node.init, scope)
        # condition and body in child scopes
        if node.cond is not None:
            cond_scope = scope.child()
            cond_block = self._block_from(self._expr(node.cond, cond_scope), cond_scope)
        else:
            one = ConstantOp(value=1, type=c_int(32))
            cond_block = dgen.Block(result=one)
        body_scope = scope.child()
        body_stmts = list(self._stmt(node.stmt, body_scope))
        if node.next is not None:
            body_stmts.extend(self._stmt(node.next, body_scope))
        body_result = body_stmts[-1] if body_stmts else dgen.Value(type=Nil())
        body_block = dgen.Block(result=body_result, captures=list(body_scope.captures))
        p = pack([])
        yield p
        yield WhileOp(initial_arguments=p, condition=cond_block, body=body_block)

    # --- Expressions ---

    def _expr(
        self,
        node: c_ast.Node,
        scope: Scope,
        target_type: dgen.Type | None = None,
    ) -> Iterator[dgen.Op]:
        self.stats.expressions += 1
        if isinstance(node, c_ast.Constant):
            return (yield from self._constant(node, target_type))
        if isinstance(node, c_ast.ID):
            return (yield from self._id(node, scope))
        if isinstance(node, c_ast.BinaryOp):
            return (yield from self._binary(node, scope))
        if isinstance(node, c_ast.UnaryOp):
            return (yield from self._unary(node, scope))
        if isinstance(node, c_ast.FuncCall):
            return (yield from self._call(node, scope))
        if isinstance(node, c_ast.Assignment):
            yield from self._assign(node, scope)
            name = node.lvalue.name if isinstance(node.lvalue, c_ast.ID) else None
            if name and scope.has(name):
                val = scope.lookup(name)
                read = ReadVariableOp(
                    variable_name=String().constant(name),
                    source=val,
                    type=val.type,
                )
                yield read
                return read
            raise LoweringError("assign-as-expression on non-variable")
        if isinstance(node, c_ast.Cast):
            inner = yield from self._expr(node.expr, scope)
            target = self.types.resolve(node.to_type)
            op = algebra.CastOp(input=inner, type=target)
            yield op
            return op
        if isinstance(node, c_ast.ArrayRef):
            base = yield from self._expr(node.name, scope)
            idx = yield from self._expr(node.subscript, scope)
            pointee = self._pointee(base.type)
            op = SubscriptOp(base=base, index=idx, type=pointee)
            yield op
            return op
        if isinstance(node, c_ast.StructRef):
            return (yield from self._struct(node, scope))
        if isinstance(node, c_ast.TernaryOp):
            cond = yield from self._expr(node.cond, scope)
            then_scope = scope.child()
            then_block = self._block_from(
                self._expr(node.iftrue, then_scope), then_scope
            )
            else_scope = scope.child()
            else_block = self._block_from(
                self._expr(node.iffalse, else_scope), else_scope
            )
            empty = pack([])
            yield empty
            op = IfOp(
                condition=cond,
                then_arguments=empty,
                else_arguments=empty,
                type=then_block.result.type,
                then_body=then_block,
                else_body=else_block,
            )
            yield op
            return op
        if isinstance(node, c_ast.ExprList):
            result = dgen.Value(type=Nil())
            for expr in node.exprs:
                result = yield from self._expr(expr, scope)
            return result
        if isinstance(node, c_ast.CompoundLiteral):
            return (yield from self._expr(node.init, scope))
        if isinstance(node, c_ast.InitList):
            if node.exprs:
                return (yield from self._expr(node.exprs[0], scope))
            raise LoweringError("empty initializer list")
        raise LoweringError(f"unsupported expression: {type(node).__name__}")

    # --- Expression helpers ---

    def _constant(
        self, node: c_ast.Constant, target_type: dgen.Type | None = None
    ) -> Iterator[dgen.Op]:
        if node.type == "int":
            s = node.value.rstrip("uUlL")
            # C octal: 0644 → Python needs 0o644
            if len(s) > 1 and s[0] == "0" and s[1:].isdigit():
                val = int(s, 8)
            else:
                val = int(s, 0)
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
            op = ConstantOp(value=float(node.value.rstrip("fFlL")), type=Float64())
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

    def _id(self, node: c_ast.ID, scope: Scope) -> Iterator[dgen.Op]:
        if node.name in self.types.enum_constants:
            op = ConstantOp(value=self.types.enum_constants[node.name], type=c_int(32))
            yield op
            return op
        val = scope.lookup(node.name)
        if isinstance(val, (VariableDeclarationOp, AssignOp, CompoundAssignOp)):
            read = ReadVariableOp(
                variable_name=String().constant(node.name),
                source=val,
                type=val.type,
            )
            yield read
            return read
        return val

    def _binary(self, node: c_ast.BinaryOp, scope: Scope) -> Iterator[dgen.Op]:
        left = yield from self._expr(node.left, scope)
        right = yield from self._expr(node.right, scope)
        cls = _ALL_BINOPS.get(node.op)
        if cls is None:
            raise LoweringError(f"unsupported binary operator: {node.op}")
        ty = self._promote(left.type, right.type)
        op = _binop(cls, left, right, ty)
        yield op
        # TODO: move to an implicit-cast pass. C comparisons return int,
        # but algebra comparisons lower to i1. Cast to widen.
        if node.op in _COMPARISONS:
            cast = algebra.CastOp(input=op, type=c_int(32))
            yield cast
            return cast
        return op

    def _unary(self, node: c_ast.UnaryOp, scope: Scope) -> Iterator[dgen.Op]:
        if node.op == "sizeof":
            target = (
                self.types.resolve(node.expr)
                if isinstance(node.expr, c_ast.Typename)
                else c_int(32)
            )
            op = SizeofOp(target_type=target, type=c_int(64, signed=False))
            yield op
            return op
        if node.op == "&":
            operand = yield from self._expr(node.expr, scope)
            op = AddressOfOp(operand=operand, type=Reference(element_type=operand.type))
            yield op
            return op
        if node.op == "*":
            inner = yield from self._expr(node.expr, scope)
            op = DereferenceOp(pointer=inner, type=self._pointee(inner.type))
            yield op
            return op
        if node.op in _INCREMENTS:
            if not isinstance(node.expr, c_ast.ID):
                raise LoweringError("increment/decrement on non-variable")
            target = scope.lookup(node.expr.name)
            op = _INCREMENTS[node.op](
                variable_name=String().constant(node.expr.name),
                target=target,
                type=target.type,
            )
            yield op
            return op
        inner = yield from self._expr(node.expr, scope)
        if node.op == "-":
            op = algebra.NegateOp(input=inner, type=inner.type)
        elif node.op == "~":
            op = algebra.ComplementOp(input=inner, type=inner.type)
        elif node.op == "!":
            op = LogicalNotOp(operand=inner, type=c_int(32))
        elif node.op == "+":
            return inner
        else:
            raise LoweringError(f"unsupported unary operator: {node.op}")
        yield op
        return op

    def _call(self, node: c_ast.FuncCall, scope: Scope) -> Iterator[dgen.Op]:
        if not isinstance(node.name, c_ast.ID):
            raise LoweringError("indirect function calls not yet supported")
        callee = node.name.name
        if self.file_scope.has(callee):
            ret_type = self.file_scope.lookup(callee).type
        else:
            ret_type = c_int(32)
        args: list[dgen.Value] = []
        if node.args:
            for arg in node.args.exprs:
                args.append((yield from self._expr(arg, scope)))
        p = pack(args) if args else pack([])
        yield p
        op = CallOp(callee=String().constant(callee), arguments=p, type=ret_type)
        yield op
        return op

    def _struct(self, node: c_ast.StructRef, scope: Scope) -> Iterator[dgen.Op]:
        base = yield from self._expr(node.name, scope)
        field = node.field.name
        if node.type == "->":
            ft = self.types.get_struct_field_type(self._pointee(base.type), field)
            op = PointerMemberAccessOp(
                field_name=String().constant(field), base=base, type=ft
            )
        else:
            ft = self.types.get_struct_field_type(base.type, field)
            op = MemberAccessOp(field_name=String().constant(field), base=base, type=ft)
        yield op
        return op

    # --- Type helpers (minimal) ---

    def _pointee(self, ty: dgen.Type) -> dgen.Type:
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
    parser = Parser()
    module = parser.parse(ast)
    return module, parser.stats
