"""Parse C AST into dgen IR.

Thin translation from pycparser AST to C-dialect ops. Name resolution
uses a Scope chain threaded through every method — no mutable self state
for scoping. Semantic transformations (type promotion, implicit casts,
compound assignment expansion) belong in passes, not here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra
from dgen.dialects.builtin import Array, ExternOp, Nil, String
from dgen.dialects.control_flow import IfOp, WhileOp
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, pack, string_value
from dgen.dialects import function

from dcc.dialects import c_int
from dcc.parser.c_literals import parse_c_char, parse_c_int
from dcc.dialects.c import (
    AddressOfOp,
    AssignOp,
    BreakOp,
    CallOp,
    CompoundAssignOp,
    ContinueOp,
    DereferenceOp,
    ElementAddressOp,
    FieldAddressOp,
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
    StoreIndirectOp,
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

# Unary ops that take an evaluated inner operand
_UNARY_OPS: dict[str, tuple[type[dgen.Op], str]] = {
    "-": (algebra.NegateOp, "input"),
    "~": (algebra.ComplementOp, "input"),
    "!": (LogicalNotOp, "operand"),
}


def _make_unary(op: str, inner: dgen.Value) -> dgen.Op:
    entry = _UNARY_OPS.get(op)
    if entry is None:
        raise LoweringError(f"unsupported unary operator: {op}")
    cls, field = entry
    return cls(**{field: inner, "type": inner.type if field == "input" else c_int(32)})


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
    skip_reasons: dict[str, int] = field(default_factory=dict)
    function_ops: list["function.FunctionOp"] = field(default_factory=list)

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
        self._current_return_type: dgen.Type | None = None
        # (callee_name, return_type) for every named c.CallOp produced,
        # including calls generated inside unreachable if/while bodies.
        # Used to synthesise ExternOps for undefined callees.
        self._named_calls: dict[str, dgen.Type] = {}

    def parse(self, ast: c_ast.FileAST) -> function.FunctionOp:
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
                elif ext.name is not None:
                    # Global variable declaration — register as extern symbol.
                    try:
                        var_type = self.types.resolve(ext.type)
                    except Exception:
                        continue
                    self.file_scope.bind(
                        ext.name,
                        ExternOp(
                            name=ext.name,
                            symbol=String().constant(ext.name),
                            type=var_type,
                        ),
                    )

        # Second pass: lower function definitions
        functions: list[function.FunctionOp] = []
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                self.stats.functions += 1
                try:
                    functions.append(self._function(ext))
                except LoweringError as e:
                    self.stats.skipped_functions += 1
                    key = str(e)
                    self.stats.skip_reasons[key] = (
                        self.stats.skip_reasons.get(key, 0) + 1
                    )

        self.stats.function_ops = functions
        _resolve_callee_captures(functions, self._named_calls)
        return functions[-1]

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

        prev_ret = self._current_return_type
        self._current_return_type = ret
        try:
            ops = list(self._compound(node.body, scope)) if node.body else []
        finally:
            self._current_return_type = prev_ret
        result = ops[-1] if ops else dgen.Value(type=Nil())
        is_void = isinstance(ret, Nil)

        return function.FunctionOp(
            name=name,
            result_type=Nil() if is_void else ret,
            body=dgen.Block(result=result, args=args),
            type=FunctionType(
                arguments=pack(arg.type for arg in args),
                result_type=Nil() if is_void else ret,
            ),
        )

    # --- Statements ---

    def _compound(self, node: c_ast.Compound, scope: Scope) -> Iterator[dgen.Op]:
        child = scope.child()
        if node.block_items:
            for item in node.block_items:
                yield from self._stmt(item, child)

    _STMT_DISPATCH: dict[type, str] = {
        c_ast.Decl: "_decl",
        c_ast.Assignment: "_assign",
        c_ast.Return: "_ret",
        c_ast.If: "_if",
        c_ast.While: "_while",
        c_ast.DoWhile: "_do_while",
        c_ast.For: "_for",
        c_ast.Compound: "_compound",
    }

    def _stmt(self, node: c_ast.Node, scope: Scope) -> Iterator[dgen.Op]:
        self.stats.statements += 1
        handler = self._STMT_DISPATCH.get(type(node))
        if handler is not None:
            yield from getattr(self, handler)(node, scope)
            return
        if isinstance(node, (c_ast.FuncCall, c_ast.UnaryOp)):
            yield from self._expr(node, scope)
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
        elif isinstance(node, (c_ast.Case, c_ast.Default)):
            for s in node.stmts or []:
                yield from self._stmt(s, scope)
        elif isinstance(node, c_ast.Typedef):
            if node.name:
                self.types.register_typedef(node.name, self.types.resolve(node.type))
                self.stats.typedefs += 1
        elif isinstance(node, c_ast.DeclList):
            for decl in node.decls:
                yield from self._decl(decl, scope)
        elif not isinstance(node, (c_ast.EmptyStatement, c_ast.Goto, c_ast.Pragma)):
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
        if isinstance(node.lvalue, c_ast.ID):
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
            return

        # Non-ID lvalue: compute its address and store through it.
        addr = yield from self._address_of(node.lvalue, scope)
        elem_type = self._pointee(addr.type)
        if node.op in _COMPOUND_OPS:
            load = DereferenceOp(pointer=addr, type=elem_type)
            yield load
            rhs = yield from self._expr(node.rvalue, scope)
            cls = _ALL_BINOPS.get(_COMPOUND_OPS[node.op])
            if cls is None:
                raise LoweringError(f"unsupported compound op: {node.op}")
            combined = _binop(cls, load, rhs, elem_type)
            yield combined
            store = StoreIndirectOp(target=addr, value=combined, type=elem_type)
        else:
            rhs = yield from self._expr(node.rvalue, scope)
            store = StoreIndirectOp(target=addr, value=rhs, type=elem_type)
        yield store

    def _address_of(self, node: c_ast.Node, scope: Scope) -> Iterator[dgen.Op]:
        """Yield ops to compute the address of an lvalue. Returns a pointer Value."""
        if isinstance(node, c_ast.ID):
            val = scope.lookup(node.name)
            op = AddressOfOp(operand=val, type=Reference(element_type=val.type))
            yield op
            return op
        if isinstance(node, c_ast.UnaryOp) and node.op == "*":
            # &*p == p
            return (yield from self._expr(node.expr, scope))
        if isinstance(node, c_ast.ArrayRef):
            base = yield from self._expr(node.name, scope)
            idx = yield from self._expr(node.subscript, scope)
            elem = self._pointee(base.type)
            op = ElementAddressOp(
                base=base, index=idx, type=Reference(element_type=elem)
            )
            yield op
            return op
        if isinstance(node, c_ast.StructRef):
            field = node.field.name
            if node.type == "->":
                base = yield from self._expr(node.name, scope)
                field_type = self.types.get_struct_field_type(
                    self._pointee(base.type), field
                )
            else:
                # obj.field — take address of obj, then field offset
                base = yield from self._address_of(node.name, scope)
                field_type = self.types.get_struct_field_type(
                    self._pointee(base.type), field
                )
            op = FieldAddressOp(
                field_name=String().constant(field),
                base=base,
                type=Reference(element_type=field_type),
            )
            yield op
            return op
        raise LoweringError(f"unsupported lvalue: {type(node).__name__}")

    def _ret(self, node: c_ast.Return, scope: Scope) -> Iterator[dgen.Op]:
        if node.expr is None:
            nil = ConstantOp(value=None, type=Nil())
            yield nil
            yield ReturnOp(value=nil)
        else:
            val = yield from self._expr(node.expr, scope)
            ret_ty = self._current_return_type
            # Insert an implicit conversion when the returned value's type
            # doesn't match the enclosing function's declared return type.
            # Limited to ptr↔int for now; algebra_to_llvm turns the cast
            # into inttoptr/ptrtoint.
            if ret_ty is not None and val.type != ret_ty:
                val_is_ptr = isinstance(val.type, Reference)
                ret_is_ptr = isinstance(ret_ty, Reference)
                if val_is_ptr != ret_is_ptr:
                    cast = algebra.CastOp(input=val, type=ret_ty)
                    yield cast
                    val = cast
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
            return (yield from self._assign_expr(node, scope))
        if isinstance(node, c_ast.Cast):
            return (yield from self._cast(node, scope))
        if isinstance(node, c_ast.ArrayRef):
            return (yield from self._subscript(node, scope))
        if isinstance(node, c_ast.StructRef):
            return (yield from self._struct(node, scope))
        if isinstance(node, c_ast.TernaryOp):
            return (yield from self._ternary(node, scope))
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

    def _assign_expr(self, node: c_ast.Assignment, scope: Scope) -> Iterator[dgen.Op]:
        """Assignment used as an expression — returns the assigned value."""
        yield from self._assign(node, scope)
        if not isinstance(node.lvalue, c_ast.ID):
            # For non-ID lvalues, re-read via the same address.
            addr = yield from self._address_of(node.lvalue, scope)
            elem_type = self._pointee(addr.type)
            load = DereferenceOp(pointer=addr, type=elem_type)
            yield load
            return load
        name = node.lvalue.name
        val = scope.lookup(name)
        read = ReadVariableOp(
            variable_name=String().constant(name), source=val, type=val.type
        )
        yield read
        return read

    def _cast(self, node: c_ast.Cast, scope: Scope) -> Iterator[dgen.Op]:
        inner = yield from self._expr(node.expr, scope)
        target = self.types.resolve(node.to_type)
        op = algebra.CastOp(input=inner, type=target)
        yield op
        return op

    def _subscript(self, node: c_ast.ArrayRef, scope: Scope) -> Iterator[dgen.Op]:
        base = yield from self._expr(node.name, scope)
        idx = yield from self._expr(node.subscript, scope)
        op = SubscriptOp(base=base, index=idx, type=self._pointee(base.type))
        yield op
        return op

    def _ternary(self, node: c_ast.TernaryOp, scope: Scope) -> Iterator[dgen.Op]:
        cond = yield from self._expr(node.cond, scope)
        then_scope = scope.child()
        then_block = self._block_from(self._expr(node.iftrue, then_scope), then_scope)
        else_scope = scope.child()
        else_block = self._block_from(self._expr(node.iffalse, else_scope), else_scope)
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

    # --- Literal and identifier helpers ---

    def _constant(
        self, node: c_ast.Constant, target_type: dgen.Type | None = None
    ) -> Iterator[dgen.Op]:
        if node.type in (
            "int",
            "unsigned int",
            "long int",
            "unsigned long int",
            "long long int",
            "unsigned long long int",
        ):
            val = parse_c_int(node.value)
            ty = target_type or self._int_type_from_suffix(node.value)
            op = ConstantOp(value=val, type=ty)
            yield op
            return op
        if node.type in ("float", "double"):
            op = ConstantOp(value=float(node.value.rstrip("fFlL")), type=Float64())
            yield op
            return op
        if node.type == "char":
            op = ConstantOp(value=parse_c_char(node.value), type=c_int(8))
            yield op
            return op
        if node.type == "string":
            op = ConstantOp(value=0, type=Reference(element_type=c_int(8)))
            yield op
            return op
        raise LoweringError(f"unsupported constant type: {node.type}")

    @staticmethod
    def _int_type_from_suffix(text: str) -> dgen.Type:
        suffix = text[len(text.rstrip("uUlL")) :].lower()
        if "ll" in suffix or "l" in suffix:
            return c_int(64, signed="u" not in suffix)
        if "u" in suffix:
            return c_int(32, signed=False)
        return c_int(32)

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
        # File-scope function references used as values (e.g. passed as a
        # function pointer argument) arrive here as bare dgen.Value stubs
        # bound by `parse()` / `_function()`. Promote them to ExternOps so
        # codegen emits `ptr @name` instead of `<retty> %name`.
        if type(val) is dgen.Value and val.name is not None:
            op = ExternOp(
                name=val.name,
                symbol=String().constant(val.name),
                type=FunctionType(
                    arguments=pack([]),
                    result_type=val.type,
                ),
            )
            yield op
            return op
        return val

    def _binary(self, node: c_ast.BinaryOp, scope: Scope) -> Iterator[dgen.Op]:
        left = yield from self._expr(node.left, scope)
        right = yield from self._expr(node.right, scope)
        cls = _ALL_BINOPS.get(node.op)
        if cls is None:
            raise LoweringError(f"unsupported binary operator: {node.op}")
        ty = self._promote(left.type, right.type)
        # Comparisons against the integer literal 0 through a pointer
        # promote the 0 to the pointer type (C null-pointer-constant
        # semantics). algebra_to_llvm lowers CastOp(const-0, Ref) to a
        # typed-null ConstantOp, giving `icmp eq ptr %p, null`.
        if node.op in _COMPARISONS and isinstance(ty, Reference):
            if not isinstance(left.type, Reference):
                cast = algebra.CastOp(input=left, type=ty)
                yield cast
                left = cast
            if not isinstance(right.type, Reference):
                cast = algebra.CastOp(input=right, type=ty)
                yield cast
                right = cast
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
            if isinstance(node.expr, c_ast.ID):
                target = scope.lookup(node.expr.name)
                op = _INCREMENTS[node.op](
                    variable_name=String().constant(node.expr.name),
                    target=target,
                    type=target.type,
                )
                yield op
                return op
            # Non-ID target (e.g. p->x, *p, arr[i]): desugar via _address_of.
            addr = yield from self._address_of(node.expr, scope)
            elem_type = self._pointee(addr.type)
            load = DereferenceOp(pointer=addr, type=elem_type)
            yield load
            one = ConstantOp(value=1, type=elem_type)
            yield one
            is_dec = node.op in ("--", "p--")
            cls = algebra.SubtractOp if is_dec else algebra.AddOp
            updated = _binop(cls, load, one, elem_type)
            yield updated
            store = StoreIndirectOp(target=addr, value=updated, type=elem_type)
            yield store
            return load if node.op.startswith("p") else updated
        inner = yield from self._expr(node.expr, scope)
        if node.op == "+":
            return inner
        op = _make_unary(node.op, inner)
        yield op
        return op

    def _call(self, node: c_ast.FuncCall, scope: Scope) -> Iterator[dgen.Op]:
        args: list[dgen.Value] = []
        if node.args:
            for arg in node.args.exprs:
                args.append((yield from self._expr(arg, scope)))
        p = pack(args) if args else pack([])
        yield p
        # Named-function call only if the name resolves (or is unknown)
        # at file scope. A same-named local variable/parameter means we're
        # calling through a function-pointer value.
        if isinstance(node.name, c_ast.ID) and not (
            scope.has(node.name.name) and not self.file_scope.has(node.name.name)
        ):
            callee = node.name.name
            if self.file_scope.has(callee):
                ret_type = self.file_scope.lookup(callee).type
            else:
                ret_type = c_int(32)
            self._named_calls.setdefault(callee, ret_type)
            op = CallOp(callee=String().constant(callee), arguments=p, type=ret_type)
            yield op
            return op
        # Indirect call through a function-pointer value.
        fn_ptr = yield from self._expr(node.name, scope)
        fn_ty = fn_ptr.type
        # Unwrap one level of pointer/reference if present.
        if isinstance(fn_ty, Reference) and isinstance(fn_ty.element_type, dgen.Type):
            fn_ty = fn_ty.element_type
        if isinstance(fn_ty, FunctionType) and isinstance(fn_ty.result_type, dgen.Type):
            ret_type = fn_ty.result_type
        else:
            ret_type = c_int(32)
        op = function.CallOp(callee=fn_ptr, arguments=p, type=ret_type)
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
        if isinstance(ty, (Reference, Array)):
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


def _resolve_callee_captures(
    functions: list[function.FunctionOp],
    named_calls: dict[str, dgen.Type],
) -> None:
    """Attach callee references as captures on every function's body.

    C's c.CallOp carries the callee as a string name, so callers don't
    naturally reference callees in their use-def graph. To make the
    program reachable from a single entry, attach every defined
    function and every undefined callee (as an ExternOp) as captures
    on every caller.

    ``named_calls`` is the complete set of called-by-name sites
    collected during lowering, including calls inside unreachable
    branch bodies (which the use-def walk would otherwise miss).
    """
    by_name: dict[str, function.FunctionOp] = {f.name: f for f in functions if f.name}
    extern_cache: dict[str, ExternOp] = {}
    for name, ret_type in named_calls.items():
        if name in by_name:
            continue
        extern_cache[name] = ExternOp(
            name=name,
            symbol=String().constant(name),
            type=FunctionType(arguments=pack([]), result_type=ret_type),
        )

    from dgen.graph import all_values

    for func in functions:
        seen: set[dgen.Value] = set()
        captures = list(func.body.captures)
        for v in all_values(func):
            if isinstance(v, CallOp):
                name = string_value(v.callee)
                target: dgen.Value | None = by_name.get(name) or extern_cache.get(name)
                if target is not None and target is not func and target not in seen:
                    seen.add(target)
                    captures.append(target)
        func.body.captures = captures


def lower(ast: c_ast.FileAST) -> tuple[function.FunctionOp, LoweringStats]:
    parser = Parser()
    entry = parser.parse(ast)
    return entry, parser.stats
