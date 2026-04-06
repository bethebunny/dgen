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
from dgen.module import ConstantOp, pack
from dgen.dialects import function

from dcc.dialects import c_int
from dcc.parser.c_literals import parse_c_char, parse_c_int
from dcc.dialects.c import (
    AddressOfOp,
    BreakOp,
    ContinueOp,
    DereferenceOp,
    ElementAddressOp,
    FieldAddressOp,
    LogicalNotOp,
    LvalueAssignOp,
    LvalueCompoundAssignOp,
    LvaluePostDecrementOp,
    LvaluePostIncrementOp,
    LvaluePreDecrementOp,
    LvaluePreIncrementOp,
    LvalueToRvalueOp,
    LvalueVarOp,
    MemberAccessOp,
    ModuloOp,
    PointerMemberAccessOp,
    PostDecrementOp,
    PostIncrementOp,
    PreDecrementOp,
    PreIncrementOp,
    ReturnOp,
    ShiftLeftOp,
    ShiftRightOp,
    SizeofOp,
    StoreIndirectOp,
    SubscriptOp,
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

    ``mutated_parent_names`` tracks names that were re-bound in this
    scope but originated in a parent scope.  The parser uses this after
    if/while statements to propagate mutations upward.
    """

    def __init__(self, parent: Scope | None = None) -> None:
        self._bindings: dict[str, dgen.Value] = {}
        self._parent = parent
        self.captures: list[dgen.Value] = []
        self._declared: dict[str, dgen.Type] = {}
        self.mutated_parent_names: set[str] = set()

    def declare(self, name: str, value: dgen.Value) -> None:
        """Bind a newly declared variable (not a mutation of a parent var)."""
        self._bindings[name] = value
        self._declared[name] = value.type

    def bind(self, name: str, value: dgen.Value) -> None:
        """Re-bind an existing variable.  Tracks parent-scope mutations."""
        self._bindings[name] = value
        if (
            name not in self._declared
            and self._parent is not None
            and self._parent.has(name)
        ):
            self.mutated_parent_names.add(name)

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

    def is_variable(self, name: str) -> bool:
        """True if ``name`` was introduced via ``declare()`` in this or a parent scope."""
        if name in self._declared:
            return True
        return self._parent.is_variable(name) if self._parent is not None else False

    def variable_type(self, name: str) -> dgen.Type:
        """Return the declared type of a variable."""
        if name in self._declared:
            return self._declared[name]
        if self._parent is not None:
            return self._parent.variable_type(name)
        raise LoweringError(f"no declared type for: {name}")

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

_LVALUE_INCREMENTS: dict[str, type[dgen.Op]] = {
    "++": LvaluePreIncrementOp,
    "p++": LvaluePostIncrementOp,
    "--": LvaluePreDecrementOp,
    "p--": LvaluePostDecrementOp,
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

    def parse(self, ast: c_ast.FileAST) -> function.FunctionOp:
        # First pass: register types, globals, and an ExternOp for every
        # file-scope function (decl AND def). Calls reference the
        # ExternOp directly. FunctionOps with the matching name provide
        # the body; codegen deduplicates the declare + define.
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef):
                if ext.name is not None:
                    self.types.register_typedef(ext.name, self.types.resolve(ext.type))
                    self.stats.typedefs += 1
            elif isinstance(ext, c_ast.Decl):
                if isinstance(ext.type, c_ast.FuncDecl) and ext.name:
                    self._bind_extern_function(ext.name, ext.type)
                elif isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                    self.types.resolve(ext.type)
                elif ext.name is not None:
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
            elif isinstance(ext, c_ast.FuncDef) and ext.decl.name:
                self._bind_extern_function(ext.decl.name, ext.decl.type)

        # Second pass: lower function definitions. Each function's body
        # may call other functions via their ExternOp handles.
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

        # Closed-block invariant: every callee ExternOp referenced from
        # a function's body must be a capture.
        _add_callee_captures(functions)

        self.stats.function_ops = functions
        return functions[-1] if functions else dgen.Value(type=Nil())

    def _bind_extern_function(self, name: str, type_node: c_ast.Node) -> None:
        """Bind a file-scope function name to an ExternOp stub. Idempotent
        across repeated declarations and forward declarations."""
        if self.file_scope.has(name):
            return
        try:
            fn_type = self._function_type(type_node)
        except Exception:
            return
        self.file_scope.bind(
            name,
            ExternOp(
                name=name,
                symbol=String().constant(name),
                type=fn_type,
            ),
        )

    def _bind_extern_function_unknown(self, name: str) -> None:
        """Bind `name` to an ExternOp with an unknown signature (empty
        args, i32 return). Used for implicit K&R-style declarations."""
        self.file_scope.bind(
            name,
            ExternOp(
                name=name,
                symbol=String().constant(name),
                type=FunctionType(arguments=pack([]), result_type=c_int(32)),
            ),
        )

    def _return_type(self, node: c_ast.Node) -> dgen.Type:
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return Reference(element_type=self._return_type(node.type))
        return self.types.resolve(node)

    def _function_type(self, node: c_ast.Node) -> FunctionType:
        """Extract a `function.Function` from a FuncDecl (optionally
        wrapped in PtrDecls). Callers pass `ext.type` or
        `funcdef.decl.type`; the node's shape is identical for both."""
        ret = self._return_type(node)
        func_decl: c_ast.Node = node
        while isinstance(func_decl, c_ast.PtrDecl):
            func_decl = func_decl.type
        arg_types: list[dgen.Type] = []
        if isinstance(func_decl, c_ast.FuncDecl) and func_decl.args:
            for param in func_decl.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl):
                    try:
                        arg_types.append(self.types.resolve(param.type))
                    except Exception:
                        # Unresolvable param type — fall back to leaving it
                        # off the signature. LLVM opaque ptrs tolerate a
                        # declare that omits args at the callsite.
                        return FunctionType(arguments=pack([]), result_type=ret)
        return FunctionType(arguments=pack(arg_types), result_type=ret)

    # --- Functions ---

    def _function(self, node: c_ast.FuncDef) -> function.FunctionOp:
        scope = self.file_scope.child()
        name = node.decl.name
        ret = self._return_type(node.decl.type)

        args: list[BlockArgument] = []
        if isinstance(node.decl.type, c_ast.FuncDecl) and node.decl.type.args:
            for param in node.decl.type.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl) and param.name:
                    arg = BlockArgument(
                        name=param.name, type=self.types.resolve(param.type)
                    )
                    scope.declare(param.name, arg)
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
        from dgen.dialects.builtin import ChainOp

        child = scope.child()
        last: dgen.Value | None = None
        if node.block_items:
            for item in node.block_items:
                ops = list(self._stmt(item, child))
                yield from ops
                if ops:
                    current = ops[-1]
                    # Chain the current statement's result through the
                    # prior statement so side effects stay reachable.
                    if last is not None and last is not current:
                        chain = ChainOp(lhs=current, rhs=last, type=current.type)
                        yield chain
                        current = chain
                    last = current
        # Propagate parent-scope mutations upward so that enclosing
        # control-flow ops (if/while) can rebind in the outer scope.
        scope.mutated_parent_names.update(child.mutated_parent_names)

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
            self._bind_extern_function(node.name, node.type)
            return
        var_type = self.types.resolve(node.type)
        if node.init is not None:
            init = yield from self._expr(node.init, scope, target_type=var_type)
        else:
            init = ConstantOp(value=0, type=var_type)
            yield init
        lv = LvalueVarOp(
            variable_name=String().constant(node.name),
            source=init,
            type=var_type,
        )
        yield lv
        assign = LvalueAssignOp(lvalue=lv, rvalue=init, type=var_type)
        yield assign
        scope.declare(node.name, assign)

    def _assign(self, node: c_ast.Assignment, scope: Scope) -> Iterator[dgen.Op]:
        if isinstance(node.lvalue, c_ast.ID):
            name = node.lvalue.name
            lv = yield from self._lvalue(node.lvalue, scope)
            if node.op in _COMPOUND_OPS:
                rhs = yield from self._expr(node.rvalue, scope)
                op = LvalueCompoundAssignOp(
                    operator=String().constant(_COMPOUND_OPS[node.op]),
                    lvalue=lv,
                    rvalue=rhs,
                    type=lv.type,
                )
                yield op
                scope.bind(name, op)
            else:
                rhs = yield from self._expr(node.rvalue, scope)
                op = LvalueAssignOp(lvalue=lv, rvalue=rhs, type=lv.type)
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
        if_op = IfOp(
            condition=cond,
            then_arguments=empty,
            else_arguments=empty,
            type=Nil(),
            then_body=then_block,
            else_body=else_block,
        )
        yield if_op
        # Propagate variable mutations from child scopes to the parent.
        # Rebinding to the IfOp creates a use-def edge so subsequent reads
        # depend on the if, making the inner stores reachable.
        for name in then_scope.mutated_parent_names | else_scope.mutated_parent_names:
            scope.bind(name, if_op)

    def _while(self, node: c_ast.While, scope: Scope) -> Iterator[dgen.Op]:
        cond_scope = scope.child()
        cond_block = self._block_from(self._expr(node.cond, cond_scope), cond_scope)
        body_scope = scope.child()
        body_block = self._block_from(self._stmt(node.stmt, body_scope), body_scope)
        p = pack([])
        yield p
        while_op = WhileOp(initial_arguments=p, condition=cond_block, body=body_block)
        yield while_op
        for name in body_scope.mutated_parent_names | cond_scope.mutated_parent_names:
            scope.bind(name, while_op)

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
        lv = yield from self._lvalue(node.lvalue, scope)
        read = LvalueToRvalueOp(lvalue=lv, type=lv.type)
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

    def _lvalue(self, node: c_ast.Node, scope: Scope) -> Iterator[dgen.Op]:
        """Emit an lvalue op for an lvalue expression. Returns the lvalue Value."""
        if isinstance(node, c_ast.ID):
            name = node.name
            val = scope.lookup(name)
            var_type = scope.variable_type(name)
            lv = LvalueVarOp(
                variable_name=String().constant(name), source=val, type=var_type
            )
            yield lv
            return lv
        raise LoweringError(f"unsupported lvalue for new path: {type(node).__name__}")

    def _id(self, node: c_ast.ID, scope: Scope) -> Iterator[dgen.Op]:
        if node.name in self.types.enum_constants:
            op = ConstantOp(value=self.types.enum_constants[node.name], type=c_int(32))
            yield op
            return op
        val = scope.lookup(node.name)
        if scope.is_variable(node.name):
            var_type = scope.variable_type(node.name)
            lv = LvalueVarOp(
                variable_name=String().constant(node.name), source=val, type=var_type
            )
            yield lv
            read = LvalueToRvalueOp(lvalue=lv, type=var_type)
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
                lv = yield from self._lvalue(node.expr, scope)
                op = _LVALUE_INCREMENTS[node.op](lvalue=lv, type=lv.type)
                yield op
                scope.bind(node.expr.name, op)
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
        # Resolve the callee to an SSA value. Direct named calls use the
        # file-scope ExternOp/FunctionOp binding; indirect calls evaluate
        # the expression.
        if isinstance(node.name, c_ast.ID) and not (
            scope.has(node.name.name) and not self.file_scope.has(node.name.name)
        ):
            name = node.name.name
            if not self.file_scope.has(name):
                # Implicit declaration: synthesise an ExternOp so the
                # callsite is still well-formed. The signature is
                # unknown; use i32 return (K&R-style default) and empty
                # args. LLVM opaque ptrs tolerate mismatched signatures
                # at declare time.
                self._bind_extern_function_unknown(name)
            callee = self.file_scope.lookup(name)
        else:
            callee = yield from self._expr(node.name, scope)
        fn_ty = callee.type
        if isinstance(fn_ty, Reference) and isinstance(fn_ty.element_type, dgen.Type):
            fn_ty = fn_ty.element_type
        if isinstance(fn_ty, FunctionType) and isinstance(fn_ty.result_type, dgen.Type):
            ret_type = fn_ty.result_type
        else:
            ret_type = c_int(32)
        op = function.CallOp(callee=callee, arguments=p, type=ret_type)
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


def _add_callee_captures(functions: list[function.FunctionOp]) -> None:
    """Ensure ExternOp callees referenced by function.CallOps in each
    body are listed as block captures (the closed-block invariant).

    Callees are always ExternOps (not FunctionOps), so adding them as
    captures doesn't create cycles in the DAG.
    """
    from dgen.graph import all_values

    for func in functions:
        seen: set[dgen.Value] = set(func.body.captures)
        extras: list[dgen.Value] = []
        for v in all_values(func):
            if isinstance(v, function.CallOp) and isinstance(v.callee, ExternOp):
                callee = v.callee
                if callee not in seen:
                    seen.add(callee)
                    extras.append(callee)
        if extras:
            func.body.captures = list(func.body.captures) + extras


def lower(ast: c_ast.FileAST) -> tuple[function.FunctionOp, LoweringStats]:
    parser = Parser()
    entry = parser.parse(ast)
    return entry, parser.stats
