"""Translate pycparser AST to C-dialect IR.

Thin 1:1 lowering from pycparser AST to dgen ops. Semantic transforms
(type promotions, struct layout, lvalue elimination) belong in passes.
Follows the Toy dialect pattern: generator methods yielding ops.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, function
from dgen.dialects.builtin import ExternOp, String
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Float64
from dgen.module import ConstantOp, pack

from dcc2.dialects import c_int
from dcc2.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp
from dcc2.parser.c_literals import parse_c_char, parse_c_int
from dcc2.parser.type_resolver import TypeResolver


class LoweringError(Exception):
    """A C construct the parser cannot yet translate."""


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
# Op tables
# ---------------------------------------------------------------------------

# Binary C operators -> algebra op class.
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
}

_COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}

# Unary C operators -> (algebra op class, operand field name).
_UNARY: dict[str, tuple[type[dgen.Op], str]] = {
    "-": (algebra.NegateOp, "input"),
    "~": (algebra.ComplementOp, "input"),
}

# C literal type string -> (parser function, dgen type).
_LITERAL_TYPES: dict[str, tuple[type, ...]] = {
    "int": (parse_c_int, 32, True),
    "char": (parse_c_char, 8, True),
}


def _binop(cls: type[dgen.Op], a: dgen.Value, b: dgen.Value, ty: dgen.Type) -> dgen.Op:
    if "left" in cls.__dataclass_fields__:
        return cls(left=a, right=b, type=ty)
    return cls(lhs=a, rhs=b, type=ty)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Translate pycparser AST to C-dialect IR."""

    # Maps pycparser expression node type -> handler method name.
    _EXPR_DISPATCH: dict[type[c_ast.Node], str] = {
        c_ast.Constant: "_constant",
        c_ast.ID: "_id",
        c_ast.BinaryOp: "_binary",
        c_ast.UnaryOp: "_unary",
        c_ast.Cast: "_cast",
        c_ast.TernaryOp: "_ternary",
        c_ast.Assignment: "_assignment",
    }

    # Maps pycparser statement node type -> handler method name.
    _STMT_DISPATCH: dict[type[c_ast.Node], str] = {
        c_ast.Return: "_return",
        c_ast.Decl: "_decl_stmt",
        c_ast.Assignment: "_assignment_stmt",
        c_ast.Compound: "_compound_stmt",
    }

    def __init__(self) -> None:
        self.types = TypeResolver()
        self.file_scope = Scope()
        self._current_return_type: dgen.Type = c_int(32)

    def parse(self, ast: c_ast.FileAST) -> function.FunctionOp:
        """Lower a FileAST to IR. Returns the last function."""
        # First pass: register types and extern stubs for all functions.
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef):
                if ext.name is not None:
                    self.types.register_typedef(ext.name, self.types.resolve(ext.type))
            elif isinstance(ext, c_ast.Decl):
                if isinstance(ext.type, c_ast.FuncDecl) and ext.name:
                    self._bind_extern(ext.name, ext.type)
                elif isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                    self.types.resolve(ext.type)
            elif isinstance(ext, c_ast.FuncDef) and ext.decl.name:
                self._bind_extern(ext.decl.name, ext.decl.type)

        # Second pass: lower function definitions.
        functions: list[function.FunctionOp] = []
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                functions.append(self._function(ext))

        if not functions:
            raise LoweringError("no functions defined")
        return functions[-1]

    def _bind_extern(self, name: str, type_node: c_ast.Node) -> None:
        """Bind a file-scope function name to an ExternOp. Idempotent."""
        if self.file_scope.has(name):
            return
        fn_type = self._function_type(type_node)
        self.file_scope.bind(
            name,
            ExternOp(name=name, symbol=String().constant(name), type=fn_type),
        )

    def _function_type(self, node: c_ast.Node) -> FunctionType:
        """Extract a function.Function type from a FuncDecl."""
        func_decl = node
        while isinstance(func_decl, c_ast.PtrDecl):
            func_decl = func_decl.type
        ret = self._return_type(node)
        arg_types: list[dgen.Type] = []
        if isinstance(func_decl, c_ast.FuncDecl) and func_decl.args:
            for param in func_decl.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl):
                    arg_types.append(self.types.resolve(param.type))
        return FunctionType(arguments=pack(arg_types), result_type=ret)

    def _return_type(self, node: c_ast.Node) -> dgen.Type:
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return Reference(element_type=self._return_type(node.type))
        return self.types.resolve(node)

    def _function(self, funcdef: c_ast.FuncDef) -> function.FunctionOp:
        """Lower a FuncDef to a FunctionOp."""
        name = funcdef.decl.name
        fn_type = self._function_type(funcdef.decl.type)
        ret_type = fn_type.result_type
        self._current_return_type = ret_type

        # Create block args for function parameters.
        scope = self.file_scope.child()
        args: list[BlockArgument] = []
        if isinstance(funcdef.decl.type, c_ast.FuncDecl):
            func_decl = funcdef.decl.type
            if func_decl.args:
                for param in func_decl.args.params:
                    if isinstance(param, c_ast.EllipsisParam):
                        continue
                    if isinstance(param, c_ast.Decl) and param.name:
                        arg = BlockArgument(
                            name=param.name, type=self.types.resolve(param.type)
                        )
                        scope.bind(param.name, arg)
                        args.append(arg)

        # Rebuild fn_type from actual resolved args.
        fn_type = FunctionType(
            arguments=pack(arg.type for arg in args),
            result_type=ret_type,
        )

        # Lower body.
        return_value, ops = self._compound(funcdef.body, scope)

        # Determine block result.
        if return_value is not None:
            block_result = return_value
        elif ops:
            block_result = ops[-1]
        else:
            block_result = ConstantOp(value=0, type=ret_type)

        return function.FunctionOp(
            name=name,
            result_type=ret_type,
            body=dgen.Block(result=block_result, args=args),
            type=fn_type,
        )

    def _compound(
        self, node: c_ast.Compound, scope: Scope
    ) -> tuple[dgen.Value | None, list[dgen.Op]]:
        """Lower a compound statement. Returns (return_value, ops)."""
        ops: list[dgen.Op] = []
        return_value: dgen.Value | None = None
        if node.block_items:
            for item in node.block_items:
                if return_value is not None:
                    break
                result = self._stmt(item, scope)
                if isinstance(result, _Return):
                    return_value = result.value
                else:
                    ops.extend(result)
        return return_value, ops

    # --- Statements ---

    def _stmt(self, node: c_ast.Node, scope: Scope) -> list[dgen.Op] | _Return:
        """Lower a statement. Dispatches to handler by node type."""
        handler_name = self._STMT_DISPATCH.get(type(node))
        if handler_name is not None:
            return getattr(self, handler_name)(node, scope)
        # Expression statement -- evaluate for side effects.
        return list(self._expr(node, scope))

    def _return(self, node: c_ast.Return, scope: Scope) -> _Return:
        """Lower a return statement."""
        if node.expr is None:
            return _Return(ConstantOp(value=0, type=self._current_return_type))
        val = _run_gen(self._expr(node.expr, scope))
        return _Return(val)

    def _compound_stmt(
        self, node: c_ast.Compound, scope: Scope
    ) -> list[dgen.Op] | _Return:
        """Lower a nested compound statement (child scope)."""
        return_value, ops = self._compound(node, scope.child())
        if return_value is not None:
            return _Return(return_value)
        return ops

    def _decl_stmt(self, node: c_ast.Decl, scope: Scope) -> list[dgen.Op]:
        """Lower a declaration statement."""
        return list(self._decl(node, scope))

    def _decl(self, node: c_ast.Decl, scope: Scope) -> Iterator[dgen.Op]:
        """Lower a declaration (e.g. int x = 5;).

        Emits lvalue_var + assign. The lvalue elimination pass
        (CLvalueToMemory) converts these to stack allocations and stores.
        """
        if node.name is None:
            # Type-only declaration (struct/union/enum definition).
            if node.type is not None:
                self.types.resolve(node.type)
            return
        var_type = self.types.resolve(node.type)
        init_val: dgen.Value
        if node.init is not None:
            init_val = _run_gen(self._expr(node.init, scope))
        else:
            init_val = ConstantOp(value=0, type=var_type)
            yield init_val
        lv = LvalueVarOp(
            var_name=String().constant(node.name),
            source=init_val,
            type=var_type,
        )
        yield lv
        assign = AssignOp(lvalue=lv, rvalue=init_val, type=var_type)
        yield assign
        scope.bind(node.name, assign)

    def _assignment_stmt(
        self, node: c_ast.Assignment, scope: Scope
    ) -> list[dgen.Op] | _Return:
        """Lower an assignment statement (e.g. x = 5;)."""
        return list(self._assignment(node, scope))

    def _assignment(
        self, node: c_ast.Assignment, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an assignment expression."""
        rhs = yield from self._expr(node.rvalue, scope)

        # For now, only handle direct variable assignment (ID on LHS).
        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(
                f"unsupported assignment target: {type(node.lvalue).__name__}"
            )
        var_name = node.lvalue.name
        prev = scope.lookup(var_name)
        lv = LvalueVarOp(
            var_name=String().constant(var_name),
            source=prev,
            type=prev.type,
        )
        yield lv
        assign = AssignOp(lvalue=lv, rvalue=rhs, type=prev.type)
        yield assign
        scope.bind(var_name, assign)
        return assign

    # --- Expressions ---

    def _expr(
        self, node: c_ast.Node, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an expression. Dispatches to handler by node type."""
        handler_name = self._EXPR_DISPATCH.get(type(node))
        if handler_name is not None:
            return (yield from getattr(self, handler_name)(node, scope))
        raise LoweringError(f"unsupported expression: {type(node).__name__}")

    def _constant(
        self, node: c_ast.Constant, _scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower a literal constant."""
        literal = _LITERAL_TYPES.get(node.type)
        if literal is not None:
            parser_fn, bits, signed = literal
            val = parser_fn(node.value)
            op = ConstantOp(value=val, type=c_int(bits, signed))
            yield op
            return op
        if node.type in ("float", "double"):
            val = float(node.value.rstrip("fFlL"))
            op = ConstantOp(value=val, type=Float64())
            yield op
            return op
        # Default: integer 0.
        op = ConstantOp(value=0, type=c_int(32))
        yield op
        return op

    def _id(self, node: c_ast.ID, scope: Scope) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an identifier reference.

        For local variables (bound to an AssignOp), emits lvalue_var +
        lvalue_to_rvalue to load the current value. For parameters and
        other values, returns the binding directly.
        """
        if node.name in self.types.enum_constants:
            op = ConstantOp(value=self.types.enum_constants[node.name], type=c_int(32))
            yield op
            return op
        binding = scope.lookup(node.name)
        if isinstance(binding, AssignOp):
            lv = LvalueVarOp(
                var_name=String().constant(node.name),
                source=binding,
                type=binding.type,
            )
            yield lv
            load = LvalueToRvalueOp(lvalue=lv, type=binding.type)
            yield load
            return load
        return binding

    def _binary(
        self, node: c_ast.BinaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower a binary operation."""
        cls = _ALGEBRA.get(node.op)
        if cls is None:
            raise LoweringError(f"unsupported binary op: {node.op}")

        lhs = yield from self._expr(node.left, scope)
        rhs = yield from self._expr(node.right, scope)

        # Comparisons produce a boolean; wrap in CastOp to widen to int.
        if node.op in _COMPARISONS:
            cmp_op = _binop(cls, lhs, rhs, lhs.type)
            yield cmp_op
            cast = algebra.CastOp(input=cmp_op, type=c_int(32))
            yield cast
            return cast

        op = _binop(cls, lhs, rhs, lhs.type)
        yield op
        return op

    def _unary(
        self, node: c_ast.UnaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower a unary operation."""
        entry = _UNARY.get(node.op)
        if entry is not None:
            cls, field_name = entry
            inner = yield from self._expr(node.expr, scope)
            op = cls(**{field_name: inner, "type": inner.type})
            yield op
            return op
        if node.op == "+":
            return (yield from self._expr(node.expr, scope))
        raise LoweringError(f"unsupported unary op: {node.op}")

    def _cast(
        self, node: c_ast.Cast, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower a cast expression (pass-through for now)."""
        return (yield from self._expr(node.expr, scope))

    def _ternary(
        self, node: c_ast.TernaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower a ternary (a ? b : c) -- requires IfOp (Brick 6)."""
        raise LoweringError("ternary operator (?:) not yet supported")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Return:
    """Sentinel that _stmt returns when encountering a return statement."""

    __slots__ = ("value",)

    def __init__(self, value: dgen.Value) -> None:
        self.value = value


def _run_gen(gen: Generator[dgen.Op, None, dgen.Value]) -> dgen.Value:
    """Exhaust a generator and return its final value."""
    # Consume all yielded ops, then capture the return value from StopIteration.
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(ast: c_ast.FileAST) -> function.FunctionOp:
    """Lower a pycparser FileAST to a FunctionOp."""
    parser = Parser()
    return parser.parse(ast)
