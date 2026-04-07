"""Translate pycparser AST to C-dialect IR.

Thin 1:1 lowering from pycparser AST to dgen ops. Semantic transforms
(type promotions, struct layout, lvalue elimination) belong in passes.
Follows the Toy dialect pattern: generator methods yielding ops.
"""

from __future__ import annotations

from collections import defaultdict
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
    """Lexical scope for C name resolution and memory ordering.

    Tracks per variable:
    - _bindings: the current value (latest write, for type resolution)
    - _variables: which names are mutable variables
    - _last_write: the latest write (reads source from this)
    - _pending_reads: reads since the last write (a write depends on all)

    Ordering rules:
    - Reads depend on the latest write only. Back-to-back reads are
      independent — they can execute in any order.
    - A write depends on ALL pending reads (via a pack), ensuring every
      read completes before the write overwrites the value.
    - Operations on different variables are fully independent.
    """

    def __init__(self, parent: Scope | None = None) -> None:
        self._bindings: dict[str, dgen.Value] = {}
        self._variables: set[str] = set()
        self._last_write: dict[str, dgen.Value] = {}
        self._pending_reads: dict[str, list[dgen.Value]] = defaultdict(list)
        self._parent = parent
        self.captures: list[dgen.Value] = []

    def bind(self, name: str, value: dgen.Value) -> None:
        self._bindings[name] = value

    def declare_variable(self, name: str, value: dgen.Value) -> None:
        """Bind a name as a mutable variable and initialize ordering."""
        self._bindings[name] = value
        self._variables.add(name)
        self._last_write[name] = value

    def is_variable(self, name: str) -> bool:
        if name in self._variables:
            return True
        return self._parent.is_variable(name) if self._parent is not None else False

    def read_source(self, name: str) -> dgen.Value:
        """Source for a read: the latest write. Reads are independent."""
        return self._find_last_write(name)

    def write_source(self, name: str) -> dgen.Value:
        """Source for a write: a pack of all pending reads (or the last
        write if no reads). The write depends on all of them."""
        reads = self._pending_reads[name]
        if reads:
            return pack(reads)
        return self._find_last_write(name)

    def record_read(self, name: str, read_op: dgen.Value) -> None:
        """Record a read. Does not advance the write chain."""
        self._pending_reads[name].append(read_op)

    def record_write(self, name: str, write_op: dgen.Value) -> None:
        """Record a write. Clears pending reads and advances the chain."""
        self._bindings[name] = write_op
        self._last_write[name] = write_op
        self._pending_reads[name] = []

    def lookup(self, name: str) -> dgen.Value:
        return self._find_binding(name)

    def _find_binding(self, name: str) -> dgen.Value:
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            value = self._parent._find_binding(name)
            if value not in self.captures:
                self.captures.append(value)
            return value
        raise LoweringError(f"undefined: {name}")

    def _find_last_write(self, name: str) -> dgen.Value:
        if name in self._last_write:
            return self._last_write[name]
        if self._parent is not None:
            return self._parent._find_last_write(name)
        raise LoweringError(f"no write for: {name}")

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

# C literal type string -> (parser function, bits, signed).
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
        c_ast.ID: "_identifier",
        c_ast.BinaryOp: "_binary",
        c_ast.UnaryOp: "_unary",
        c_ast.Cast: "_cast",
        c_ast.TernaryOp: "_ternary",
        c_ast.Assignment: "_assignment",
    }

    # Maps pycparser statement node type -> handler method name.
    _STMT_DISPATCH: dict[type[c_ast.Node], str] = {
        c_ast.Return: "_return_statement",
        c_ast.Decl: "_declaration_statement",
        c_ast.Assignment: "_assignment_statement",
        c_ast.Compound: "_compound_statement",
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
        function_type = self._function_type(type_node)
        self.file_scope.bind(
            name,
            ExternOp(name=name, symbol=String().constant(name), type=function_type),
        )

    def _function_type(self, node: c_ast.Node) -> FunctionType:
        """Extract a function.Function type from a FuncDecl."""
        function_decl = node
        while isinstance(function_decl, c_ast.PtrDecl):
            function_decl = function_decl.type
        return_type = self._return_type(node)
        argument_types: list[dgen.Type] = []
        if isinstance(function_decl, c_ast.FuncDecl) and function_decl.args:
            for parameter in function_decl.args.params:
                if isinstance(parameter, c_ast.EllipsisParam):
                    continue
                if isinstance(parameter, c_ast.Decl):
                    argument_types.append(self.types.resolve(parameter.type))
        return FunctionType(arguments=pack(argument_types), result_type=return_type)

    def _return_type(self, node: c_ast.Node) -> dgen.Type:
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return Reference(element_type=self._return_type(node.type))
        return self.types.resolve(node)

    def _function(self, funcdef: c_ast.FuncDef) -> function.FunctionOp:
        """Lower a FuncDef to a FunctionOp."""
        name = funcdef.decl.name
        return_type = self._function_type(funcdef.decl.type).result_type
        self._current_return_type = return_type

        # Create block args for function parameters, then bind each as
        # a local variable (LvalueVarOp + AssignOp) so _identifier treats
        # parameters and locals uniformly.
        scope = self.file_scope.child()
        arguments: list[BlockArgument] = []
        if isinstance(funcdef.decl.type, c_ast.FuncDecl):
            function_decl = funcdef.decl.type
            if function_decl.args:
                for parameter in function_decl.args.params:
                    if isinstance(parameter, c_ast.EllipsisParam):
                        continue
                    if isinstance(parameter, c_ast.Decl) and parameter.name:
                        argument = BlockArgument(
                            name=parameter.name,
                            type=self.types.resolve(parameter.type),
                        )
                        arguments.append(argument)
                        lvalue = LvalueVarOp(
                            var_name=String().constant(parameter.name),
                            source=argument,
                            type=argument.type,
                        )
                        assign = AssignOp(
                            lvalue=lvalue, rvalue=argument, type=argument.type
                        )
                        scope.declare_variable(parameter.name, assign)

        function_type = FunctionType(
            arguments=pack(arg.type for arg in arguments),
            result_type=return_type,
        )

        return_value, ops = self._compound(funcdef.body, scope)

        if return_value is not None:
            block_result = return_value
        elif ops:
            block_result = ops[-1]
        else:
            block_result = ConstantOp(value=0, type=return_type)

        return function.FunctionOp(
            name=name,
            result_type=return_type,
            body=dgen.Block(result=block_result, args=arguments),
            type=function_type,
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
                result = self._statement(item, scope)
                if isinstance(result, _Return):
                    return_value = result.value
                else:
                    ops.extend(result)
        return return_value, ops

    # --- Statements ---

    def _statement(self, node: c_ast.Node, scope: Scope) -> list[dgen.Op] | _Return:
        """Lower a statement. Dispatches to handler by node type."""
        handler_name = self._STMT_DISPATCH.get(type(node))
        if handler_name is not None:
            return getattr(self, handler_name)(node, scope)
        # Expression statement -- evaluate for side effects.
        return list(self._expression(node, scope))

    def _return_statement(self, node: c_ast.Return, scope: Scope) -> _Return:
        if node.expr is None:
            return _Return(ConstantOp(value=0, type=self._current_return_type))
        return _Return(_run_gen(self._expression(node.expr, scope)))

    def _compound_statement(
        self, node: c_ast.Compound, scope: Scope
    ) -> list[dgen.Op] | _Return:
        return_value, ops = self._compound(node, scope.child())
        if return_value is not None:
            return _Return(return_value)
        return ops

    def _declaration_statement(self, node: c_ast.Decl, scope: Scope) -> list[dgen.Op]:
        return list(self._declaration(node, scope))

    def _declaration(self, node: c_ast.Decl, scope: Scope) -> Iterator[dgen.Op]:
        """Lower a declaration (e.g. int x = 5;)."""
        if node.name is None:
            if node.type is not None:
                self.types.resolve(node.type)
            return
        variable_type = self.types.resolve(node.type)
        if node.init is not None:
            initial_value = _run_gen(self._expression(node.init, scope))
        else:
            yield (initial_value := ConstantOp(value=0, type=variable_type))
        yield (
            lvalue := LvalueVarOp(
                var_name=String().constant(node.name),
                source=initial_value,
                type=variable_type,
            )
        )
        yield (
            assign := AssignOp(lvalue=lvalue, rvalue=initial_value, type=variable_type)
        )
        scope.declare_variable(node.name, assign)

    def _assignment_statement(
        self, node: c_ast.Assignment, scope: Scope
    ) -> list[dgen.Op] | _Return:
        return list(self._assignment(node, scope))

    def _assignment(
        self, node: c_ast.Assignment, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an assignment expression."""
        rhs = yield from self._expression(node.rvalue, scope)

        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(
                f"unsupported assignment target: {type(node.lvalue).__name__}"
            )
        variable_name = node.lvalue.name
        variable_type = scope.lookup(variable_name).type
        yield (
            lvalue := LvalueVarOp(
                var_name=String().constant(variable_name),
                source=scope.write_source(variable_name),
                type=variable_type,
            )
        )
        yield (assign := AssignOp(lvalue=lvalue, rvalue=rhs, type=variable_type))
        scope.record_write(variable_name, assign)
        return assign

    # --- Expressions ---

    def _expression(
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
        literal = _LITERAL_TYPES.get(node.type)
        if literal is not None:
            parser_fn, bits, signed = literal
            yield (
                op := ConstantOp(value=parser_fn(node.value), type=c_int(bits, signed))
            )
            return op
        if node.type in ("float", "double"):
            yield (
                op := ConstantOp(value=float(node.value.rstrip("fFlL")), type=Float64())
            )
            return op
        yield (op := ConstantOp(value=0, type=c_int(32)))
        return op

    def _identifier(
        self, node: c_ast.ID, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an identifier reference.

        Variables emit lvalue_var + lvalue_to_rvalue. The source comes
        from the latest write (so back-to-back reads are independent),
        but we record the read so a subsequent write depends on it.
        """
        if node.name in self.types.enum_constants:
            yield (
                op := ConstantOp(
                    value=self.types.enum_constants[node.name], type=c_int(32)
                )
            )
            return op
        if not scope.is_variable(node.name):
            return scope.lookup(node.name)
        variable_type = scope.lookup(node.name).type
        yield (
            lvalue := LvalueVarOp(
                var_name=String().constant(node.name),
                source=scope.read_source(node.name),
                type=variable_type,
            )
        )
        yield (load := LvalueToRvalueOp(lvalue=lvalue, type=variable_type))
        scope.record_read(node.name, load)
        return load

    def _binary(
        self, node: c_ast.BinaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        lhs = yield from self._expression(node.left, scope)
        rhs = yield from self._expression(node.right, scope)

        cls = _ALGEBRA.get(node.op)
        if cls is None:
            raise LoweringError(f"unsupported binary op: {node.op}")

        if node.op in _COMPARISONS:
            yield (comparison := _binop(cls, lhs, rhs, lhs.type))
            yield (cast := algebra.CastOp(input=comparison, type=c_int(32)))
            return cast

        yield (op := _binop(cls, lhs, rhs, lhs.type))
        return op

    def _unary(
        self, node: c_ast.UnaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        entry = _UNARY.get(node.op)
        if entry is not None:
            cls, field_name = entry
            inner = yield from self._expression(node.expr, scope)
            yield (op := cls(**{field_name: inner, "type": inner.type}))
            return op
        if node.op == "+":
            return (yield from self._expression(node.expr, scope))
        raise LoweringError(f"unsupported unary op: {node.op}")

    def _cast(
        self, node: c_ast.Cast, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        return (yield from self._expression(node.expr, scope))

    def _ternary(
        self, node: c_ast.TernaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        raise LoweringError("ternary operator (?:) not yet supported")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Return:
    """Sentinel that _statement returns when encountering a return."""

    __slots__ = ("value",)

    def __init__(self, value: dgen.Value) -> None:
        self.value = value


def _run_gen(gen: Generator[dgen.Op, None, dgen.Value]) -> dgen.Value:
    """Exhaust a generator and return its final value."""
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
    return Parser().parse(ast)
