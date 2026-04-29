"""Translate pycparser AST to C-dialect IR.

Thin 1:1 lowering from pycparser AST to dgen ops. Semantic transforms
(type promotions, struct layout, lvalue elimination) belong in passes.

Every handler returns a single Value whose transitive dependencies
contain every op it produced. No lists, no generators — graph
structure is preserved in use-def edges.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects import algebra, control_flow, function
from dgen.dialects.builtin import ChainOp, ExternOp, Never, Nil, String
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Boolean, Float64
from dgen.builtins import pack

from dcc.dialects import c_int
from dcc.dialects.c import AssignOp, LvalueToRvalueOp, LvalueVarOp
from dcc.parser.c_literals import parse_c_char, parse_c_int
from dcc.parser.type_resolver import TypeResolver


class LoweringError(Exception):
    """A C construct the parser cannot yet translate."""


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass
class Variable:
    """Per-variable state: declared type, ordering chain, and current binding."""

    type: dgen.Type
    ordering: dgen.Value | None = None
    pending_reads: list[dgen.Value] = field(default_factory=list)
    binding: dgen.Value | None = None


class Scope:
    """Lexical scope for C name resolution and memory ordering.

    Two namespaces:
    - **_names**: immutable bindings (externs, enum constants).
    - **_variables**: mutable locals, each a `Variable` that tracks
      declared type, ordering chain, pending reads, and current binding.
    """

    def __init__(self, parent: Scope | None = None) -> None:
        self._names: dict[str, dgen.Value] = {}
        self._variables: dict[str, Variable] = {}
        self.local_reads: set[str] = set()
        self.local_writes: set[str] = set()
        self._parent = parent
        self.captures: list[dgen.Value] = []
        # %div BlockParameter for the enclosing loop, set on the loop
        # body's scope. ``resolve_div`` cascades captures across
        # intermediate scopes (e.g. break inside an if inside a loop).
        self._div: BlockParameter | None = None

    def set_div(self, div: BlockParameter) -> None:
        self._div = div

    def resolve_div(self) -> BlockParameter:
        if self._div is not None:
            return self._div
        if self._parent is None:
            raise LoweringError("'break' or 'continue' outside of loop")
        div = self._parent.resolve_div()
        self.capture(div)
        return div

    def bind(self, name: str, value: dgen.Value) -> None:
        """Bind an immutable name (e.g. an extern function)."""
        self._names[name] = value

    def declare(self, name: str, variable_type: dgen.Type) -> None:
        """Declare a mutable variable. No ops emitted, no ordering yet."""
        self._variables[name] = Variable(type=variable_type)

    def is_variable(self, name: str) -> bool:
        if name in self._variables:
            return True
        return self._parent.is_variable(name) if self._parent is not None else False

    def variable_type(self, name: str) -> dgen.Type:
        """Return the declared type of a variable."""
        if name in self._variables:
            return self._variables[name].type
        if self._parent is not None:
            return self._parent.variable_type(name)
        raise LoweringError(f"undeclared variable: {name}")

    # --- Ordering ---

    def read_ordering(self, name: str) -> dgen.Value | None:
        """Ordering token for a read: the latest ordering node, or None
        if the variable has never been written."""
        if name in self._variables:
            return self._variables[name].ordering
        if self._parent is not None:
            value = self._parent.read_ordering(name)
            if value is not None:
                self.capture(value)
            return value
        return None

    def write_ordering(self, name: str) -> dgen.Value | None:
        """Ordering token for a write: a pack of all pending reads (or the
        latest ordering node, or None if never written)."""
        if name in self._variables:
            var = self._variables[name]
            if var.pending_reads:
                return pack(var.pending_reads)
            return var.ordering
        return self.read_ordering(name)

    def record_read(self, name: str, read_op: dgen.Value) -> None:
        """Record a read. Does not advance the ordering chain."""
        var = self._find_variable(name)
        var.pending_reads.append(read_op)
        self.local_reads.add(name)

    def record_write(self, name: str, write_op: dgen.Value) -> None:
        """Record a write. Updates the binding, clears pending reads,
        and advances the ordering chain."""
        var = self._find_variable(name)
        var.binding = write_op
        var.ordering = write_op
        var.pending_reads = []
        self.local_writes.add(name)

    def record_fence(self, name: str, fence_op: dgen.Value) -> None:
        """Record an ordering fence (e.g. control flow op) without updating
        the name binding. Subsequent reads/writes will depend on it."""
        var = self._find_variable(name)
        var.ordering = fence_op
        var.pending_reads = []
        self.local_writes.add(name)

    def _find_variable(self, name: str) -> Variable:
        """Find a variable in this scope, creating a local shadow if it
        exists only in a parent scope."""
        if name in self._variables:
            return self._variables[name]
        if self._parent is not None and self._parent.is_variable(name):
            # Create a local proxy that inherits the parent's ordering.
            var = Variable(type=self._parent.variable_type(name))
            ordering = self._parent.read_ordering(name)
            if ordering is not None:
                var.ordering = ordering
                self.capture(ordering)
            self._variables[name] = var
            return var
        raise LoweringError(f"undeclared variable: {name}")

    # --- Name resolution ---

    def __contains__(self, name: str) -> bool:
        if name in self._names or name in self._variables:
            return True
        return name in self._parent if self._parent is not None else False

    def __getitem__(self, name: str) -> dgen.Value:
        if name in self._names:
            return self._names[name]
        if name in self._variables:
            var = self._variables[name]
            if var.binding is not None:
                return var.binding
        if self._parent is not None:
            return self._parent[name]
        raise LoweringError(f"undefined: {name}")

    def capture(self, value: dgen.Value) -> None:
        """Declare a cross-scope dependency."""
        if value not in self.captures:
            self.captures.append(value)

    def resolve(self, name: str) -> dgen.Value:
        """Look up a name, adding captures for cross-scope references."""
        if name in self._names:
            return self._names[name]
        if name in self._variables:
            var = self._variables[name]
            if var.binding is not None:
                return var.binding
        if self._parent is not None:
            value = self._parent.resolve(name)
            self.capture(value)
            return value
        raise LoweringError(f"undefined: {name}")

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

# pycparser Constant.type -> (parser function, bits, signed).
# Matches the strings pycparser emits for integer literals with u/l/ll suffixes.
_LITERAL_TYPES: dict[str, tuple[type, ...]] = {
    "int": (parse_c_int, 32, True),
    "unsigned int": (parse_c_int, 32, False),
    "long int": (parse_c_int, 64, True),
    "unsigned long int": (parse_c_int, 64, False),
    "long long int": (parse_c_int, 64, True),
    "unsigned long long int": (parse_c_int, 64, False),
    "char": (parse_c_char, 8, True),
}


def _binop(cls: type[dgen.Op], a: dgen.Value, b: dgen.Value, ty: dgen.Type) -> dgen.Op:
    if "left" in cls.__dataclass_fields__:
        return cls(left=a, right=b, type=ty)
    return cls(lhs=a, rhs=b, type=ty)


# ---------------------------------------------------------------------------
# Dispatch registration
# ---------------------------------------------------------------------------

_EXPR_HANDLERS: dict[type[c_ast.Node], Callable[..., dgen.Value]] = {}
_STMT_HANDLERS: dict[type[c_ast.Node], Callable[..., object]] = {}


def _expr(
    node_type: type[c_ast.Node],
) -> Callable[[Callable[..., dgen.Value]], Callable[..., dgen.Value]]:
    """Register a Parser method as the expression handler for a node type."""

    def decorator(fn: Callable[..., dgen.Value]) -> Callable[..., dgen.Value]:
        _EXPR_HANDLERS[node_type] = fn
        return fn

    return decorator


def _stmt(
    node_type: type[c_ast.Node],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register a Parser method as the statement handler for a node type."""

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        _STMT_HANDLERS[node_type] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Translate pycparser AST to C-dialect IR."""

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
        if name in self.file_scope:
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

        # Create block args for function parameters. Each parameter is a
        # mutable local in C, so we declare it and emit an initial assign.
        scope = self.file_scope.child()
        arguments: list[BlockArgument] = []
        if isinstance(funcdef.decl.type, c_ast.FuncDecl):
            function_decl = funcdef.decl.type
            if function_decl.args:
                for parameter in function_decl.args.params:
                    if isinstance(parameter, c_ast.EllipsisParam):
                        continue
                    if isinstance(parameter, c_ast.Decl) and parameter.name:
                        param_type = self.types.resolve(parameter.type)
                        argument = BlockArgument(name=parameter.name, type=param_type)
                        arguments.append(argument)
                        lvalue = LvalueVarOp(
                            var_name=String().constant(parameter.name),
                            source=argument,
                            type=param_type,
                        )
                        assign = AssignOp(
                            lvalue=lvalue, rvalue=argument, type=param_type
                        )
                        scope.declare(parameter.name, param_type)
                        scope.record_write(parameter.name, assign)

        function_type = FunctionType(
            arguments=pack(arg.type for arg in arguments),
            result_type=return_type,
        )

        result = self._compound(funcdef.body, scope)
        block_result = result.value if isinstance(result, _Return) else result

        return function.FunctionOp(
            name=name,
            result_type=return_type,
            body=dgen.Block(result=block_result, args=arguments),
            type=function_type,
        )

    def _compound(self, node: c_ast.Compound, scope: Scope) -> dgen.Value | _Return:
        """Lower a compound statement. Returns the last statement's value.

        Stops processing on _Return (propagated upward) or jump ops
        (BreakOp, ContinueOp — remaining statements are dead code).
        Preceding effects are chained into the jump so they execute first.
        """
        last: dgen.Value | None = None
        if node.block_items:
            for item in node.block_items:
                result = self._statement(item, scope)
                if isinstance(result, _Return):
                    return result
                if isinstance(result.type, Never):
                    if last is not None:
                        return ChainOp(lhs=last, rhs=result, type=result.type)
                    return result
                last = result
        return last if last is not None else Nil().constant(None)

    # --- Statements ---

    def _statement(self, node: c_ast.Node, scope: Scope) -> dgen.Value | _Return:
        """Lower a statement. Returns a single Value or _Return."""
        handler = _STMT_HANDLERS.get(type(node))
        if handler is not None:
            return handler(self, node, scope)
        # Expression statement -- evaluate for side effects.
        return self._expression(node, scope)

    @_stmt(c_ast.Return)
    def _return_statement(self, node: c_ast.Return, scope: Scope) -> _Return:
        if node.expr is None:
            return _Return(self._current_return_type.constant(0))
        return _Return(self._expression(node.expr, scope))

    @_stmt(c_ast.Break)
    def _break_statement(self, node: c_ast.Break, scope: Scope) -> dgen.Value:
        return control_flow.BreakOp(
            loop_handler=scope.resolve_div(), arguments=pack([])
        )

    @_stmt(c_ast.Continue)
    def _continue_statement(self, node: c_ast.Continue, scope: Scope) -> dgen.Value:
        return control_flow.ContinueOp(
            loop_handler=scope.resolve_div(), arguments=pack([])
        )

    @_stmt(c_ast.Compound)
    def _compound_statement(
        self, node: c_ast.Compound, scope: Scope
    ) -> dgen.Value | _Return:
        return self._compound(node, scope.child())

    @_stmt(c_ast.Decl)
    def _declaration_statement(
        self, node: c_ast.Decl, scope: Scope
    ) -> dgen.Value | _Return:
        return self._declaration(node, scope)

    def _declaration(self, node: c_ast.Decl, scope: Scope) -> dgen.Value:
        """Lower a declaration. Emits assign only when an initializer exists."""
        if node.name is None:
            if node.type is not None:
                self.types.resolve(node.type)
            return Nil().constant(None)
        variable_type = self.types.resolve(node.type)
        scope.declare(node.name, variable_type)
        if node.init is None:
            return Nil().constant(None)
        initial_value = self._expression(node.init, scope)
        return self._assign_variable(node.name, initial_value, scope)

    @_stmt(c_ast.Assignment)
    def _assignment_statement(
        self, node: c_ast.Assignment, scope: Scope
    ) -> dgen.Value | _Return:
        return self._assignment(node, scope)

    def _assignment(self, node: c_ast.Assignment, scope: Scope) -> dgen.Value:
        """Lower an assignment expression. Returns the AssignOp."""
        rhs = self._expression(node.rvalue, scope)
        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(
                f"unsupported assignment target: {type(node.lvalue).__name__}"
            )
        return self._assign_variable(node.lvalue.name, rhs, scope)

    def _assign_variable(self, name: str, rhs: dgen.Value, scope: Scope) -> AssignOp:
        """Emit lvalue_var + assign for a variable write."""
        variable_type = scope.variable_type(name)
        ordering = scope.write_ordering(name)
        source = ordering if ordering is not None else rhs
        lvalue = LvalueVarOp(
            var_name=String().constant(name),
            source=source,
            type=variable_type,
        )
        assign = AssignOp(lvalue=lvalue, rvalue=rhs, type=variable_type)
        scope.record_write(name, assign)
        return assign

    # --- Control flow ---

    def _to_bool(self, value: dgen.Value) -> dgen.Value:
        """Convert a value to Boolean (for control flow conditions)."""
        if isinstance(value.type, Boolean):
            return value
        zero = value.type.constant(0)
        return _binop(algebra.NotEqualOp, value, zero, value.type)

    def _block_from(self, node: c_ast.Node, scope: Scope) -> dgen.Block:
        """Lower a statement into a Nil-typed Block for control flow bodies."""
        body = self._body_result(node, scope)
        nil = Nil().constant(None)
        return dgen.Block(
            result=ChainOp(lhs=nil, rhs=body, type=Nil()),
            captures=scope.captures,
        )

    def _body_result(self, node: c_ast.Node, scope: Scope) -> dgen.Value:
        """Lower control-flow body statements, packing independent results.

        For compound bodies, uses the provided scope directly (no extra
        child scope) so variable writes are visible to the caller.

        Returns a single value that transitively depends on every
        statement's result, keeping independent mutations alive.

        Jump ops (BreakOp, ContinueOp) terminate the statement sequence.
        Preceding effects are chained through the jump op so it becomes
        the block's effective result.
        """
        if isinstance(node, c_ast.Compound) and node.block_items:
            results: list[dgen.Value] = []
            for item in node.block_items:
                result = self._statement(item, scope)
                if isinstance(result, _Return):
                    raise LoweringError(
                        "return inside control flow not yet supported (Brick 6.5)"
                    )
                if isinstance(result.type, Never):
                    if results:
                        effects = pack(results) if len(results) != 1 else results[0]
                        return ChainOp(lhs=effects, rhs=result, type=result.type)
                    return result
                results.append(result)
            return pack(results) if len(results) != 1 else results[0]
        result = self._statement(node, scope)
        if isinstance(result, _Return):
            raise LoweringError(
                "return inside control flow not yet supported (Brick 6.5)"
            )
        return result

    def _propagate_cf_ordering(
        self,
        scope: Scope,
        child_scopes: list[Scope],
        cf_op: dgen.Value,
    ) -> None:
        """Propagate variable ordering from child scopes after control flow.

        Written variables: cf_op becomes the new ordering fence.
        Read-only variables: cf_op is added to pending reads so a subsequent
        write in the parent will fence after the cf_op.
        """
        all_writes: set[str] = set()
        all_reads: set[str] = set()
        for child in child_scopes:
            all_writes |= child.local_writes
            all_reads |= child.local_reads
        for name in all_writes:
            if scope.is_variable(name):
                scope.record_fence(name, cf_op)
        for name in all_reads - all_writes:
            if scope.is_variable(name):
                scope.record_read(name, cf_op)

    @_stmt(c_ast.If)
    def _if_statement(self, node: c_ast.If, scope: Scope) -> dgen.Value | _Return:
        cond = self._expression(node.cond, scope)
        bool_cond = self._to_bool(cond)

        then_scope = scope.child()
        then_block = self._block_from(node.iftrue, then_scope)

        else_scope = scope.child()
        if node.iffalse is not None:
            else_block = self._block_from(node.iffalse, else_scope)
        else:
            else_block = dgen.Block(result=Nil().constant(None))

        empty = pack([])
        if_op = control_flow.IfOp(
            condition=bool_cond,
            then_arguments=empty,
            else_arguments=empty,
            type=Nil(),
            then_body=then_block,
            else_body=else_block,
        )

        self._propagate_cf_ordering(scope, [then_scope, else_scope], if_op)
        return if_op

    @_stmt(c_ast.While)
    def _while_statement(self, node: c_ast.While, scope: Scope) -> dgen.Value | _Return:
        cond_scope = scope.child()
        cond_val = self._expression(node.cond, cond_scope)
        cond_block = dgen.Block(
            result=self._to_bool(cond_val), captures=cond_scope.captures
        )

        div = BlockParameter(name="div", type=control_flow.Loop())
        body_scope = scope.child()
        body_scope.set_div(div)
        body_block = self._block_from(node.stmt, body_scope)
        body_block.parameters.append(div)

        empty = pack([])
        while_op = control_flow.WhileOp(
            initial_arguments=empty, condition=cond_block, body=body_block
        )

        self._propagate_cf_ordering(scope, [cond_scope, body_scope], while_op)
        return while_op

    @_stmt(c_ast.For)
    def _for_statement(self, node: c_ast.For, scope: Scope) -> dgen.Value | _Return:
        # Init runs in current scope.
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                for decl in node.init.decls:
                    self._declaration(decl, scope)
            else:
                result = self._statement(node.init, scope)
                if isinstance(result, _Return):
                    raise LoweringError("return in for-init")

        # Condition block.
        cond_scope = scope.child()
        if node.cond is not None:
            cond_val = self._expression(node.cond, cond_scope)
            cond_block = dgen.Block(
                result=self._to_bool(cond_val), captures=cond_scope.captures
            )
        else:
            cond_block = dgen.Block(result=c_int(32).constant(1))

        # Body block: body statements + update appended.
        div = BlockParameter(name="div", type=control_flow.Loop())
        body_scope = scope.child()
        body_scope.set_div(div)
        body = self._body_result(node.stmt, body_scope)
        if node.next is not None:
            update = self._statement(node.next, body_scope)
            if isinstance(update, _Return):
                raise LoweringError("return in for-update")
            body = ChainOp(lhs=update, rhs=body, type=update.type)
        nil = Nil().constant(None)
        body_block = dgen.Block(
            result=ChainOp(lhs=nil, rhs=body, type=Nil()),
            captures=body_scope.captures,
            parameters=[div],
        )

        empty = pack([])
        while_op = control_flow.WhileOp(
            initial_arguments=empty, condition=cond_block, body=body_block
        )

        self._propagate_cf_ordering(scope, [cond_scope, body_scope], while_op)
        return while_op

    # --- Expressions ---

    def _expression(self, node: c_ast.Node, scope: Scope) -> dgen.Value:
        """Lower an expression. Returns a single Value."""
        handler = _EXPR_HANDLERS.get(type(node))
        if handler is not None:
            return handler(self, node, scope)
        raise LoweringError(f"unsupported expression: {type(node).__name__}")

    @_expr(c_ast.Constant)
    def _constant(self, node: c_ast.Constant, _scope: Scope) -> dgen.Value:
        literal = _LITERAL_TYPES.get(node.type)
        if literal is not None:
            parser_fn, bits, signed = literal
            return c_int(bits, signed).constant(parser_fn(node.value))
        if node.type in ("float", "double"):
            return Float64().constant(float(node.value.rstrip("fFlL")))
        raise LoweringError(f"unsupported literal type: {node.type}")

    @_expr(c_ast.ID)
    def _identifier(self, node: c_ast.ID, scope: Scope) -> dgen.Value:
        """Lower an identifier reference.

        Variables emit lvalue_var + lvalue_to_rvalue. The ordering token
        comes from the latest write (so back-to-back reads are independent),
        but we record the read so a subsequent write depends on it.
        """
        if node.name in self.types.enum_constants:
            return c_int(32).constant(self.types.enum_constants[node.name])
        if not scope.is_variable(node.name):
            return scope.resolve(node.name)
        variable_type = scope.variable_type(node.name)
        ordering = scope.read_ordering(node.name)
        source = ordering if ordering is not None else variable_type.constant(0)
        lvalue = LvalueVarOp(
            var_name=String().constant(node.name),
            source=source,
            type=variable_type,
        )
        load = LvalueToRvalueOp(lvalue=lvalue, type=variable_type)
        scope.record_read(node.name, load)
        return load

    @_expr(c_ast.BinaryOp)
    def _binary(self, node: c_ast.BinaryOp, scope: Scope) -> dgen.Value:
        if node.op in ("&&", "||"):
            return self._short_circuit(node, scope)

        lhs = self._expression(node.left, scope)
        rhs = self._expression(node.right, scope)

        cls = _ALGEBRA.get(node.op)
        if cls is None:
            raise LoweringError(f"unsupported binary op: {node.op}")

        if node.op in _COMPARISONS:
            comparison = _binop(cls, lhs, rhs, lhs.type)
            return algebra.CastOp(input=comparison, type=c_int(32))

        return _binop(cls, lhs, rhs, lhs.type)

    def _short_circuit(self, node: c_ast.BinaryOp, scope: Scope) -> dgen.Value:
        """Lower && and || to expression-level IfOp (short-circuit)."""
        lhs = self._expression(node.left, scope)
        bool_lhs = self._to_bool(lhs)

        # RHS evaluated lazily inside a block.
        rhs_scope = scope.child()
        rhs_val = self._expression(node.right, rhs_scope)
        rhs_cast = algebra.CastOp(input=self._to_bool(rhs_val), type=c_int(32))
        rhs_block = dgen.Block(result=rhs_cast, captures=rhs_scope.captures)

        i32 = c_int(32)
        one_block = dgen.Block(result=i32.constant(1))
        zero_block = dgen.Block(result=i32.constant(0))

        empty = pack([])

        if node.op == "&&":
            then_block = rhs_block
            else_block = zero_block
        else:
            then_block = one_block
            else_block = rhs_block

        return control_flow.IfOp(
            condition=bool_lhs,
            then_arguments=empty,
            else_arguments=empty,
            type=i32,
            then_body=then_block,
            else_body=else_block,
        )

    @_expr(c_ast.UnaryOp)
    def _unary(self, node: c_ast.UnaryOp, scope: Scope) -> dgen.Value:
        entry = _UNARY.get(node.op)
        if entry is not None:
            cls, field_name = entry
            inner = self._expression(node.expr, scope)
            return cls(**{field_name: inner, "type": inner.type})
        if node.op == "+":
            return self._expression(node.expr, scope)
        if node.op == "!":
            # C11 6.5.3.3p5: !E is equivalent to (0 == E), result type int.
            inner = self._expression(node.expr, scope)
            zero = inner.type.constant(0)
            equal = _binop(algebra.EqualOp, inner, zero, inner.type)
            return algebra.CastOp(input=equal, type=c_int(32))
        raise LoweringError(f"unsupported unary op: {node.op}")

    @_expr(c_ast.Cast)
    def _cast(self, node: c_ast.Cast, scope: Scope) -> dgen.Value:
        return self._expression(node.expr, scope)

    @_expr(c_ast.TernaryOp)
    def _ternary(self, node: c_ast.TernaryOp, scope: Scope) -> dgen.Value:
        raise LoweringError("ternary operator (?:) not yet supported")

    @_expr(c_ast.Assignment)
    def _assignment_expression(
        self, node: c_ast.Assignment, scope: Scope
    ) -> dgen.Value:
        return self._assignment(node, scope)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Return:
    """Sentinel that _statement returns when encountering a return."""

    __slots__ = ("value",)

    def __init__(self, value: dgen.Value) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(ast: c_ast.FileAST) -> function.FunctionOp:
    """Lower a pycparser FileAST to a FunctionOp."""
    return Parser().parse(ast)
