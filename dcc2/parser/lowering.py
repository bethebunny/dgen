"""Translate pycparser AST to C-dialect IR.

Thin 1:1 lowering from pycparser AST to dgen ops. Semantic transforms
(type promotions, struct layout, lvalue elimination) belong in passes.

Every statement handler returns a single Value whose transitive
dependencies contain every op the statement produced. Expression
handlers use generators internally but the statement layer does not
return lists — graph structure is preserved in the use-def edges.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, control_flow, function
from dgen.dialects.builtin import ChainOp, ExternOp, Nil, String
from dgen.dialects.function import Function as FunctionType
from dgen.dialects.memory import Reference
from dgen.dialects.number import Boolean, Float64
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
        self._local_reads: set[str] = set()
        self._local_writes: set[str] = set()
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
        self._local_reads.add(name)

    def record_write(self, name: str, write_op: dgen.Value) -> None:
        """Record a write. Clears pending reads and advances the chain."""
        self._bindings[name] = write_op
        self._last_write[name] = write_op
        self._pending_reads[name] = []
        self._local_writes.add(name)

    def record_ordering(self, name: str, fence_op: dgen.Value) -> None:
        """Record an ordering fence (e.g. control flow op) without updating
        the variable's binding. Advances the write chain so subsequent
        reads/writes depend on the fence."""
        self._last_write[name] = fence_op
        self._pending_reads[name] = []
        self._local_writes.add(name)

    @property
    def local_reads(self) -> set[str]:
        """Variable names read in this scope (not inherited from parent)."""
        return self._local_reads

    @property
    def local_writes(self) -> set[str]:
        """Variable names written in this scope (not inherited from parent)."""
        return self._local_writes

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
    }

    # Maps pycparser statement node type -> handler method name.
    _STMT_DISPATCH: dict[type[c_ast.Node], str] = {
        c_ast.Return: "_return_statement",
        c_ast.Decl: "_declaration_statement",
        c_ast.Assignment: "_assignment_statement",
        c_ast.Compound: "_compound_statement",
        c_ast.If: "_if_statement",
        c_ast.While: "_while_statement",
        c_ast.For: "_for_statement",
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

        result = self._compound(funcdef.body, scope)
        block_result = result.value if isinstance(result, _Return) else result

        return function.FunctionOp(
            name=name,
            result_type=return_type,
            body=dgen.Block(result=block_result, args=arguments),
            type=function_type,
        )

    def _compound(self, node: c_ast.Compound, scope: Scope) -> dgen.Value | _Return:
        """Lower a compound statement. Returns the last statement's value."""
        last: dgen.Value | None = None
        if node.block_items:
            for item in node.block_items:
                result = self._statement(item, scope)
                if isinstance(result, _Return):
                    return result
                last = result
        return last if last is not None else ConstantOp(value=0, type=Nil())

    # --- Statements ---

    def _statement(self, node: c_ast.Node, scope: Scope) -> dgen.Value | _Return:
        """Lower a statement. Returns a single Value or _Return."""
        handler_name = self._STMT_DISPATCH.get(type(node))
        if handler_name is not None:
            return getattr(self, handler_name)(node, scope)
        # Expression statement -- evaluate for side effects.
        return _run_gen(self._expression(node, scope))

    def _return_statement(self, node: c_ast.Return, scope: Scope) -> _Return:
        if node.expr is None:
            return _Return(ConstantOp(value=0, type=self._current_return_type))
        return _Return(_run_gen(self._expression(node.expr, scope)))

    def _compound_statement(
        self, node: c_ast.Compound, scope: Scope
    ) -> dgen.Value | _Return:
        return self._compound(node, scope.child())

    def _declaration_statement(
        self, node: c_ast.Decl, scope: Scope
    ) -> dgen.Value | _Return:
        return self._declaration(node, scope)

    def _declaration(self, node: c_ast.Decl, scope: Scope) -> dgen.Value:
        """Lower a declaration (e.g. int x = 5;). Returns the AssignOp."""
        if node.name is None:
            if node.type is not None:
                self.types.resolve(node.type)
            return ConstantOp(value=0, type=Nil())
        variable_type = self.types.resolve(node.type)
        if node.init is not None:
            initial_value = _run_gen(self._expression(node.init, scope))
        else:
            initial_value = ConstantOp(value=0, type=variable_type)
        lvalue = LvalueVarOp(
            var_name=String().constant(node.name),
            source=initial_value,
            type=variable_type,
        )
        assign = AssignOp(lvalue=lvalue, rvalue=initial_value, type=variable_type)
        scope.declare_variable(node.name, assign)
        return assign

    def _assignment_statement(
        self, node: c_ast.Assignment, scope: Scope
    ) -> dgen.Value | _Return:
        return self._assignment(node, scope)

    def _assignment(self, node: c_ast.Assignment, scope: Scope) -> dgen.Value:
        """Lower an assignment expression. Returns the AssignOp."""
        rhs = _run_gen(self._expression(node.rvalue, scope))

        if not isinstance(node.lvalue, c_ast.ID):
            raise LoweringError(
                f"unsupported assignment target: {type(node.lvalue).__name__}"
            )
        variable_name = node.lvalue.name
        variable_type = scope.lookup(variable_name).type
        lvalue = LvalueVarOp(
            var_name=String().constant(variable_name),
            source=scope.write_source(variable_name),
            type=variable_type,
        )
        assign = AssignOp(lvalue=lvalue, rvalue=rhs, type=variable_type)
        scope.record_write(variable_name, assign)
        return assign

    # --- Control flow ---

    def _to_bool(self, value: dgen.Value) -> Generator[dgen.Op, None, dgen.Value]:
        """Convert a value to Boolean (for control flow conditions)."""
        if isinstance(value.type, Boolean):
            return value
        yield (zero := ConstantOp(value=0, type=value.type))
        yield (cmp := _binop(algebra.NotEqualOp, value, zero, value.type))
        return cmp

    def _block_from(self, node: c_ast.Node, scope: Scope) -> dgen.Block:
        """Lower a statement into a Nil-typed Block for control flow bodies."""
        return self._nil_block(self._body_result(node, scope), scope)

    def _nil_block(self, body: dgen.Value, scope: Scope) -> dgen.Block:
        """Wrap a body value in a Nil-typed block result."""
        nil = ConstantOp(value=0, type=Nil())
        result = ChainOp(lhs=nil, rhs=body, type=Nil())
        return dgen.Block(result=result, captures=scope.captures)

    def _body_result(self, node: c_ast.Node, scope: Scope) -> dgen.Value:
        """Lower control-flow body statements, chaining all results.

        For compound bodies, uses the provided scope directly (no extra
        child scope) so variable writes are visible to the caller.

        Returns a single value that transitively depends on every
        statement's result, keeping independent mutations alive.
        """
        if isinstance(node, c_ast.Compound) and node.block_items:
            results: list[dgen.Value] = []
            for item in node.block_items:
                result = self._statement(item, scope)
                if isinstance(result, _Return):
                    raise LoweringError(
                        "return inside control flow not yet supported (Brick 6.5)"
                    )
                results.append(result)
            return _chain_all(results)
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

        Written variables: cf_op becomes the new ordering fence (last_write).
        Read-only variables: cf_op is added to pending reads so a subsequent
        write in the parent will fence after the cf_op.

        Uses record_ordering (not record_write) to avoid overwriting the
        variable's binding/type with the Nil-typed control flow op.
        """
        all_writes: set[str] = set()
        all_reads: set[str] = set()
        for child in child_scopes:
            all_writes |= child.local_writes
            all_reads |= child.local_reads
        for name in all_writes:
            if scope.is_variable(name):
                scope.record_ordering(name, cf_op)
        for name in all_reads - all_writes:
            if scope.is_variable(name):
                scope.record_read(name, cf_op)

    def _if_statement(self, node: c_ast.If, scope: Scope) -> dgen.Value | _Return:
        cond = _run_gen(self._expression(node.cond, scope))
        bool_cond = _run_gen(self._to_bool(cond))

        then_scope = scope.child()
        then_block = self._block_from(node.iftrue, then_scope)

        else_scope = scope.child()
        if node.iffalse is not None:
            else_block = self._block_from(node.iffalse, else_scope)
        else:
            else_block = dgen.Block(result=ConstantOp(value=0, type=Nil()))

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

    def _while_statement(self, node: c_ast.While, scope: Scope) -> dgen.Value | _Return:
        # Condition block: evaluate condition in child scope, convert to bool.
        cond_scope = scope.child()
        cond_val = _run_gen(self._expression(node.cond, cond_scope))
        cond_result = _run_gen(self._to_bool(cond_val))
        cond_block = dgen.Block(result=cond_result, captures=cond_scope.captures)

        # Body block.
        body_scope = scope.child()
        body_block = self._block_from(node.stmt, body_scope)

        empty = pack([])
        while_op = control_flow.WhileOp(
            initial_arguments=empty, condition=cond_block, body=body_block
        )

        self._propagate_cf_ordering(scope, [cond_scope, body_scope], while_op)
        return while_op

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
            cond_val = _run_gen(self._expression(node.cond, cond_scope))
            cond_result = _run_gen(self._to_bool(cond_val))
            cond_block = dgen.Block(result=cond_result, captures=cond_scope.captures)
        else:
            cond_block = dgen.Block(result=ConstantOp(value=1, type=c_int(32)))

        # Body block: body statements + update appended.
        body_scope = scope.child()
        body_result = self._body_result(node.stmt, body_scope)
        if node.next is not None:
            update = self._statement(node.next, body_scope)
            if isinstance(update, _Return):
                raise LoweringError("return in for-update")
            body_result = ChainOp(lhs=update, rhs=body_result, type=update.type)
        body_block = self._nil_block(body_result, body_scope)

        empty = pack([])
        while_op = control_flow.WhileOp(
            initial_arguments=empty, condition=cond_block, body=body_block
        )

        self._propagate_cf_ordering(scope, [cond_scope, body_scope], while_op)
        return while_op

    # --- Expressions ---

    def _expression(
        self, node: c_ast.Node, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower an expression. Dispatches to handler by node type."""
        # Assignment is a plain function, not a generator.
        if isinstance(node, c_ast.Assignment):
            return self._assignment(node, scope)
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
        if node.op in ("&&", "||"):
            return (yield from self._short_circuit(node, scope))

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

    def _short_circuit(
        self, node: c_ast.BinaryOp, scope: Scope
    ) -> Generator[dgen.Op, None, dgen.Value]:
        """Lower && and || to expression-level IfOp (short-circuit)."""
        lhs = yield from self._expression(node.left, scope)
        yield from (bool_ops := list(self._to_bool(lhs)))
        bool_lhs = bool_ops[-1] if bool_ops else lhs

        # RHS evaluated lazily inside a block.
        rhs_scope = scope.child()
        rhs_val = _run_gen(self._expression(node.right, rhs_scope))
        rhs_bool_ops = list(self._to_bool(rhs_val))
        rhs_bool = rhs_bool_ops[-1] if rhs_bool_ops else rhs_val
        rhs_cast = algebra.CastOp(input=rhs_bool, type=c_int(32))
        rhs_block = dgen.Block(result=rhs_cast, captures=rhs_scope.captures)

        one_block = dgen.Block(result=ConstantOp(value=1, type=c_int(32)))
        zero_block = dgen.Block(result=ConstantOp(value=0, type=c_int(32)))

        empty = pack([])
        yield empty

        if node.op == "&&":
            then_block = rhs_block
            else_block = zero_block
        else:
            then_block = one_block
            else_block = rhs_block

        op = control_flow.IfOp(
            condition=bool_lhs,
            then_arguments=empty,
            else_arguments=empty,
            type=c_int(32),
            then_body=then_block,
            else_body=else_block,
        )
        yield op
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


def _chain_all(values: list[dgen.Value]) -> dgen.Value:
    """Chain a list of values so the result depends on all of them.

    Independent values (e.g. mutations to different variables) have no
    use-def edge between them. ChainOp makes the last value depend on
    all earlier ones, keeping them alive in the block.
    """
    if not values:
        return ConstantOp(value=0, type=Nil())
    result = values[0]
    for v in values[1:]:
        result = ChainOp(lhs=v, rhs=result, type=v.type)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(ast: c_ast.FileAST) -> function.FunctionOp:
    """Lower a pycparser FileAST to a FunctionOp."""
    return Parser().parse(ast)
