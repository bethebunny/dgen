"""Lower pycparser AST to dgen IR using the C dialect."""

from __future__ import annotations

from collections.abc import Iterator

from pycparser import c_ast

import dgen
from dgen.block import BlockArgument
from dgen.dialects import function
from dgen.dialects.builtin import Nil, String
from dgen.dialects.function import Function as FunctionType
from dgen.module import ConstantOp, Module, pack

from dgen.dialects import algebra, memory
from dgen.dialects.control_flow import IfOp, WhileOp
from dgen.dialects.number import Float64
from dgen_c.dialects import c_int
from dgen_c.dialects.c import (
    BreakOp,
    CallOp,
    ContinueOp,
    LognotOp,
    ModOp,
    ReturnValueOp,
    ReturnVoidOp,
    ShlOp,
    ShrOp,
    SizeofOp,
    StructMemberOp,
    StructPtrMemberOp,
    TernaryOp,
)
from dgen_c.parser.type_resolver import TypeResolver


# ---------------------------------------------------------------------------
# Binary op dispatch — algebra ops use left/right, C ops use lhs/rhs
# ---------------------------------------------------------------------------


def _binop(
    op_cls: type[dgen.Op], a: dgen.Value, b: dgen.Value, ty: dgen.Type
) -> dgen.Op:
    """Construct a binary op, handling algebra (left/right) vs C (lhs/rhs)."""
    if (
        hasattr(op_cls, "__dataclass_fields__")
        and "left" in op_cls.__dataclass_fields__
    ):
        return op_cls(left=a, right=b, type=ty)
    return op_cls(lhs=a, rhs=b, type=ty)


_BINOP_MAP: dict[str, type[dgen.Op]] = {
    "+": algebra.AddOp,
    "-": algebra.SubtractOp,
    "*": algebra.MultiplyOp,
    "/": algebra.DivideOp,
    "%": ModOp,
    "&": algebra.MeetOp,
    "|": algebra.JoinOp,
    "^": algebra.SymmetricDifferenceOp,
    "<<": ShlOp,
    ">>": ShrOp,
    "==": algebra.EqualOp,
    "!=": algebra.NotEqualOp,
    "<": algebra.LessThanOp,
    "<=": algebra.LessEqualOp,
    ">": algebra.GreaterThanOp,
    ">=": algebra.GreaterEqualOp,
    "&&": algebra.MeetOp,
    "||": algebra.JoinOp,
}

# Compound assignment: +=, -=, etc. -> base operator
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


def _closed_block(
    result: dgen.Value,
    args: list[BlockArgument] | None = None,
    *,
    local_ops: list[dgen.Op] | None = None,
) -> dgen.Block:
    """Build a Block with automatically computed captures.

    *local_ops* is the list of ops yielded during lowering of this block's
    body. Any value reachable from *result* that is not a local op, not a
    block arg, and not a Type is declared as a capture.
    """
    if args is None:
        args = []
    if local_ops is None:
        local_ops = []
    local_ids: set[int] = {id(op) for op in local_ops}
    local_ids |= {id(a) for a in args}

    captures: list[dgen.Value] = []
    seen: set[int] = set()

    def _maybe_capture(dep: dgen.Value) -> None:
        vid = id(dep)
        if vid in seen or vid in local_ids:
            return
        if isinstance(dep, dgen.Type):
            return
        seen.add(vid)
        captures.append(dep)

    # Check if result itself is external
    if id(result) not in local_ids and not isinstance(result, dgen.Type):
        _maybe_capture(result)

    # Walk all local ops and capture their external dependencies
    for op in local_ops:
        for dep in op.dependencies:
            _maybe_capture(dep)

    return dgen.Block(result=result, args=args, captures=captures)


class Lowering:
    """Lower a pycparser FileAST to a dgen Module."""

    def __init__(self) -> None:
        self.types = TypeResolver()
        # Per-function state
        self.scope: dict[str, dgen.Value] = {}
        # Per-variable memory token: tracks the last store to each variable
        self.var_mem: dict[str, dgen.Value] = {}
        # Global function declarations for call resolution
        self.func_types: dict[str, dgen.Type] = {}
        # Track all function ops for the module
        self.functions: list[function.FunctionOp] = []
        # Stats
        self.stats = LoweringStats()

    def _coerce(self, val: dgen.Value, target: dgen.Type) -> Iterator[dgen.Op]:
        """Insert a cast if *val*'s type doesn't match *target*.

        Handles C implicit conversions: int↔pointer, int↔float,
        width changes. Yields the cast op if needed, returns the
        (possibly new) value.
        """
        if isinstance(val.type, type(target)):
            return val
        cast = algebra.CastOp(input=val, type=target)
        yield cast
        return cast

    def _mem_for(self, name: str) -> dgen.Value:
        """Get the memory token for a variable (its last store, or its alloca)."""
        return self.var_mem.get(
            name, self.scope.get(name, ConstantOp(value=None, type=Nil()))
        )

    def lower_file(self, ast: c_ast.FileAST) -> Module:
        """Lower an entire C translation unit to a dgen Module."""
        # First pass: collect type definitions and function declarations
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef):
                self._process_typedef(ext)
            elif isinstance(ext, c_ast.Decl):
                if isinstance(ext.type, c_ast.FuncDecl):
                    self._process_func_decl(ext)
                elif isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                    self.types.resolve(ext.type)
                # Global variable declarations — skip for now

        # Second pass: lower function definitions
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                self.stats.functions += 1
                func_op = self._lower_func_def(ext)
                if func_op is not None:
                    self.functions.append(func_op)

        return Module(ops=list(self.functions))

    # -----------------------------------------------------------------------
    # Top-level declarations
    # -----------------------------------------------------------------------

    def _process_typedef(self, node: c_ast.Typedef) -> None:
        """Register a typedef."""
        if node.name is not None:
            resolved = self.types.resolve(node.type)
            self.types.register_typedef(node.name, resolved)
            self.stats.typedefs += 1

    def _process_func_decl(self, node: c_ast.Decl) -> None:
        """Register a function declaration (prototype)."""
        if node.name is not None:
            ret_type = self._get_func_return_type(node.type)
            self.func_types[node.name] = ret_type

    def _get_func_return_type(self, node: c_ast.Node) -> dgen.Type:
        """Extract the return type from a function declaration."""
        if isinstance(node, c_ast.FuncDecl):
            return self.types.resolve(node.type)
        if isinstance(node, c_ast.PtrDecl):
            return memory.Reference(element_type=self._get_func_return_type(node.type))
        return self.types.resolve(node)

    # -----------------------------------------------------------------------
    # Function definitions
    # -----------------------------------------------------------------------

    def _lower_func_def(self, node: c_ast.FuncDef) -> function.FunctionOp | None:
        """Lower a function definition to a FunctionOp."""
        self.scope = {}
        self.var_mem = {}

        func_name = node.decl.name
        ret_type = self._get_func_return_type(node.decl.type)
        self.current_ret_type = ret_type
        self.func_types[func_name] = ret_type

        # Create block args for function parameters
        args: list[BlockArgument] = []
        if isinstance(node.decl.type, c_ast.FuncDecl) and node.decl.type.args:
            for param in node.decl.type.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.Decl) and param.name:
                    param_type = self.types.resolve(param.type)
                    arg = BlockArgument(name=param.name, type=param_type)
                    self.scope[param.name] = arg
                    args.append(arg)

        # Lower the body
        ops: list[dgen.Op] = []
        if node.body is not None:
            ops.extend(self._lower_compound(node.body))

        # Determine result
        is_void = isinstance(ret_type, Nil)
        if is_void:
            result_type: dgen.Type = Nil()
        else:
            result_type = ret_type

        # Find the block result
        if ops:
            block_result: dgen.Value = ops[-1]
        else:
            block_result = dgen.Value(type=Nil())

        return function.FunctionOp(
            name=func_name,
            result=result_type,
            body=dgen.Block(result=block_result, args=args),
            type=FunctionType(result=result_type),
        )

    # -----------------------------------------------------------------------
    # Statements
    # -----------------------------------------------------------------------

    def _lower_compound(self, node: c_ast.Compound) -> Iterator[dgen.Op]:
        """Lower a compound statement (block)."""
        if node.block_items is None:
            return
        for item in node.block_items:
            yield from self._lower_stmt(item)

    def _lower_stmt(self, node: c_ast.Node) -> Iterator[dgen.Op]:
        """Lower a single statement."""
        self.stats.statements += 1

        if isinstance(node, c_ast.Decl):
            yield from self._lower_decl(node)
        elif isinstance(node, c_ast.Assignment):
            yield from self._lower_assignment(node)
        elif isinstance(node, c_ast.Return):
            yield from self._lower_return(node)
        elif isinstance(node, c_ast.If):
            yield from self._lower_if(node)
        elif isinstance(node, c_ast.While):
            yield from self._lower_while(node)
        elif isinstance(node, c_ast.DoWhile):
            yield from self._lower_do_while(node)
        elif isinstance(node, c_ast.For):
            yield from self._lower_for(node)
        elif isinstance(node, c_ast.Compound):
            yield from self._lower_compound(node)
        elif isinstance(node, c_ast.FuncCall):
            yield from self._lower_expr(node)
            # discard result — statement-level call
        elif isinstance(node, c_ast.UnaryOp):
            if node.op in ("p++", "p--", "++", "--"):
                yield from self._lower_expr(node)
        elif isinstance(node, c_ast.Goto):
            # C goto is unstructured control flow. For the prototype,
            # skip it — the target label's code will still run in sequence.
            pass
        elif isinstance(node, c_ast.Label):
            # Lower the labeled statement directly (skip the label itself).
            if node.stmt is not None:
                yield from self._lower_stmt(node.stmt)
        elif isinstance(node, c_ast.Switch):
            yield from self._lower_switch(node)
        elif isinstance(node, c_ast.Break):
            yield BreakOp()
        elif isinstance(node, c_ast.Continue):
            yield ContinueOp()
        elif isinstance(node, c_ast.EmptyStatement):
            pass  # no-op
        elif isinstance(node, c_ast.Typedef):
            self._process_typedef(node)
        elif isinstance(node, c_ast.Pragma):
            pass  # ignore pragmas
        elif isinstance(node, c_ast.Case):
            # Case labels inside switch — lower stmts
            if node.stmts is not None:
                for s in node.stmts:
                    yield from self._lower_stmt(s)
        elif isinstance(node, c_ast.Default):
            if node.stmts is not None:
                for s in node.stmts:
                    yield from self._lower_stmt(s)
        else:
            # Try to lower as expression
            try:
                yield from self._lower_expr(node)
            except Exception:
                self.stats.skipped_stmts += 1

    def _lower_decl(self, node: c_ast.Decl) -> Iterator[dgen.Op]:
        """Lower a local variable declaration."""
        if node.name is None:
            # Anonymous struct/union/enum definition
            if isinstance(node.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                self.types.resolve(node.type)
            return

        var_type = self.types.resolve(node.type)

        # Function declarations inside a function body
        if isinstance(node.type, c_ast.FuncDecl):
            self.func_types[node.name] = self._get_func_return_type(node.type)
            return

        # Allocate stack space
        alloca = memory.StackAllocateOp(
            element_type=var_type, type=memory.Reference(element_type=var_type)
        )
        yield alloca
        self.scope[node.name] = alloca

        # Initialize if there's an initializer
        if node.init is not None:
            init_val = yield from self._lower_expr(node.init)
            store = memory.StoreOp(mem=alloca, value=init_val, ptr=alloca)
            yield store
            self.var_mem[node.name] = store

    def _lower_assignment(self, node: c_ast.Assignment) -> Iterator[dgen.Op]:
        """Lower an assignment statement."""
        # Extract variable name for per-variable mem tracking
        var_name = node.lvalue.name if isinstance(node.lvalue, c_ast.ID) else None

        ptr = yield from self._lower_lvalue(node.lvalue)
        rhs = yield from self._lower_expr(node.rvalue)

        if node.op != "=":
            base_op = _COMPOUND_ASSIGN.get(node.op)
            if base_op is not None:
                mem = self._mem_for(var_name) if var_name else ptr
                current = memory.LoadOp(
                    mem=mem, ptr=ptr, type=self._expr_type(node.lvalue)
                )
                yield current
                op_cls = _BINOP_MAP.get(base_op)
                if op_cls is not None:
                    combined = _binop(op_cls, current, rhs, current.type)
                    yield combined
                    rhs = combined

        mem = self._mem_for(var_name) if var_name else ptr
        store = memory.StoreOp(mem=mem, value=rhs, ptr=ptr)
        yield store
        if var_name:
            self.var_mem[var_name] = store

    def _lower_return(self, node: c_ast.Return) -> Iterator[dgen.Op]:
        """Lower a return statement."""
        if node.expr is None:
            yield ReturnVoidOp()
        else:
            val = yield from self._lower_expr(node.expr)
            val = yield from self._coerce(val, self.current_ret_type)
            yield ReturnValueOp(value=val)

    def _lower_if(self, node: c_ast.If) -> Iterator[dgen.Op]:
        """Lower an if statement using control_flow.IfOp."""
        cond = yield from self._lower_expr(node.cond)

        # Lower then branch
        then_ops: list[dgen.Op] = list(self._lower_stmt(node.iftrue))
        then_result: dgen.Value = then_ops[-1] if then_ops else dgen.Value(type=Nil())
        then_block = _closed_block(then_result, local_ops=then_ops)

        # Lower else branch
        else_ops: list[dgen.Op] = []
        if node.iffalse is not None:
            else_ops = list(self._lower_stmt(node.iffalse))
        else_result: dgen.Value = else_ops[-1] if else_ops else dgen.Value(type=Nil())
        else_block = _closed_block(else_result, local_ops=else_ops)

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

    def _lower_while(self, node: c_ast.While) -> Iterator[dgen.Op]:
        """Lower a while loop using control_flow.WhileOp."""
        # Condition ops go inside the condition block, not the parent
        cond_ops: list[dgen.Op] = list(self._lower_expr(node.cond))
        cond: dgen.Value = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())
        body_ops: list[dgen.Op] = list(self._lower_stmt(node.stmt))
        body_result: dgen.Value = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        cond_block = _closed_block(cond, local_ops=cond_ops)
        body_block = _closed_block(body_result, local_ops=body_ops)

        p = pack([])
        yield p
        yield WhileOp(initial_arguments=p, condition=cond_block, body=body_block)

    def _lower_do_while(self, node: c_ast.DoWhile) -> Iterator[dgen.Op]:
        """Lower a do-while loop."""
        body_ops: list[dgen.Op] = list(self._lower_stmt(node.stmt))
        body_result: dgen.Value = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        cond_ops: list[dgen.Op] = list(self._lower_expr(node.cond))
        cond: dgen.Value = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())

        body_block = _closed_block(body_result, local_ops=body_ops)
        cond_block = _closed_block(cond, local_ops=cond_ops)

        p = pack([])
        yield p
        from dgen_c.dialects.c import DoWhileOp

        yield DoWhileOp(init=p, body=body_block, condition=cond_block)

    def _lower_for(self, node: c_ast.For) -> Iterator[dgen.Op]:
        """Lower a for loop as init + control_flow.WhileOp."""
        # Emit init statements
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                for decl in node.init.decls:
                    yield from self._lower_decl(decl)
            else:
                yield from self._lower_stmt(node.init)

        # Condition — collected into the condition block, not yielded to parent
        if node.cond is not None:
            cond_ops: list[dgen.Op] = list(self._lower_expr(node.cond))
            cond: dgen.Value = cond_ops[-1] if cond_ops else dgen.Value(type=Nil())
        else:
            cond = ConstantOp(value=1, type=c_int(32))
            cond_ops = [cond]
        cond_block = _closed_block(cond, local_ops=cond_ops)

        # Body = original body + update
        body_ops: list[dgen.Op] = list(self._lower_stmt(node.stmt))
        if node.next is not None:
            body_ops.extend(self._lower_stmt(node.next))
        body_result: dgen.Value = body_ops[-1] if body_ops else dgen.Value(type=Nil())
        body_block = _closed_block(body_result, local_ops=body_ops)

        p = pack([])
        yield p
        yield WhileOp(initial_arguments=p, condition=cond_block, body=body_block)

    def _lower_switch(self, node: c_ast.Switch) -> Iterator[dgen.Op]:
        """Lower a switch statement (simplified — flatten to if/else chain)."""
        yield from self._lower_expr(node.cond)

        # Just lower the body — cases become labels effectively
        if node.stmt is not None:
            yield from self._lower_stmt(node.stmt)

    # -----------------------------------------------------------------------
    # Expressions
    # -----------------------------------------------------------------------

    def _lower_expr(self, node: c_ast.Node) -> Iterator[dgen.Op]:
        """Lower an expression, yielding ops and returning the result Value.

        This is a generator that yields intermediate ops and returns
        (via StopIteration.value) the final dgen.Value for the expression.
        """
        self.stats.expressions += 1

        if isinstance(node, c_ast.Constant):
            return (yield from self._lower_constant(node))

        if isinstance(node, c_ast.ID):
            return (yield from self._lower_id(node))

        if isinstance(node, c_ast.BinaryOp):
            return (yield from self._lower_binop(node))

        if isinstance(node, c_ast.UnaryOp):
            return (yield from self._lower_unaryop(node))

        if isinstance(node, c_ast.FuncCall):
            return (yield from self._lower_func_call(node))

        if isinstance(node, c_ast.Assignment):
            return (yield from self._lower_assign_expr(node))

        if isinstance(node, c_ast.Cast):
            return (yield from self._lower_cast(node))

        if isinstance(node, c_ast.ArrayRef):
            return (yield from self._lower_array_ref(node))

        if isinstance(node, c_ast.StructRef):
            return (yield from self._lower_struct_ref(node))

        if isinstance(node, c_ast.TernaryOp):
            return (yield from self._lower_ternary(node))

        if isinstance(node, c_ast.ExprList):
            # Comma expression — evaluate all, return last
            result: dgen.Value = dgen.Value(type=Nil())
            for expr in node.exprs:
                result = yield from self._lower_expr(expr)
            return result

        if isinstance(node, c_ast.Compound):
            # GCC statement expression — lower body, return last
            ops = list(self._lower_compound(node))
            return ops[-1] if ops else dgen.Value(type=Nil())

        if isinstance(node, c_ast.CompoundLiteral):
            # (type){init} — lower init expression
            return (yield from self._lower_expr(node.init))

        if isinstance(node, c_ast.InitList):
            # {a, b, c} — lower first element for now
            if node.exprs:
                return (yield from self._lower_expr(node.exprs[0]))
            op = ConstantOp(value=0, type=c_int(32))
            yield op
            return op

        # Fallback: constant 0
        self.stats.skipped_exprs += 1
        op = ConstantOp(value=0, type=c_int(32))
        yield op
        return op

    def _lower_constant(self, node: c_ast.Constant) -> Iterator[dgen.Op]:
        """Lower a literal constant."""
        if node.type == "int":
            s = node.value.rstrip("uUlL")
            if s.startswith(("0x", "0X")):
                val = int(s, 16)
            elif len(s) > 1 and s.startswith("0") and s[1:].isdigit():
                val = int(s, 8)
            else:
                val = int(s)
            # Determine type from suffix
            suffix = node.value[len(s) :]
            if "ll" in suffix.lower() or "LL" in suffix:
                ty = c_int(64, signed="u" not in suffix.lower())
            elif "l" in suffix.lower():
                ty = c_int(64, signed="u" not in suffix.lower())
            elif "u" in suffix.lower():
                ty = c_int(32, signed=False)
            else:
                ty = c_int(32)
            op = ConstantOp(value=val, type=ty)
            yield op
            return op

        if node.type in ("float", "double"):
            s = node.value.rstrip("fFlL")
            val = float(s)
            from dgen.dialects.number import Float64

            op = ConstantOp(value=val, type=Float64())
            yield op
            return op

        if node.type == "char":
            ch = node.value[1:-1]  # strip surrounding quotes
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
            # String literals -> pointer to char
            # Store the string value as an integer constant (address placeholder)
            op = ConstantOp(value=0, type=memory.Reference(element_type=c_int(8)))
            yield op
            return op

        # Fallback
        op = ConstantOp(value=0, type=c_int(32))
        yield op
        return op

    def _lower_id(self, node: c_ast.ID) -> Iterator[dgen.Op]:
        """Lower an identifier reference.

        Local variables are allocas — reading them emits a LoadOp chained
        through the last side effect so stores are in the use-def graph.
        """
        # Check enum constants
        if node.name in self.types.enum_constants:
            val = self.types.enum_constants[node.name]
            op = ConstantOp(value=val, type=c_int(32))
            yield op
            return op

        # Check local scope
        if node.name in self.scope:
            val = self.scope[node.name]
            # Local variables (StackAllocateOp) need a load to read their value.
            # Function parameters (BlockArgument) with pointer type are values,
            # not memory locations — return them directly.
            if isinstance(val, memory.StackAllocateOp):
                elem = val.type.element_type
                if isinstance(elem, dgen.Type):
                    load = memory.LoadOp(
                        mem=self._mem_for(node.name), ptr=val, type=elem
                    )
                    yield load
                    return load
            return val

        # Unknown — return a named placeholder value
        return dgen.Value(name=node.name, type=c_int(64))

    _COMPARISON_OPS: set[str] = {"==", "!=", "<", "<=", ">", ">="}

    def _lower_binop(self, node: c_ast.BinaryOp) -> Iterator[dgen.Op]:
        """Lower a binary operation."""
        lhs = yield from self._lower_expr(node.left)
        rhs = yield from self._lower_expr(node.right)

        op_cls = _BINOP_MAP.get(node.op)
        if op_cls is None:
            return lhs

        result_type = self._promote_types(lhs.type, rhs.type)
        op = _binop(op_cls, lhs, rhs, result_type)
        yield op

        # Comparisons produce i1 in LLVM; cast to C int
        if node.op in self._COMPARISON_OPS:
            cast = algebra.CastOp(input=op, type=c_int(32))
            yield cast
            return cast

        return op

    def _lower_unaryop(self, node: c_ast.UnaryOp) -> Iterator[dgen.Op]:
        """Lower a unary operation."""
        if node.op == "sizeof":
            # sizeof(expr) or sizeof(type)
            if isinstance(node.expr, c_ast.Typename):
                target = self.types.resolve(node.expr)
            else:
                target = self._expr_type(node.expr)
            op = SizeofOp(target_type=target, type=c_int(64, signed=False))
            yield op
            return op

        if node.op == "&":
            # Address-of
            ptr = yield from self._lower_lvalue(node.expr)
            return ptr

        if node.op == "*":
            # Dereference
            inner = yield from self._lower_expr(node.expr)
            pointee = self._deref_type(inner.type)
            op = memory.LoadOp(mem=inner, ptr=inner, type=pointee)
            yield op
            return op

        # Pre/post increment/decrement
        if node.op in ("++", "p++"):
            var_name = node.expr.name if isinstance(node.expr, c_ast.ID) else None
            ptr = yield from self._lower_lvalue(node.expr)
            val_type = self._expr_type(node.expr)
            mem = self._mem_for(var_name) if var_name else ptr
            load = memory.LoadOp(mem=mem, ptr=ptr, type=val_type)
            yield load
            one = ConstantOp(value=1, type=val_type)
            yield one
            inc = algebra.AddOp(left=load, right=one, type=val_type)
            yield inc
            store = memory.StoreOp(mem=load, value=inc, ptr=ptr)
            yield store
            if var_name:
                self.var_mem[var_name] = store
            return load if node.op == "p++" else inc

        if node.op in ("--", "p--"):
            var_name = node.expr.name if isinstance(node.expr, c_ast.ID) else None
            ptr = yield from self._lower_lvalue(node.expr)
            val_type = self._expr_type(node.expr)
            mem = self._mem_for(var_name) if var_name else ptr
            load = memory.LoadOp(mem=mem, ptr=ptr, type=val_type)
            yield load
            one = ConstantOp(value=1, type=val_type)
            yield one
            dec = algebra.SubtractOp(left=load, right=one, type=val_type)
            yield dec
            store = memory.StoreOp(mem=load, value=dec, ptr=ptr)
            yield store
            if var_name:
                self.var_mem[var_name] = store
            return load if node.op == "p--" else dec

        # Standard unary ops
        inner = yield from self._lower_expr(node.expr)

        if node.op == "-":
            op = algebra.NegateOp(input=inner, type=inner.type)
            yield op
            return op

        if node.op == "~":
            op = algebra.ComplementOp(input=inner, type=inner.type)
            yield op
            return op

        if node.op == "!":
            op = LognotOp(operand=inner, type=c_int(32))
            yield op
            return op

        if node.op == "+":
            return inner

        return inner

    def _lower_func_call(self, node: c_ast.FuncCall) -> Iterator[dgen.Op]:
        """Lower a function call."""
        # Determine callee name
        if isinstance(node.name, c_ast.ID):
            callee_name = node.name.name
            ret_type = self.func_types.get(callee_name, c_int(32))

            # Lower arguments
            arg_vals: list[dgen.Value] = []
            if node.args is not None:
                for arg in node.args.exprs:
                    val = yield from self._lower_expr(arg)
                    arg_vals.append(val)

            p = pack(arg_vals) if arg_vals else pack([])
            yield p

            op = CallOp(
                callee=String().constant(callee_name),
                arguments=p,
                type=ret_type,
            )
            yield op
            return op

        # Indirect call (function pointer)
        callee = yield from self._lower_expr(node.name)
        arg_vals = []
        if node.args is not None:
            for arg in node.args.exprs:
                val = yield from self._lower_expr(arg)
                arg_vals.append(val)

        p = pack(arg_vals) if arg_vals else pack([])
        yield p

        from dgen_c.dialects.c import CallIndirectOp

        op = CallIndirectOp(callee=callee, arguments=p, type=c_int(64))
        yield op
        return op

    def _lower_assign_expr(self, node: c_ast.Assignment) -> Iterator[dgen.Op]:
        """Lower an assignment used as an expression (returns the assigned value)."""
        ptr = yield from self._lower_lvalue(node.lvalue)
        rhs = yield from self._lower_expr(node.rvalue)

        if node.op != "=":
            base_op = _COMPOUND_ASSIGN.get(node.op)
            if base_op is not None:
                current = memory.LoadOp(
                    mem=ptr, ptr=ptr, type=self._expr_type(node.lvalue)
                )
                yield current
                op_cls = _BINOP_MAP.get(base_op)
                if op_cls is not None:
                    combined = _binop(op_cls, current, rhs, current.type)
                    yield combined
                    rhs = combined

        store = memory.StoreOp(mem=ptr, value=rhs, ptr=ptr)
        yield store
        return rhs

    def _lower_cast(self, node: c_ast.Cast) -> Iterator[dgen.Op]:
        """Lower a type cast."""
        inner = yield from self._lower_expr(node.expr)
        target = self.types.resolve(node.to_type)
        op = algebra.CastOp(input=inner, type=target)
        yield op
        return op

    def _lower_array_ref(self, node: c_ast.ArrayRef) -> Iterator[dgen.Op]:
        """Lower an array subscript: a[i]."""
        base = yield from self._lower_expr(node.name)
        idx = yield from self._lower_expr(node.subscript)
        elem_type = self._deref_type(base.type)
        op = memory.OffsetOp(
            ptr=base, index=idx, type=memory.Reference(element_type=elem_type)
        )
        yield op
        # Load the element
        load = memory.LoadOp(mem=op, ptr=op, type=elem_type)
        yield load
        return load

    def _lower_struct_ref(self, node: c_ast.StructRef) -> Iterator[dgen.Op]:
        """Lower struct member access: s.field or s->field."""
        base = yield from self._lower_expr(node.name)
        field_name = node.field.name

        if node.type == "->":
            # Pointer dereference + member access
            field_type = self.types.get_struct_field_type(
                self._deref_type(base.type), field_name
            )
            op = StructPtrMemberOp(
                field_name=String().constant(field_name),
                base=base,
                type=field_type,
            )
            yield op
            return op

        # Direct member access
        field_type = self.types.get_struct_field_type(base.type, field_name)
        op = StructMemberOp(
            field_name=String().constant(field_name),
            base=base,
            type=field_type,
        )
        yield op
        return op

    def _lower_ternary(self, node: c_ast.TernaryOp) -> Iterator[dgen.Op]:
        """Lower a ternary expression: cond ? a : b."""
        cond = yield from self._lower_expr(node.cond)
        true_val = yield from self._lower_expr(node.iftrue)
        false_val = yield from self._lower_expr(node.iffalse)
        result_type = self._promote_types(true_val.type, false_val.type)
        op = TernaryOp(
            condition=cond, true_val=true_val, false_val=false_val, type=result_type
        )
        yield op
        return op

    # -----------------------------------------------------------------------
    # L-value lowering (produces a pointer to the target)
    # -----------------------------------------------------------------------

    def _lower_lvalue(self, node: c_ast.Node) -> Iterator[dgen.Op]:
        """Lower an lvalue expression, returning a pointer to the target."""
        if isinstance(node, c_ast.ID):
            if node.name in self.scope:
                val = self.scope[node.name]
                return val
            return dgen.Value(
                name=node.name, type=memory.Reference(element_type=c_int(64))
            )

        if isinstance(node, c_ast.UnaryOp) and node.op == "*":
            # *ptr — the pointer itself is the lvalue
            return (yield from self._lower_expr(node.expr))

        if isinstance(node, c_ast.ArrayRef):
            base = yield from self._lower_expr(node.name)
            idx = yield from self._lower_expr(node.subscript)
            elem_type = self._deref_type(base.type)
            op = memory.OffsetOp(
                ptr=base, index=idx, type=memory.Reference(element_type=elem_type)
            )
            yield op
            return op

        if isinstance(node, c_ast.StructRef):
            base = yield from self._lower_expr(node.name)
            field_name = node.field.name
            if node.type == "->":
                op = StructPtrMemberOp(
                    field_name=String().constant(field_name),
                    base=base,
                    type=memory.Reference(element_type=c_int(64)),  # address of field
                )
            else:
                op = StructMemberOp(
                    field_name=String().constant(field_name),
                    base=base,
                    type=memory.Reference(element_type=c_int(64)),
                )
            yield op
            return op

        # Fallback: lower as expression
        return (yield from self._lower_expr(node))

    # -----------------------------------------------------------------------
    # Type helpers
    # -----------------------------------------------------------------------

    def _expr_type(self, node: c_ast.Node) -> dgen.Type:
        """Infer the type of a C expression (best-effort)."""
        if isinstance(node, c_ast.Constant):
            if node.type in ("int",):
                return c_int(32)
            if node.type in ("float",):
                return Float64()
            if node.type in ("double",):
                return Float64()
            if node.type in ("char",):
                return c_int(8)
            return c_int(32)

        if isinstance(node, c_ast.ID):
            if node.name in self.scope:
                val = self.scope[node.name]
                # Local variables are allocas — their expression type is the element type
                if isinstance(val, memory.StackAllocateOp):
                    return val.type.element_type
                return val.type
            return c_int(32)

        if isinstance(node, c_ast.Cast):
            return self.types.resolve(node.to_type)

        return c_int(32)

    def _deref_type(self, ty: dgen.Type) -> dgen.Type:
        """Get the pointee type from a pointer type."""
        if isinstance(ty, memory.Reference):
            elem = ty.element_type
            if isinstance(elem, dgen.Type):
                return elem
        return c_int(64)

    def _promote_types(self, a: dgen.Type, b: dgen.Type) -> dgen.Type:
        """C-style type promotion (simplified)."""
        # Float wins over int
        if isinstance(a, Float64) or isinstance(b, Float64):
            return Float64()

        # Pointer wins
        if isinstance(a, memory.Reference):
            return a
        if isinstance(b, memory.Reference):
            return b

        # Default: use left type
        return a


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class LoweringStats:
    """Track lowering statistics."""

    def __init__(self) -> None:
        self.functions: int = 0
        self.typedefs: int = 0
        self.statements: int = 0
        self.expressions: int = 0
        self.skipped_stmts: int = 0
        self.skipped_exprs: int = 0

    def summary(self) -> str:
        return (
            f"Functions: {self.functions}, Typedefs: {self.typedefs}, "
            f"Statements: {self.statements}, Expressions: {self.expressions}, "
            f"Skipped stmts: {self.skipped_stmts}, Skipped exprs: {self.skipped_exprs}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(ast: c_ast.FileAST) -> tuple[Module, LoweringStats]:
    """Lower a pycparser FileAST to a dgen Module."""
    lowering = Lowering()
    module = lowering.lower_file(ast)
    return module, lowering.stats
