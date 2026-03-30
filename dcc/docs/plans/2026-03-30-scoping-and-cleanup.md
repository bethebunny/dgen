# Plan: Restructure dcc around clean scoping and high-level IR

**Date:** 2026-03-30
**Status:** Proposed (replaces 2026-03-29 plan)

## Problem

The parser (AST→IR translation) is ~730 lines of tangled logic that mixes three
concerns: AST traversal, name resolution, and semantic lowering. The result is
fragile, hard to review, and produces incorrect IR for many real-world patterns.

Specific issues:
- `self.scope` is a flat mutable dict that simulates block scoping but doesn't
  handle shadowing, nested scopes, or scope lifetimes.
- Semantic transformations (compound assignment expansion, comparison casting,
  null pointer matching, type promotion) are interleaved with structural
  AST mapping.
- `_closed_block` computes captures ad-hoc instead of deriving them from scope.

## Design

### Principle: the parser does name resolution, passes do everything else

The parser's job is to translate pycparser AST nodes to C-dialect ops with all
names resolved. Every `read_variable<"x">(var)` carries its resolved
`variable_declaration` or `assign` op as an operand, creating a use-def edge.
Every `assign<"x">(var, value)` carries the variable it's assigning to.

This means:
- The IR is fully connected through use-def edges (no dangling name references)
- `block.ops` sees all ops in dependency order
- No post-hoc name resolution pass is needed

Semantic transformations (type promotion, implicit casts, compound assignment
expansion) belong in lowering passes, not the parser.

### Scoping

C has four namespaces. The parser needs to handle one — ordinary identifiers
(variables and functions). Struct tags, members, and labels are handled by the
type resolver and by structural op parameters.

Within ordinary identifiers, C has nested block scopes with shadowing:

```c
int x = 1;           // scope 0: x → decl_0
{
    int x = 2;        // scope 1: x → decl_1 (shadows scope 0)
    x = x + 1;        // resolves to decl_1
}
x = x + 1;            // resolves to decl_0
```

The parser models this with a **scope stack**:

```python
class Scope:
    """Lexical scope for C name resolution."""

    def __init__(self, parent: Scope | None = None):
        self._bindings: dict[str, dgen.Value] = {}
        self._parent = parent

    def bind(self, name: str, value: dgen.Value) -> None:
        self._bindings[name] = value

    def lookup(self, name: str) -> dgen.Value:
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise LoweringError(f"undefined: {name}")

    def child(self) -> Scope:
        return Scope(parent=self)
```

The parser:
- Creates a root scope per function (containing parameters)
- Pushes a child scope for each `{}` compound statement
- Pops when leaving the compound
- `variable_declaration` binds in the current scope
- `assign` updates the binding in whichever scope owns the name
- `read_variable` looks up through the chain

### Assign updates the scope binding

When the parser sees `x = expr`, it:
1. Looks up `x` in the scope → gets the current binding (a `VariableDeclarationOp`
   or a previous `AssignOp`)
2. Emits `AssignOp(variable_name="x", target=current_binding, value=expr_result)`
3. Updates the scope: `scope.bind("x", assign_op)`

The next `read_variable<"x">` will get `assign_op` as its source operand,
creating the use-def edge: read → assign → previous read/decl → ... → declaration.

This is an SSA-like renaming scheme where each mutation creates a new value
that subsequent reads depend on. The CToMemory pass converts this chain into
alloca/store/load with memory tokens.

### Block captures from scope

When the parser builds a block (for if/while/for bodies), it knows exactly
which scope the block lives in. Captures are the scope bindings that the block's
ops reference — the scope stack makes this explicit.

Instead of the current `_closed_block` which scans op dependencies post-hoc,
the parser can compute captures directly: any value looked up from a parent
scope during block body lowering is a capture.

```python
class Scope:
    def __init__(self, parent: Scope | None = None):
        self._bindings: dict[str, dgen.Value] = {}
        self._parent = parent
        self.captures: list[dgen.Value] = []  # populated during lookup

    def lookup(self, name: str) -> dgen.Value:
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            val = self._parent.lookup(name)
            if val not in self.captures:
                self.captures.append(val)
            return val
        raise LoweringError(f"undefined: {name}")
```

When building a block, the parser creates a child scope, lowers the body,
then uses `scope.captures` as the block's captures. No post-hoc scanning.

### What the parser does NOT do

- **Type promotion**: `int + double` → the parser emits `algebra.add(int_val, double_val)`.
  A type resolution pass inserts casts.
- **Comparison result widening**: `a < b` produces an i1. A pass inserts `cast` to int.
- **Null pointer matching**: `ptr == 0` — the parser emits the literal 0 as int.
  A pass converts it to null.
- **Compound assignment expansion**: `x += 1` — the parser emits `c.compound_assign<"x", "+">(1)`.
  The CToMemory pass expands it to read + binop + store.
- **For→while conversion**: the parser emits `c.for_loop` with init/cond/update/body blocks.
  A pass converts it.

### C dialect ops (revised)

```dgen
# Variables — the source/target operand creates use-def edges for scoping
op variable_declaration<variable_name: String, variable_type: Type>(initializer) -> Type
op read_variable<variable_name: String>(source) -> Type
op assign<variable_name: String>(target, value) -> Type
op compound_assign<variable_name: String, operator: String>(target, value) -> Type

# Increment/decrement
op pre_increment<variable_name: String>(target) -> Type
op post_increment<variable_name: String>(target) -> Type
op pre_decrement<variable_name: String>(target) -> Type
op post_decrement<variable_name: String>(target) -> Type

# Pointer and array
op dereference(pointer) -> Type
op address_of(operand) -> Type
op subscript(base, index) -> Type

# Struct access
op member_access<field_name: String>(base) -> Type
op pointer_member_access<field_name: String>(base) -> Type

# Functions
op call<callee: String>(arguments: Span) -> Type
op return(value) -> Nil

# Arithmetic without shared dialect equivalent
op modulo(lhs, rhs) -> Type
op shift_left(lhs, rhs) -> Type
op shift_right(lhs, rhs) -> Type
op logical_not(operand) -> Type

# Control flow
op for_loop() -> Nil:
    block initializer
    block condition
    block update
    block body
op do_while(initial: Span) -> Nil:
    block body
    block condition
op switch<case_values: Span<Index>>(selector) -> Nil:
    block default_body
op break() -> Nil
op continue() -> Nil

# Misc
op sizeof<target_type: Type>() -> Type
op comma(lhs, rhs) -> Type
```

Note: `logical_not` and `compound_assign` are back — the parser shouldn't
expand these. They're C semantics that a pass handles.

### Passes

1. **CToMemory** — lower variable ops to memory dialect:
   - `variable_declaration` → `stack_allocate` + `store`
   - `read_variable` → `load` (mem token from source operand chain)
   - `assign` → `store`
   - `compound_assign` → `load` + binop + `store`
   - `pre/post_increment` → `load` + `add`/`sub` + `store`
   - `dereference` → `load`
   - `subscript` → `offset` + `load`
   - `member_access` / `pointer_member_access` → GEP + `load`

2. **CToStructured** — lower C control flow and arithmetic:
   - `for_loop` → init block inline + `control_flow.while`
   - `logical_not` → `algebra.equal(x, 0)` + `algebra.cast`
   - `modulo` → `sdiv` + `mul` + `sub`
   - `shift_left`/`shift_right` → llvm ops (until upstream)
   - `sizeof` → constant from layout
   - `call` → `llvm.call`
   - `return` → pass through (block result)
   - `comma` → second operand
   - `do_while`, `switch`, `break`, `continue` → stubs for now

3. **Existing passes** — `AlgebraToLLVM`, `MemoryToLLVM`, `ControlFlowToGoto`

### Parser structure

```python
class Parser:
    def __init__(self):
        self.types = TypeResolver()
        self.file_scope = Scope()      # functions live here

    def parse_function(self, node) -> FunctionOp:
        scope = self.file_scope.child()  # function body scope
        for param in params:
            scope.bind(param.name, BlockArgument(...))
        ops = list(self._compound(node.body, scope))
        ...

    def _compound(self, node, scope) -> Iterator[Op]:
        child = scope.child()
        for item in node.block_items:
            yield from self._statement(item, child)

    def _statement(self, node, scope) -> Iterator[Op]:
        if isinstance(node, c_ast.Decl):
            ...
            scope.bind(name, decl_op)
        elif isinstance(node, c_ast.Assignment):
            target = scope.lookup(name)
            ...
            scope.bind(name, assign_op)
        ...

    def _expression(self, node, scope) -> Iterator[Op]:
        if isinstance(node, c_ast.ID):
            val = scope.lookup(node.name)
            if is_variable(val):
                read = ReadVariableOp(source=val, ...)
                yield read
                return read
            return val  # function param
        ...
```

Every method takes `scope` as a parameter — no `self.scope` mutable state.
The scope stack is threaded explicitly through the call tree.

### File sizes (target)

| File | Target | Notes |
|------|--------|-------|
| `parser/lowering.py` | ~500 lines | Scope class + mechanical AST dispatch |
| `parser/type_resolver.py` | ~270 lines | Unchanged |
| `passes/c_to_memory.py` | ~200 lines | Variable + struct ops → memory |
| `passes/c_to_structured.py` | ~100 lines | C control flow + arithmetic → shared |
| `dialects/c.dgen` | ~60 lines | High-level C ops |

Total: ~1130 lines (currently ~1305).

The reduction is modest in lines but significant in complexity — the scope
stack eliminates `self.scope`, `_closed_block`, and the fragile
scope-update-on-assign pattern.

## Migration

1. Implement `Scope` class
2. Update parser to thread scope explicitly, remove `self.scope`
3. Add `compound_assign`, `logical_not`, `for_loop` to dialect
4. Split CToLLVM into CToStructured (control flow + arithmetic)
5. Update CToMemory for `compound_assign`
6. Verify with existing tests + sqlite3
