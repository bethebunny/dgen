# C Frontend Redesign: Design & Implementation Plan

## Problem Statement

The current C frontend is a single-pass AST→IR translator that conflates
parsing, type resolution, implicit conversion insertion, and IR construction.
Each C feature requires another `isinstance` special case. The result is
~30% hacks/workarounds by line count, struct field offsets hardcoded to 0,
no lvalue model, no integer promotions, and a scope system that conflates
storage identity with current value.

This document proposes a clean redesign following dgen's own patterns
(as demonstrated by the Toy dialect), targeting full C11 conformance where
it matters for correct compilation.

## Design Principles

1. **Work with dgen, not around it.** Use the pass framework, staging
   model, closed blocks, and sea-of-nodes semantics as designed.
2. **Each concern gets exactly one pass.** No implicit conversion
   insertion in the parser. No type promotion interleaved with lowering.
3. **The C dialect models C semantics.** Lvalues, implicit conversions,
   and type qualifiers are explicit ops, not special cases.
4. **Follow the Toy dialect pattern.** Two-phase types (unresolved →
   resolved), generator-based lowering, `@lowering_for` passes.
5. **Correctness over completeness.** Get the common subset right before
   chasing corner cases. But design the architecture so corner cases
   slot in without hacks.

## Architecture Overview

```
pycparser AST (untyped, unresolved)
      │
      ▼  Phase 1: Type-annotated lowering
C dialect IR  (lvalue ops, unresolved promotions, explicit decay)
      │
      ▼  Pass: Implicit conversion insertion
C dialect IR  (all casts explicit, types consistent)
      │
      ▼  Pass: Lvalue elimination → memory ops
Memory dialect IR  (alloca/load/store, GEP with real offsets)
      │
      ▼  Pass: C arithmetic → algebra/LLVM
Algebra + LLVM dialect IR
      │
      ▼  Pass: Control flow → goto
Goto dialect IR
      │
      ▼  Codegen
LLVM IR text → JIT
```

## C Dialect Design (`dcc/dialects/c.dgen`)

### Type System

C types are dgen types with correct memory layouts.

```dgen
from builtin import Index, Nil, String, Span, Pointer
from number import SignedInteger, UnsignedInteger, Float64

# Integers carry their C-level width. Layout uses the declared width.
# The existing number.SignedInteger/UnsignedInteger work for this;
# we alias for readability.

# Struct type: carries field names, types, and offsets.
type Struct<tag: String, fields: Span<StructField>>:
    data: Index  # total size

type StructField<name: String, field_type: Type, offset: Index>:
    data: Nil

# Enum: underlying integer type + named constants
type Enum<tag: String, underlying: Type>:
    data: Index

# Function type is already provided by function.Function.
# Qualified types (const/volatile) are tracked via wrapper types
# or metadata — defer to v2.
```

### Lvalue Ops

The key insight: C has two "worlds" of values. An **lvalue** designates
a storage location; an **rvalue** is a plain value. Most expressions
produce rvalues. Variables, dereferences, subscripts, and field accesses
produce lvalues. An implicit "lvalue-to-rvalue conversion" (a load)
happens whenever an lvalue appears in rvalue context.

Rather than tracking lvalue-ness as a type property, model it as
**explicit ops in the IR**:

```dgen
# Lvalue-producing ops
op lvalue_var<name: String>(source)          # variable reference → lvalue
op lvalue_deref(ptr)                         # *ptr → lvalue
op lvalue_subscript(base, index)             # base[index] → lvalue
op lvalue_member<field: String>(base)        # base.field → lvalue
op lvalue_arrow<field: String>(ptr)          # ptr->field → lvalue

# Lvalue-consuming ops
op lvalue_to_rvalue(lvalue)                  # the "load" — implicit in C
op address_of(lvalue)                        # &lvalue → pointer rvalue
op assign(lvalue, rvalue)                    # lvalue = rvalue
op compound_assign<operator: String>(lvalue, rvalue)  # lvalue op= rvalue
op pre_increment(lvalue)
op post_increment(lvalue)
op pre_decrement(lvalue)
op post_decrement(lvalue)
```

This makes every load/store explicit in the IR. The "lvalue elimination"
pass lowers `lvalue_to_rvalue` → `memory.load`, `assign` → `memory.store`,
etc. But until that pass runs, the IR preserves C's lvalue semantics —
which means passes can reason about aliasing, const-correctness, and
volatile access at the C level.

### Implicit Conversion Ops

C requires implicit conversions at specific points (assignments, function
args, returns, comparisons). Rather than inserting these in the parser,
the parser emits raw ops and a dedicated pass inserts conversion ops:

```dgen
# Inserted by the implicit-conversion pass, not the parser
op integer_promote(input)           # char/short → int
op arithmetic_convert(input)        # usual arithmetic conversions
op array_decay(array)               # array → pointer to first element
op function_decay(func)             # function → pointer to function
op null_to_pointer(zero)            # integer 0 → null pointer
op pointer_to_bool(ptr)             # pointer → _Bool (for conditions)
```

Each of these is lowered trivially by later passes (e.g., `integer_promote`
→ `llvm.sext` or no-op if already `int`-width).

### Control Flow

Reuse `control_flow.IfOp`, `control_flow.WhileOp`, and
`control_flow.ForOp` from the existing dgen dialects. These already
lower to `goto.label`/`goto.branch` via `ControlFlowToGoto`.

Add C-specific:

```dgen
op c_return(value) -> Nil
op c_switch<case_values: Span<Index>>(selector) -> Nil:
    block default_body
# break/continue are handled by control-flow-to-goto lowering
```

### Calls

Follow the architecture from the current refactor: `function.CallOp`
with `ExternOp` or `FunctionOp` callee. No string-named calls.

## Passes

### Pass 1: Implicit Conversion Insertion (`c_implicit_conversions.py`)

**Input:** Raw C dialect IR from the parser.
**Output:** C dialect IR with all conversions explicit.

Walk every op. At each "conversion point" (per C11 spec):
- **Binary ops:** Apply usual arithmetic conversions to both operands.
- **Assignments:** Convert RHS to LHS type.
- **Function args:** Convert each arg to parameter type (or apply default
  promotions for variadic/unprototyped).
- **Return:** Convert to function's declared return type.
- **Conditions (if/while/for):** Convert to `_Bool` (non-zero test).
- **Comparisons:** Apply usual arithmetic conversions.

For pointer operands in comparisons, insert `null_to_pointer` when
comparing to integer 0.

This pass replaces ALL of the current inline isinstance-based conversion
logic in `_binary`, `_ret`, etc.

### Pass 2: Lvalue Elimination (`c_lvalue_to_memory.py`)

**Input:** C dialect IR with explicit lvalue ops.
**Output:** Memory dialect IR (alloca/load/store/GEP).

For each function body:
1. **Alloca hoisting:** Scan all `lvalue_var` ops. Create ONE
   `memory.StackAllocateOp` per unique variable name at function entry.
   This is the "slot" for the variable, shared across all branches.
2. **Lower lvalue ops:**
   - `lvalue_var(source)` → the alloca ptr
   - `lvalue_deref(ptr)` → ptr itself
   - `lvalue_subscript(base, idx)` → `llvm.GepOp(base, idx)`
   - `lvalue_member<field>(base)` → `llvm.GepOp(base, field_offset)`
   - `lvalue_arrow<field>(ptr)` → `llvm.GepOp(ptr, field_offset)`
3. **Lower lvalue consumers:**
   - `lvalue_to_rvalue(lv)` → `memory.LoadOp(ptr=lv)`
   - `assign(lv, rv)` → `memory.StoreOp(ptr=lv, value=rv)`
   - `address_of(lv)` → lv itself (it's already a pointer)
   - `pre_increment(lv)` → load + add 1 + store, return new value
   - `post_increment(lv)` → load + add 1 + store, return old value

**Alloca hoisting solves the slot-collision bug** (#2 bucket, ~240 errors).
Currently, each branch body creates its own alloca for the same variable.
With hoisting, there's exactly one alloca per variable per function.

### Pass 3: Struct Layout (`c_struct_layout.py`)

Compute field offsets from the struct's field types and alignment rules.
Attach offsets to `StructField` metadata. This runs before lvalue
elimination so GEP indices are available.

Rules (System V AMD64 ABI):
- Each field placed at next offset satisfying its alignment.
- Struct alignment = max field alignment.
- Struct size padded to alignment multiple.

### Pass 4: C Arithmetic → LLVM (`c_to_llvm.py`)

Lower remaining C-specific ops:
- `c.modulo` → `llvm.srem` (signed) or `llvm.urem` (unsigned)
- `c.shift_left` → `llvm.shl`
- `c.shift_right` → `llvm.ashr` (signed) or `llvm.lshr` (unsigned)
- `c.logical_not` → `icmp eq %x, 0`
- `c.sizeof<T>` → constant (layout.size)

The signed/unsigned dispatch uses the operand type's signedness,
which is available from the type (SignedInteger vs UnsignedInteger).

### Existing passes (reused as-is)

- `AlgebraToLLVM` — algebra ops → LLVM ops
- `MemoryToLLVM` — memory.load/store → llvm.load/store
- `ControlFlowToGoto` — structured control flow → goto labels
- `LLVMCodegen` — LLVM IR emission and JIT

## Parser Design (`dcc/parser/lowering.py`)

The parser becomes a **thin, dumb translator** from pycparser AST to
C dialect IR. It does NOT:
- Insert implicit conversions
- Resolve type promotions
- Compute struct offsets
- Decide load vs store (that's lvalue elimination's job)

It DOES:
- Create scope bindings (reuse existing Scope class)
- Emit lvalue ops for lvalue expressions, rvalue ops for rvalue expressions
- Emit `function.CallOp` with resolved ExternOp/FunctionOp callees
- Track the current function's return type for `c_return` ops
- Emit control flow ops (`IfOp`, `WhileOp`, `ForOp`)

### Lvalue/Rvalue Decision

The parser knows which AST nodes produce lvalues:
- `c_ast.ID` (variable reference) → `lvalue_var`
- `c_ast.UnaryOp("*", ...)` (dereference) → `lvalue_deref`
- `c_ast.ArrayRef` (subscript) → `lvalue_subscript`
- `c_ast.StructRef(".", ...)` → `lvalue_member`
- `c_ast.StructRef("->", ...)` → `lvalue_arrow`
- Everything else → rvalue

When an lvalue appears in rvalue context (e.g., RHS of `+`), the parser
wraps it in `lvalue_to_rvalue`. When it appears in lvalue context (LHS
of `=`, operand of `&`), the parser does NOT wrap it.

This is a **mechanical decision** based on AST position, not a semantic
analysis. The parser doesn't need to understand type rules.

## Type Resolver

Mostly reused. Fix:
- **Struct field metadata:** Store `(name, type, offset_placeholder)` tuples
  in `CStruct`. Offset is `None` until the struct layout pass runs.
- **Integer widths:** Track actual C widths (8, 16, 32, 64) via
  `SignedInteger`/`UnsignedInteger` bits parameter.
- **Qualification:** Track const/volatile as boolean flags on types (v2).

## Side Effect Chaining

The "unreachable ops" problem (lessons.md) is solved by the parser
chaining all statements via `ChainOp`:

```python
def _compound(self, node, scope):
    result = None
    for stmt in node.block_items:
        val = yield from self._stmt(stmt, scope)
        if result is not None and val is not None:
            val = ChainOp(lhs=val, rhs=result, type=val.type)
        if val is not None:
            result = val
    return result
```

Every statement's result is chained to the next. Side-effecting ops
(stores, calls) are always reachable from the block's result.

## Implementation Plan

### Phase 1: Foundation (types + lvalue model + parser rewrite)

1. Rewrite `c.dgen` with lvalue ops, explicit conversion ops, and
   proper struct type.
2. Rewrite `lowering.py` as a thin AST→IR translator. Emit lvalue ops,
   chain all statements, use ExternOp callees.
3. Write the implicit conversion pass (handles promotions, decay,
   null-to-pointer).
4. Write the lvalue elimination pass (alloca hoisting, load/store).
5. Compute struct field offsets.
6. Existing tests should pass with updated snapshots.

### Phase 2: Arithmetic + control flow

7. Proper shift/modulo with signed/unsigned dispatch.
8. Verify control flow lowering works with new IR shape.
9. Run sqlite3 codegen — target >90% verified.

### Phase 3: Polish

10. Integer promotion pass (char/short → int).
11. Volatile access support.
12. Improved error messages (reject undefined functions, type mismatches).
13. Run sqlite3 codegen — target >95% verified.

## What We Keep

| Component | Status |
|-----------|--------|
| `Scope` class | Keep as-is |
| `TypeResolver` | Keep structure, fix struct metadata |
| `type_resolver.py` | Keep ~80%, fix struct and qualification |
| `c_to_memory.py` (memory token threading) | Rewrite as lvalue elimination |
| `c_to_llvm.py` | Rewrite with signed/unsigned dispatch |
| `algebra_to_llvm.py` (ptrtoint/inttoptr) | Keep |
| Test suite | Keep all tests, update snapshots |
| sqlite3 test infrastructure | Keep as-is |
| `DUMP_IR_FOR`, ratchets, xfails | Keep |

## What We Delete

| Component | Reason |
|-----------|--------|
| Inline isinstance conversion checks in parser | Replaced by conversion pass |
| `_promote()` | Replaced by conversion pass |
| `_named_calls` tracking | Already removed |
| `_resolve_callee_captures` | Already removed |
| String-named `c.CallOp` | Already removed |
| Struct field offset stubs (GEP index=0) | Replaced by layout pass |
| Shift-as-multiply/divide | Replaced by proper llvm.shl/ashr/lshr |

## Risks

1. **Snapshot churn.** Every test's IR snapshot changes. Mitigated by
   `--snapshot-update` and graph-equivalence checking.
2. **Scope of rewrite.** Phase 1 touches parser + 3 new passes + dialect.
   Mitigated by keeping existing tests as the correctness oracle.
3. **Struct layout complexity.** Alignment/padding rules are fiddly.
   Mitigated by following System V AMD64 ABI exactly, which is
   well-documented and matches what gcc -E produces.
