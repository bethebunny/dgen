# C Frontend Redesign: Design & Implementation Plan

*Draft v2 — revised after review by three agents (dgen architectural
fit, C11 spec coverage, implementation feasibility).*

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
6. **Incremental migration.** New ops coexist with old ops. Migrate
   one construct at a time. The sqlite3 ratchet must not regress.

## Architecture Overview

```
pycparser AST (untyped, unresolved)
      │
      ▼  Parser: thin 1:1 lowering
C dialect IR  (lvalue ops, raw arithmetic, no implicit conversions)
      │
      ▼  Pass: Struct layout (compute field offsets)
      ▼  Pass: Implicit conversion insertion (promotions, decay, casts)
      ▼  Pass: Lvalue elimination → memory ops (alloca hoisting)
      ▼  Pass: C arithmetic → algebra/LLVM (signed/unsigned dispatch)
      │
      ▼  Existing passes: ControlFlowToGoto, MemoryToLLVM, AlgebraToLLVM
      ▼  Codegen: LLVM IR text → JIT
```

### Pass Ordering Rationale

Struct layout **must precede** lvalue elimination because GEP indices
require field offsets. Implicit conversions **must precede** lvalue
elimination because the types of loads/stores depend on promotion
results. Integer promotions are part of implicit conversions, not a
separate late phase — every binary op's result type depends on them.

## C Dialect Design (`dcc/dialects/c.dgen`)

### Type System

C types are dgen types with correct memory layouts.

```dgen
from builtin import Index, Nil, String, Span, Pointer
from number import SignedInteger, UnsignedInteger, Float64

# Integers carry their C-level width. Layout uses the declared width.
# The existing number.SignedInteger/UnsignedInteger work for this.

# Float32 for C's `float` (distinct from Float64 / `double`).
type Float32:
    data: Float64  # layout: 32-bit float (needs a Float32 layout primitive)

# Struct: tag + field metadata. Frozen — offsets are computed by the
# struct layout pass, which constructs NEW Struct types with offsets
# filled in (frozen types cannot be mutated in place).
type Struct<tag: String, fields: Span<StructField>>:
    data: Index  # total size in bytes

type StructField<name: String, field_type: Type, offset: Index>:
    data: Nil

# Union: tag + field metadata. Size = max field size.
type Union<tag: String, fields: Span<StructField>>:
    data: Index

# Enum: underlying integer type + named constants.
type Enum<tag: String, underlying: Type>:
    data: Index

# Function type: reuse function.Function, but add variadic tracking.
# is_variadic: whether the function has `...` parameters.
# n_fixed_params: number of positional parameters before `...`.
# The implicit conversion pass uses these to decide whether to apply
# default argument promotions (variadic positions) vs parameter-type
# conversion (fixed positions).
type CFunctionType<arguments: Span<Type>, result_type: Type,
                   is_variadic: Index, n_fixed_params: Index>:
    data: Nil
```

### Lvalue Ops

C has two "worlds" of values. An **lvalue** designates a storage
location; an **rvalue** is a plain value. Variables, dereferences,
subscripts, and field accesses produce lvalues. An implicit
"lvalue-to-rvalue conversion" (a load) happens whenever an lvalue
appears in rvalue context.

Model lvalue-ness as **explicit ops in the IR**:

```dgen
# Lvalue-producing ops
op lvalue_var<name: String>(source)          # variable reference → lvalue
op lvalue_deref(ptr)                         # *ptr → lvalue
op lvalue_subscript(base, index)             # base[index] → lvalue
op lvalue_member<field: String>(base)        # base.field → lvalue
op lvalue_arrow<field: String>(ptr)          # ptr->field → lvalue
op lvalue_compound_literal(init)             # (struct S){...} → lvalue

# Lvalue-consuming ops
op lvalue_to_rvalue(lvalue)                  # the "load" — implicit in C
op address_of(lvalue)                        # &lvalue → pointer rvalue
op assign(lvalue, rvalue)                    # lvalue = rvalue
op compound_assign<operator: String>(lvalue, rvalue)
op pre_increment(lvalue)
op post_increment(lvalue)
op pre_decrement(lvalue)
op post_decrement(lvalue)
```

**Closed-block interaction:** When lvalue elimination hoists allocas to
function entry, inner blocks (if-bodies, loop-bodies) that reference
the alloca must list it as a **capture**. The lvalue elimination pass
is responsible for threading captures correctly — it scans the function
body, creates allocas, then for each inner block that references an
alloca, adds it to that block's capture list.

### Implicit Conversion Ops

```dgen
# Inserted by the implicit-conversion pass, not the parser
op integer_promote(input)           # char/short → int (C11 6.3.1.1)
op arithmetic_convert(input)        # usual arithmetic conversions (6.3.1.8)
op array_decay(array)               # T[] → T* (6.3.2.1p3)
op function_decay(func)             # f → &f (6.3.2.1p4)
op null_to_pointer(zero)            # 0 → null pointer (6.3.2.3p3)
op scalar_to_bool(val)              # any scalar → _Bool (6.3.1.2)
op bitfield_promote(input)          # bit-field → int (6.3.1.1)
```

### Control Flow

Reuse `control_flow.IfOp`, `control_flow.WhileOp`, and
`control_flow.ForOp` from dgen. These lower to goto labels via
`ControlFlowToGoto`.

**Short-circuit `&&`/`||`:** These are NOT bitwise operations. They
require short-circuit evaluation (C11 6.5.13-14). The parser must
lower them to `control_flow.IfOp`:

```python
# a && b  →  if (a) then b else 0
# a || b  →  if (a) then 1 else b
```

This is the parser's job (not a pass) because the AST structure
directly determines control flow.

C-specific control flow ops:

```dgen
op c_return(value) -> Nil
op c_switch<case_values: Span<Index>>(selector) -> Nil:
    block default_body
op c_goto<label: String>() -> Nil
op c_label<name: String>() -> Nil
```

**`c_return` and ChainOp interaction:** The lowering of `c_return`
must preserve the side-effect chain. When the parser emits
`ChainOp(lhs=return_val, rhs=prior_effects)`, the `c_return` wraps
this chain, and `lower_return` extracts the chain (not just the bare
value). The block's result IS the `c_return` op, which carries the
full chain.

### Calls

`function.CallOp` with `ExternOp` callee for ALL cross-function calls
(even to functions defined in the same translation unit). This avoids
DAG cycles from mutual recursion — ExternOps are leaf nodes with no
body to recurse into. FunctionOps provide definitions; ExternOps
provide callable handles. Codegen deduplicates `declare` vs `define`
for the same symbol.

## Passes

### Pass 1: Struct Layout (`c_struct_layout.py`)

**Runs first** because both implicit conversions and lvalue elimination
need offset/size information.

Compute field offsets using System V AMD64 ABI rules:
- Each field at next offset satisfying its alignment.
- Struct alignment = max field alignment.
- Struct size padded to alignment multiple.
- Union: all fields at offset 0, size = max field size.

Since dgen types are `frozen=True` dataclasses, the pass constructs
**new** `Struct`/`Union` type instances with offsets filled in and
uses `block.replace_uses_of(old_type, new_type)` to swap them
throughout the IR.

**Deferred:** Bit-field layout, flexible array members, `_Alignas`.
These are documented limitations, not silent stubs.

### Pass 2: Implicit Conversion Insertion (`c_implicit_conversions.py`)

Walk every op. At each "conversion point" (per C11 spec):

- **Binary arithmetic ops:** Apply integer promotions to both operands
  (6.3.1.1), then usual arithmetic conversions (6.3.1.8). The full
  algorithm: float types dominate → else promote both to int → then
  reconcile signed/unsigned by rank.
- **Assignments (6.5.16.1):** Convert RHS to LHS type.
- **Function args (6.5.2.2):** For fixed positions, convert to parameter
  type. For variadic positions (and unprototyped functions), apply
  default argument promotions: integer promotions + `float` → `double`.
  Uses `CFunctionType.is_variadic` and `n_fixed_params` to distinguish.
- **Return (6.8.6.4):** Convert to function's declared return type.
- **Conditions (if/while/for):** Insert `scalar_to_bool`.
- **Pointer comparisons:** Insert `null_to_pointer` when comparing
  pointer to integer 0.

**Array decay (6.3.2.1p3):** Insert `array_decay` for array-typed
values in expression context, EXCEPT:
- Operand of `sizeof` → parser resolves `sizeof` on the undecayed
  type BEFORE this pass runs (sizeof is a compile-time computation
  resolved in the parser using type layout).
- Operand of `address_of` → no decay (address-of an array gives
  a pointer to the array, not a pointer to the first element).
- String literal initializing `char[]` → no decay.

**Function decay (6.3.2.1p4):** Insert `function_decay` for
function-typed values, EXCEPT operand of `address_of`.

### Pass 3: Lvalue Elimination (`c_lvalue_to_memory.py`)

For each function body:

1. **Alloca hoisting:** Scan all `lvalue_var` ops (across all nested
   blocks, including if/loop bodies). Create ONE
   `memory.StackAllocateOp` per unique variable name, placed as a
   dependency of the function body's result (not "at function entry"
   — in dgen's sea-of-nodes, allocas with no operands naturally sort
   first in topological order).
2. **Capture threading:** For each inner block that references a
   hoisted alloca, add the alloca to that block's captures list.
3. **Lower lvalue ops** to memory ops (load/store/GEP).

### Pass 4: C Arithmetic → LLVM (`c_to_llvm.py`)

Lower remaining C-specific ops with correct signed/unsigned dispatch:
- `c.modulo` → `llvm.srem` or `llvm.urem`
- `c.shift_left` → `llvm.shl`
- `c.shift_right` → `llvm.ashr` (signed) or `llvm.lshr` (unsigned)
- `c.logical_not` → `icmp eq %x, 0`
- `c.sizeof<T>` → constant (T's layout size)
- `c.c_return` → extract value from chain, return it (chain kept
  alive by being the block result's dependency)

## Parser Design (`dcc/parser/lowering.py`)

The parser is a **thin, mechanical translator** from pycparser AST to
C dialect IR. It does NOT insert implicit conversions, resolve type
promotions, compute struct offsets, or decide load vs store.

It DOES:
- Create scope bindings (reuse existing Scope class)
- Emit lvalue ops for lvalue-producing expressions
- Wrap lvalues in `lvalue_to_rvalue` in rvalue context
- Suppress `lvalue_to_rvalue` for: LHS of `=`, operand of `&`,
  operand of `sizeof` (and later `_Alignof`)
- Lower `&&`/`||` to `IfOp` (short-circuit control flow)
- Chain all statements via `ChainOp` (solves unreachable-ops problem)
- Resolve `sizeof(expr)` to the expression's undecayed type before
  any pass runs, using the type's layout size directly
- Emit `function.CallOp` with resolved ExternOp callees
- Track the current function's return type for `c_return` ops

### Lvalue/Rvalue Decision

The parser knows which AST nodes produce lvalues (C11 6.5):
- `c_ast.ID` (variable reference) → `lvalue_var`
- `c_ast.UnaryOp("*", ...)` (dereference) → `lvalue_deref`
- `c_ast.ArrayRef` (subscript) → `lvalue_subscript`
- `c_ast.StructRef(".", ...)` → `lvalue_member`
- `c_ast.StructRef("->", ...)` → `lvalue_arrow`
- `c_ast.CompoundLiteral` → `lvalue_compound_literal`
- String literals → lvalue of `char[N]` type
- Everything else → rvalue

## Type Resolver

Mostly reused. Changes:
- **Struct field metadata:** `StructField(name, type, offset=0)` tuples.
  Offset filled in by struct layout pass (via type replacement since
  types are frozen).
- **Integer widths:** Track actual C widths. `char` = 8, `short` = 16,
  `int` = 32, `long`/`long long` = 64.
- **`float` vs `double`:** Distinct types (`Float32` vs `Float64`).
- **Function types:** `CFunctionType` with `is_variadic` and
  `n_fixed_params`. `EllipsisParam` records the variadic boundary
  instead of being silently skipped.
- **Parameter type adjustment (6.7.6.3p7-8):** Array parameters
  adjusted to pointer, function parameters adjusted to pointer-to-function.

## Side Effect Chaining

The "unreachable ops" problem (lessons.md) is solved by the parser
chaining all statements via `ChainOp`. `ChainOp(lhs=X, rhs=Y)` has
runtime value = `lhs`, with a dependency on `rhs`. The accumulation
pattern:

```python
def _compound(self, node, scope):
    chain = None
    for stmt in node.block_items:
        val = yield from self._stmt(stmt, scope)
        if val is None:
            continue
        if chain is not None:
            val = ChainOp(lhs=val, rhs=chain, type=val.type)
        chain = val
    return chain
```

Every statement's value chains to all prior statements. The final
value is the block's result, with every side effect reachable via
`rhs` edges.

## Implementation Plan

### Strategy: Incremental Migration (Not Big-Bang)

New lvalue ops are added to `c.dgen` **alongside** existing ops. The
parser is migrated **one expression kind at a time** (variables first,
then assignments, then struct access). Old `@lowering_for` handlers
and new ones coexist in the pipeline. At each step, the full test
suite passes. Only after all constructs are migrated do we delete the
old ops.

### Step 0: Preparation

- Add new lvalue ops and conversion ops to `c.dgen` (additive, no
  existing code changes).
- Add `Float32` type, `CFunctionType` with variadic tracking.
- Add `llvm.shl`, `llvm.ashr`, `llvm.lshr`, `llvm.srem`, `llvm.urem`
  to `llvm.dgen`.

### Step 1: Variable read/write via lvalues

Change `_id` to emit `lvalue_var` + `lvalue_to_rvalue` (in rvalue
context) instead of `ReadVariableOp`. Add `@lowering_for(LvalueVarOp)`
in `CToMemory` that does what `lower_read_variable` does today.

**First test:** `test_local_variable` — `int f(void) { int x = 5; return x; }`

### Step 2: Assignment via lvalues

Change `_assign` to emit `assign(lvalue, rvalue)` instead of `AssignOp`.
Add `@lowering_for(AssignOp_new)` handler.

**Test:** `test_local_mutation`

### Step 3: Struct access via lvalues

Change `_struct` to emit `lvalue_member`/`lvalue_arrow`. Add struct
layout computation for GEP offsets.

**Test:** `test_struct_field_access`

### Step 4: `&&`/`||` as IfOp

Replace `MeetOp`/`JoinOp` mapping for `&&`/`||` with `IfOp`-based
short-circuit evaluation.

**Test:** new test for `int f(int *p) { return p && *p; }`

### Step 5: Implicit conversion pass

Add `c_implicit_conversions.py`. Handle integer promotions, usual
arithmetic conversions, and array/function decay.

### Step 6: Side-effect chaining

Add `ChainOp` threading in `_compound`. Fix `c_return` to preserve
chains.

### Step 7: Cleanup

Remove old ops from `c.dgen`. Delete `ReadVariableOp`, old `AssignOp`,
old `CallOp` (already done), old `CompoundAssignOp` handlers. Update
snapshots.

### Step 8: sqlite3 push

Run sqlite3 codegen with new pipeline. Target >90% verified.
Add proper sizeof, switch, do_while, goto, break/continue.

## Known Limitations (v1)

These are **documented** limitations, not silent stubs:
- No bit-field layout (fields after a bit-field will have wrong offsets)
- No `_Alignas` / `_Alignof`
- No `_Atomic`
- No `_Generic`
- No VLA (variable-length arrays)
- No `restrict` optimization
- No complex number support (`_Complex`)
- No `goto` across scopes (computed goto)
- Flexible array members not sized

## What We Keep

| Component | Status |
|-----------|--------|
| `Scope` class | Keep as-is |
| `TypeResolver` structure | Keep ~80%, fix struct + float + variadic |
| Test suite + sqlite3 infrastructure | Keep as-is |
| `algebra_to_llvm.py` (ptrtoint/inttoptr) | Keep |
| `DUMP_IR_FOR`, ratchets, xfails | Keep |
| ExternOp-based callee architecture | Keep (from this session's refactor) |

## What We Delete (after migration)

| Component | Reason |
|-----------|--------|
| `ReadVariableOp` / old `AssignOp` | Replaced by lvalue ops |
| Inline isinstance conversion checks | Replaced by conversion pass |
| `_promote()` | Replaced by conversion pass |
| `&&`/`||` → MeetOp/JoinOp | Replaced by IfOp short-circuit |
| GEP index=0 struct stub | Replaced by layout pass |
| Shift-as-multiply/divide | Replaced by llvm.shl/ashr/lshr |
| sizeof = 8 | Replaced by layout-based sizeof |

## Risks

1. **Snapshot churn.** Every test's IR snapshot changes. Mitigated by
   incremental migration + `--snapshot-update` + graph-equivalence.
2. **Alloca hoisting + captures.** Threading hoisted allocas as
   captures into inner blocks is the riskiest piece — prototype it
   on `test_local_variable` with an if-branch early.
3. **Struct layout complexity.** Alignment/padding rules are fiddly.
   Mitigated by following System V AMD64 ABI exactly. Bit-fields are
   an explicit deferral.
4. **Frozen types.** Struct layout pass must construct new type
   instances (not mutate). This is unusual for a dgen pass but
   architecturally sound (type replacement via `replace_uses_of`).
