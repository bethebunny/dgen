# Plan: Restructure dcc around a high-level C dialect

**Date:** 2026-03-29
**Status:** Proposed

## Problem

The C frontend has two large files with tangled responsibilities:

- `parser/lowering.py` (~1050 lines) does AST translation, memory management
  (alloca/load/store/mem-token threading), and ad-hoc type inference all in one pass.
- `passes/c_to_llvm.py` (~110 lines) exists solely because the lowering emits ops
  at the wrong abstraction level.

The lowering directly emits `memory.StackAllocateOp`, `memory.LoadOp`,
`memory.StoreOp`, `algebra.AddOp`, etc. — skipping the C semantic level entirely.
There is no IR stage where you can look at the program and see "this is C."

Consequences:
- `_lower_id` has 20 lines of load/mem-token logic for variable reads.
- `_lower_decl` manually emits alloca + store + mem-token update.
- `_lower_assignment` re-derives the variable name from the AST to track per-variable
  mem tokens.
- Increment/decrement (`++`/`--`) is 15 lines each of load + arithmetic + store + mem
  update, duplicated for pre and post variants.
- `_expr_type`, `_promote_types`, `_match_ptr_int`, `_coerce`, `_deref_type` are
  ~100 lines of type helpers that are incomplete and ad-hoc.

## Proposed architecture

Three layers with clear boundaries:

```
pycparser AST  →  [parser/lowering.py]  →  C dialect IR
                                              ↓
                                       [CToMemory pass]  →  memory/algebra/control_flow IR
                                              ↓
                                       [existing passes]  →  LLVM IR
```

### Layer 1: Parser lowering (AST → C dialect)

A thin, mechanical translation. Each pycparser node maps to one C dialect op.
No memory ops, no type inference, no coercions. The lowering's job is structural
fidelity to the C source.

Example — `int y = x + 1; y = y * 2; return y;` becomes:

```
%y = c.variable_declaration<"y", i32>(%add_result)
%assign = c.assign(%y, %mul_result)
%ret = c.return(%y)
```

Not:

```
%alloca = memory.stack_allocate<i32>()
%store = memory.store(%alloca, %add_result, %alloca)
%load = memory.load(%store, %alloca)
...
```

**Target: ~200 lines.** No `_mem_for`, no `var_mem` dict, no `_expr_type`.

### Layer 2: C dialect (`dcc/dialects/c.dgen`)

High-level ops that model C semantics:

```dgen
# Variables
op variable_declaration<variable_name: String, variable_type: Type>(initializer) -> Type
op read_variable<variable_name: String>() -> Type
op assign<variable_name: String>(value) -> Type

# Increment / decrement
op pre_increment<variable_name: String>() -> Type
op post_increment<variable_name: String>() -> Type
op pre_decrement<variable_name: String>() -> Type
op post_decrement<variable_name: String>() -> Type

# Struct access (returns the field VALUE, not a pointer)
op member_access<field_name: String>(base) -> Type
op pointer_member_access<field_name: String>(base) -> Type

# Function calls and returns
op call<callee: String>(arguments: Span) -> Type
op return(value) -> Nil
op return_void() -> Nil

# C-specific control flow (if/while/for use control_flow dialect)
op do_while(initial: Span) -> Nil:
    block body
    block condition

# C-specific arithmetic (pending upstream algebra additions)
op modulo(lhs, rhs) -> Type
op shift_left(lhs, rhs) -> Type
op shift_right(lhs, rhs) -> Type
op logical_not(operand) -> Type

# Misc
op sizeof<target_type: Type>() -> Type
op ternary(condition, if_true, if_false) -> Type
```

**Types:** Only `CStruct` and `CUnion` remain. All other C types map to shared dgen
types (`number.SignedInteger`, `number.Float64`, `memory.Reference`, `builtin.Nil`,
etc.) — this is already done.

### Layer 3: CToMemory pass

One pass that lowers all C-specific ops to memory/algebra ops:

| C op | Lowered to |
|------|-----------|
| `variable_declaration<name, T>(init)` | `stack_allocate<T>()` + `store(alloc, init, alloc)` + register in var map |
| `read_variable<name>` | `load(var_mem[name], var_alloc[name])` |
| `assign<name>(val)` | `store(var_mem[name], val, var_alloc[name])` + update var_mem |
| `pre_increment<name>` | `load` + `add(_, 1)` + `store` |
| `post_increment<name>` | `load` + `add(_, 1)` + `store`, return pre-add value |
| `member_access<field>(base)` | GEP + load |
| `pointer_member_access<field>(base)` | GEP + load |
| `modulo(a, b)` | `sdiv` + `mul` + `sub` |
| `logical_not(x)` | `equal(x, 0)` + `cast` |
| `sizeof<T>` | constant (from layout) |
| `ternary(c, t, f)` | `control_flow.if` |
| `return(val)` | pass through (function body result) |
| `call<name>(args)` | `llvm.call` or `function.call` |

The pass owns the `var_mem` dict and mem-token threading — it's the single place
that manages memory state. **Target: ~150 lines.**

### Remaining: CResidualToLLVM

After CToMemory runs, the only C-dialect ops that should survive are `CStruct`
and `CUnion` types (used by type resolver for field lookup). If struct access
is fully lowered to GEP + load in CToMemory, this pass is eliminated entirely.

## What gets deleted

| File/section | Lines | Reason |
|---|---|---|
| `lowering._lower_id` load/mem logic | ~20 | Replaced by `read_variable` op |
| `lowering._lower_decl` alloca/store/mem | ~15 | Replaced by `variable_declaration` op |
| `lowering._lower_assignment` store/mem | ~25 | Replaced by `assign` op |
| `lowering._lower_unaryop` inc/dec | ~40 | Replaced by `pre_increment` etc. |
| `lowering._lower_struct_ref` GEP+load | ~25 | Replaced by `member_access` op |
| `lowering._expr_type` | ~25 | Not needed — types come from the IR |
| `lowering._promote_types` | ~15 | Moved to implicit cast pass (future) |
| `lowering._coerce` | ~10 | Moved to implicit cast pass (future) |
| `lowering._match_ptr_int` | ~10 | Moved to implicit cast pass (future) |
| `lowering._mem_for`, `var_mem` | ~10 | Moved to CToMemory pass |
| `lowering._closed_block` | ~25 | Moved to `dgen.Block` (core) |
| `passes/c_to_llvm.py` (entire file) | ~110 | Subsumed by CToMemory |
| **Total removed** | **~330** | |
| **New CToMemory pass** | **~150** | |
| **Net reduction** | **~180 lines** | |

Combined with the lowering shrinking from ~1050 to ~200 lines, the total goes from
~1160 to ~500 lines — a 57% reduction.

## Migration order

1. **Add new ops to `c.dgen`** — `variable_declaration`, `read_variable`, `assign`,
   `pre_increment`, `post_increment`, `member_access`, `pointer_member_access`, `return`.

2. **Write CToMemory pass** — lower new ops to memory/algebra ops. Test with small
   C programs (the existing 21 JIT end-to-end tests).

3. **Migrate lowering.py** — replace alloca/load/store emission with new C ops.
   One op type at a time, keeping tests green.

4. **Delete c_to_llvm.py** — everything it does is now in CToMemory.

5. **Move `_closed_block` to core** — it's useful for any frontend that builds
   blocks from statement lists.

Each step is independently testable. The 21 end-to-end JIT tests validate
correctness at every stage.

## Non-goals

- Full C99 compliance — this is a prototype for understanding dgen's IR properties.
- Global variables — blocked on upstream module-level declarations.
- Implicit cast pass — useful but orthogonal; belongs in dgen core.
- Indirect function calls — need function pointer type resolution.
