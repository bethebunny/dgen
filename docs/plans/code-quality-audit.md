# Code Quality Audit

Audit of overfit, fragile, and iteratively-developed-to-pass-tests patterns across the
codebase. Organized by impact: correctness risks first, then extensibility blockers,
then cross-cutting API hygiene, then low-priority observations.

Items marked **[TODO.md]** are already tracked there.

---

## Phase 1 — Correctness and silent-failure risks

These issues can silently produce wrong output. Fix first.

### 1a. Hardcoded "i64" fallback type

`types.get(val, "i64")` appears at six sites in codegen.py (lines 140, 173, 489, 533,
616, 628). If a value's type is missing from the `types` dict — due to a pass bug, a new
op, or an incomplete lowering — codegen silently emits `i64` instead of crashing. Type
errors surface as wrong LLVM IR at runtime (segfaults, wrong results) rather than as
clear errors at compile time.

**Files:** `dgen/codegen.py` (6 sites)

**Fix:** Replace every `types.get(val, "i64")` with a helper that raises on missing type
with a diagnostic (op name, value name, available types). If there are legitimate cases
where a value has no LLVM type (e.g. `ChainOp`), handle those explicitly rather than via
a catch-all default.

### 1b. Fragile type reconstruction from dicts

`_type_from_dict` in `dgen/type.py:74–104` and `_resolve_layout` in
`dgen/layout.py:260–278` both parse type tags via `tag.split(".")` with no validation,
bare dict access, and no error context. A malformed tag produces an unhelpful
`ValueError` or `KeyError` deep in the stack.

**Files:** `dgen/type.py:74–104`, `dgen/layout.py:260–278`

**Fix:** Extract tag parsing into a shared `parse_type_tag(tag: str) -> tuple[str, str]`
that validates format and raises a clear error. Wrap `Dialect.get()` and
`dialect.types[type_name]` to produce messages like
`"Unknown type tag 'foo.Bar': dialect 'foo' not registered"`.

### 1c. Stage classification logic duplicated

The condition "is this value stage 0?" is checked in `compute_stages` (line 175–176,
isinstance against `(Constant, Type, FunctionOp, BlockParameter)`) and in
`_unresolved_boundaries` (line 223–225, isinstance against
`(Op, BlockArgument, BlockParameter)` then excluding `(Constant, Type, FunctionOp)`).
These are logically the same predicate expressed differently. If one is updated without
the other, staging silently misclassifies values.

**Files:** `dgen/staging.py:151–233`

**Fix:** Extract a single `is_comptime(value: Value) -> bool` predicate that both
functions call.

---

## Phase 2 — Extensibility blockers

These cause friction every time a new op or builtin is added.

### 2a. Codegen isinstance dispatch ladder **[TODO.md]**

`_emit_op` in codegen.py (lines 527–607) is an 80-line `elif` chain. Every new LLVM op
requires a new branch, even though most follow one of four patterns: binary arith
(`fadd`, `add`, `sub`, …), comparison (`icmp`, `fcmp`), unary (`fneg`, `zext`), memory
(`load`, `store`, `alloca`, `gep`).

**Files:** `dgen/codegen.py:527–607`

**Fix:** Dispatch table keyed on op class or `(dialect_name, asm_name)`. Define pattern
functions for each family:

```python
def _binary_float(name, op, refs):
    return f"  %{name} = fadd double {refs.lhs}, {refs.rhs}"
```

Or better: ops declare their LLVM instruction template as a class attribute so the table
is co-located with the op definition. The `else` clause already raises `ValueError`, so
the table approach preserves error handling.

### 2b. Codegen monolithic `_emit_func` **[TODO.md]**

`_emit_func` (lines 192–656) is a 556-line closure with 7+ mutable dicts (`if_blocks`,
`if_phis`, `if_merge_targets`, `param_to_label`, `predecessors`, `types`, `constants`).
Three implicit phases: label/op separation, linearization with IfOp expansion, emit.

**Files:** `dgen/codegen.py:192–656`

**Fix:** Extract into a `FuncEmitter` class with the mutable dicts as instance
attributes. Each phase becomes a method: `linearize()`, `build_predecessors()`,
`emit()`. Phase boundaries become explicit and each piece independently testable.

### 2c. Builtin function dispatch in lowering

`_lower_call` in `toy/parser/lowering.py:159–226` has six hardcoded string comparisons
(`transpose`, `tile`, `nonzero_count`, `concat`, `dim_size`, `add_index`) each with
duplicated arity-checking boilerplate. Adding a new builtin requires copying ~10 lines.

**Files:** `toy/parser/lowering.py:159–226`

**Fix:** Table-driven builtin registry:

```python
_BUILTINS: dict[str, tuple[int, Callable]] = {
    "transpose": (1, _lower_transpose),
    "tile":      (2, _lower_tile),
    ...
}
```

Dispatch becomes a 5-line lookup. Better yet: encode arity in the op definition itself
so builtins are self-describing.

### 2d. Magic string parameter names **[TODO.md]**

`param.name.startswith("exit")` (codegen.py:464) and `param.name == "self"`
(codegen.py:474) are magic-string contracts between `control_flow_to_goto.py` and
codegen. If someone renames a parameter or adds one that happens to start with "exit",
codegen behavior changes silently.

**Files:** `dgen/codegen.py:464,474`, `dgen/passes/control_flow_to_goto.py`

**Fix:** Use structural properties instead of name matching. Simplest: add a
`BlockParameter.role` attribute (`role="exit"`, `role="self"`). Then codegen checks
`param.role == "exit"` instead of string-matching on the name.

---

## Phase 3 — API hygiene and code duplication

Cross-cutting patterns that add friction but do not block correctness.

### 3a. `__constant__.to_json()` as the de-facto public API

The pattern `value.__constant__.to_json()` appears 20+ times across the codebase
(codegen, shape inference, toy_to_structured, ndbuffer_to_memory, control_flow_to_goto,
formatting). It accesses a dunder attribute to get a `Memory` object, then calls
`.to_json()` to extract the Python value. This is the most common operation on constant
values and should have a clean public API.

**Files:** scattered across `dgen/codegen.py`, `dgen/module.py`, `dgen/type.py`,
`dgen/asm/formatting.py`, `toy/passes/shape_inference.py`,
`toy/passes/toy_to_structured.py`, `dgen/passes/ndbuffer_to_memory.py`,
`dgen/passes/control_flow_to_goto.py`

**Fix:** Add a method to `Value`:

```python
def constant_value(self) -> object:
    """Return the Python value if this Value is a Constant, else raise."""
    return self.__constant__.to_json()
```

Replace all `x.__constant__.to_json()` call sites. The existing `string_value()` in
`dgen/module.py:101` is already a specialized version of this pattern.

### 3b. Duplicated `_shape()` helper

Nearly identical `_shape()` methods exist in three places:

- `toy/passes/shape_inference.py:42–48`
- `toy/passes/toy_to_structured.py:73–77`
- `dgen/passes/ndbuffer_to_memory.py:28–32`

All do `val.type.shape.__constant__.to_json()` with defensive assertions.

**Files:** the three above

**Fix:** After 3a, this becomes `val.type.shape.constant_value()`. Create a shared
`shape_of(val: Value) -> list[int]` utility (or a method on Tensor/NDBuffer types).

### 3c. PackOp unpacking duplicated

`list(val) if isinstance(val, PackOp) else [val]` appears in `dgen/codegen.py:125–126`
and `dgen/staging.py:79–82`.

**Files:** `dgen/codegen.py:125–126`, `dgen/staging.py:79–82`

**Fix:** Add an `unpack` function to a shared location (it already exists in codegen as a
local; just promote it).

---

## Phase 4 — Low priority / observational

Worth noting but not urgent.

### 4a. `resolve_stage0` break-restart loop

The loop in `staging.py:344–370` breaks the inner loop to restart the outer loop after
each resolution. Recomputes stages for all functions each iteration. In practice the
number of functions and boundaries is small, so this is not a performance concern yet.

**Files:** `dgen/staging.py:344–370`

**Fix:** Track which functions were modified, or compute a topological order of
boundaries upfront and resolve in order without restarting.

### 4b. Callback thunk deep-copies entire module

`_build_callback_thunk` (staging.py:402–477) does `deepcopy` of the entire module in the
callback path and uses `next(f for f in template.functions if ...)` for linear search.

**Files:** `dgen/staging.py:402–477`

**Fix:** Clone only the relevant function. Use a dict for function lookup by name.

### 4c. `print()` special-cased in parser

`print` gets special parsing in `toy_parser.py:151–155` with a dedicated `PrintExpr` AST
node instead of going through generic `CallExpr`.

**Files:** `toy/parser/toy_parser.py:148–163`

**Fix:** If `print` is semantically different (side-effecting, void-returning,
single-argument), the special case is justified but should be documented. Otherwise unify
with `CallExpr` and handle the difference in lowering.

### 4d. Postcondition verification pattern

`verify_postconditions` in `toy/passes/optimize.py:14–25` hand-rolls op walking to check
invariants (no double transposes, no consecutive reshapes).

**Files:** `toy/passes/optimize.py:14–25`

**Fix:** The pattern itself is good practice. Consider generalizing: a
`no_nested(OpType)` helper that checks "no op of type X has an input of type X".

### 4e. Snapshot test brittleness

Many tests compare exact IR strings. Formatting or scheduling changes break tests even
when behavior is correct. Not a code change — for new tests, prefer structural assertions
(e.g. "output contains an op of type X with operand Y") or property-based tests.
