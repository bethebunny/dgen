# Block Scoping Implementation Plan

> **Design reference:** `docs/block-scoping.md`

**Goal:** Enforce the closed-block invariant, introduce block parameters (Option C `%self`
mechanism for `llvm.label`), migrate ops with implicit capture, and simplify `walk_ops` to a
guard-free DAG walk.

**Current state (as of 2026-03-24):**
- `_stored_ops` removed; `Block` already uses `walk_ops` always ✓
- `PhiOp` removed from `llvm.dgen`; branch ops already have `args`/`true_args`/`false_args` ✓
- `verify.py` has `verify_closed_blocks()` but with hard-coded exceptions for `FunctionOp` and
  `LabelOp` — these exceptions are the structural problem to fix ✗
- `walk_ops` has the `FunctionOp` guard + `id()`-based visited set ✗
- No block parameter concept (`body<%self: T>` syntax) ✗
- `builtin.if` has no per-branch arg lists ✗
- `affine.for` has no explicit operand threading ✗
- Lowering passes (`toy_to_affine.py`, `affine_to_llvm.py`, `builtin_to_llvm.py`) still produce
  implicit capture ✗

**Tech stack:** Python, pytest, dgen IR framework, llvmlite

---

## Overview: Phases

| Phase | Goal | Key files |
|-------|------|-----------|
| 1 | Block parameters in IR, formatter, parser | `block.py`, `asm/formatting.py`, `asm/parser.py` |
| 2 | `builtin.if` per-branch args | `builtin.dgen`, `builtin.pyi`, `builtin_to_llvm.py`, `toy_to_affine.py` |
| 3 | `affine.for` explicit capture | `affine.dgen`, `affine.pyi`, `toy_to_affine.py` |
| 4 | `affine_to_llvm.py` closed blocks with `%self` | `affine_to_llvm.py` |
| 5 | `walk_ops` + verifier cleanup | `graph.py`, `verify.py` |

Each phase ends with a passing full test suite (`pytest . -q`).

**Verifier:** `verify_closed_blocks()` already exists in `dgen/verify.py` (with `FunctionOp` /
`LabelOp` exceptions). Use it as a post-pass assertion from Phase 2 onward. The exceptions are
removed in Phase 5 once implicit capture is fully eliminated.

---

## Phase 1: Block Parameters

Introduce **block parameters**: a new binding position on a `Block`, distinct from block
arguments. Block parameters are in scope inside the block and are NOT passed by callers — their
value is populated by each op's lowering pass.

ASM syntax:
```
op_name() block_name<%param: Type>(%arg1: T1, %arg2: T2):
    ...
```

If a block has no parameters, the syntax is unchanged (parameter list elided). The block name
is always printed when parameters are present, even for the first block (overriding the current
carveout that elides it).

### 1.1 IR model: `Block.parameters`

**File:** `dgen/block.py`

Add `parameters: list[BlockArgument]` to `Block.__init__` (default `[]`). Parameters are
structurally identical to args — both are `BlockArgument` instances — but occupy a distinct
list. The block's full scope set is `block.parameters + block.args`.

Update `Block.__init__`:
```python
def __init__(
    self,
    *,
    result: dgen.Value,
    args: list[BlockArgument] | None = None,
    parameters: list[BlockArgument] | None = None,
) -> None:
    self.result = result
    self.args = args if args is not None else []
    self.parameters = parameters if parameters is not None else []
```

No change to `ops` property — block parameters are leaves in `walk_ops`, not roots.

**File:** `dgen/graph.py`

In `walk_ops`, extend the block traversal to also visit block parameter types:

```python
for _, block in value.blocks:
    for param in block.parameters: visit(param.type)
    for arg in block.args: visit(arg.type)
```

**File:** `dgen/verify.py`

In `_verify_block`, add block parameters to the valid set alongside args:

```python
valid: set[int] = (
    {id(p) for p in block.parameters} | {id(a) for a in block.args}
)
```

### 1.2 ASM formatter

**File:** `dgen/asm/formatting.py`

When printing a block header:
- If `block.parameters` is non-empty: print `block_name<%param: T, ...>(%arg: T, ...):`
- The block name must be printed when parameters are non-empty, even for the first block.
- Format: `<%param1: T1, %param2: T2>` immediately after the block name, before `(`.

### 1.3 ASM parser

**File:** `dgen/asm/parser.py`

After parsing a block name, check for `<`. If present, parse a comma-separated list of
`%name: Type` declarations as block parameters (same grammar as block args, different list).

Grammar addition:
```
block_header ::= block_name ('<' param_list '>')? '(' arg_list ')' ':'
param_list   ::= '%' name ':' type (',' '%' name ':' type)*
```

### 1.4 Tests

**File:** `test/test_block_params.py` (new test file)

Write ASM round-trip tests covering:
- Block with parameters, no args: `body<%self: llvm.Label>():`
- Block with parameters and args: `body<%self: llvm.Label>(%i: Index):`
- First block with parameters: name must appear in formatter output
- Parser produces correct `block.parameters` list
- `walk_ops` treats block parameters as leaves (does not descend into their values)
- `verify_closed_blocks` accepts block parameter references within the same block

### 1.5 Commit

```bash
jj commit -m "Block: add parameters list, formatter/parser support for block parameter syntax"
```

---

## Phase 2: `builtin.if` Per-Branch Args

Currently `if`'s then/else blocks take no arguments, enabling implicit capture. Under the
closed-block invariant, outer values used in branches must be threaded explicitly.

After this phase, run `verify_closed_blocks(module)` as a post-pass assertion in
`builtin_to_llvm.py` and `toy_to_affine.py` to confirm the invariant holds.

### 2.1 Update `builtin.dgen`

**File:** `dgen/dialects/builtin.dgen`

Change the `if` op to accept per-branch operand lists:
```
op if(cond: Index, then_args: List, else_args: List) -> Type:
    block then_body
    block else_body
```

`then_args` is a `List` of values passed as block arguments to `then_body`; similarly
`else_args` for `else_body`. Callers that thread no values use empty lists: `if(%cond, [], [])`.

### 2.2 Regenerate stubs

```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.pyi
```

### 2.3 Update existing call sites

**File:** `dgen/passes/builtin_to_llvm.py`

Locate `IfOp` construction (in `_lower_if`). Thread values from the enclosing scope that are
used in branch bodies through `then_args`/`else_args`, and declare them as block args on the
respective branch blocks:

```python
then_captures = [...]   # Value objects used in then_body from outer scope
else_captures = [...]

then_body = Block(
    result=...,
    args=[BlockArgument(name=v.name, type=v.type) for v in then_captures],
)
# then_args PackOp carries then_captures; else_args carries else_captures
```

**File:** `toy/passes/toy_to_affine.py`

Locate `IfOp` construction. Apply the same threading pattern.

### 2.4 ASM formatter/parser

The `if` op formatter emits per-branch arg lists adjacent to the block headers. The current
`if` ASM prints the `then_body` block args in `(...)` after `if(cond)`. With `then_args` and
`else_args` as op operands, the formatter must distinguish the operand list from the block arg
list. Recommended layout:

```
%r : T = if(%cond, [%a, %b], [%a]) (%a: TA, %b: TB):
    ...
else (%a: TA):
    ...
```

The op's operand lists (`[%a, %b]`, `[%a]`) are inlined as operand syntax; each branch block's
arg list `(%a: TA, %b: TB)` / `(%a: TA)` follows immediately. Update the parser to match.

### 2.5 Tests

Update all existing `if`-containing IR tests to use the new syntax with explicit arg lists.
Add tests that:
- Verify `verify_closed_blocks` passes for an `if` op with explicit capture
- Verify it fails (under the current exceptions still in place) for an `if` with an
  outer-scope reference not in `then_args`

Run: `pytest . -q`

### 2.6 Commit

```bash
jj commit -m "builtin.if: per-branch arg threading for explicit capture"
```

---

## Phase 3: `affine.for` Explicit Capture

Currently `affine.for` takes no operands; the body block freely references outer values.
Under the closed-block invariant, external values must be passed as additional block args.

### 3.1 Update `affine.dgen`

**File:** `toy/dialects/affine.dgen`

Change the `for` op to accept an initial-args list:
```
op for<lo: Index, hi: Index>(init_args: List) -> Nil:
    block body
    has trait HasSingleBlock
```

Body block arguments are `(%i: Index, %captured_val: T, ...)` — induction variable first,
then one arg per value in `init_args`.

### 3.2 Regenerate stubs

```bash
python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.pyi
```

### 3.3 Update `toy_to_affine.py`

**File:** `toy/passes/toy_to_affine.py`

Identify all values from the enclosing scope used in the loop body. Pass them in `init_args`
and declare them as block args on the body block (after `%i`):

```
affine.for<0, 4>([%src]) (%i: Index, %src: llvm.Ptr):
    affine.load(%src, [%i])
```

After the pass, assert `verify_closed_blocks(module)`.

### 3.4 Update snapshot tests

```bash
pytest toy/test/test_toy_to_affine.py --snapshot-update -q
```

Review: `init_args` present, body block args include captured values.

### 3.5 Commit

```bash
jj commit -m "affine.for: explicit capture via init_args operand list"
```

---

## Phase 4: `affine_to_llvm.py` Closed Blocks with `%self`

Rewrite `affine_to_llvm.py` so every generated `llvm.label` body is closed. This is where the
`%self` block parameter is first used in production: the loop header block receives `%self:
llvm.Label` as its block parameter, and the back-edge branch passes user-visible args only.

**File:** `toy/passes/affine_to_llvm.py`

### 4.1 Rewrite `_lower_for`

The loop lowering creates header, body, exit labels. Each label body references only its own
block args, block parameters, and local ops — no implicit outer-scope references.

The header label uses `%self` as its block parameter for the back-edge:
```
%header : llvm.Label = llvm.label() body<%self: llvm.Label>(%i: Index, %src: llvm.Ptr):
    ...
    %_ : Nil = llvm.br(%self, [%i_next, %src])   // back-edge via %self; %self auto-supplied
```

Entry branch passes only user-visible args (no `%self`):
```
%_ : Nil = llvm.br(%header, [%lo, %src])
```

Lowering populates `%self` when emitting branches to a label that has a `%self` block
parameter. Concretely: when the lowering sees `llvm.br(%header, args)`, it prepends `%header`
to the actual argument list for the `%self` slot. `%self` never appears explicitly in
branch-site arg lists in the IR — it is resolved by the lowering pass.

### 4.2 Assert verifier

After the pass completes, call `verify_closed_blocks(module)`. This will catch any remaining
implicit captures immediately.

### 4.3 Update snapshots

```bash
pytest toy/test/test_affine_to_llvm.py --snapshot-update -q
pytest toy/test/test_end_to_end.py -q   # JIT output must be unchanged
```

### 4.4 Commit

```bash
jj commit -m "affine_to_llvm: closed label blocks with explicit capture and %self block param"
```

---

## Phase 5: `walk_ops` and Verifier Cleanup

With the closed-block invariant enforced throughout the pipeline, the structural special cases
in `walk_ops` and `verify.py` can be removed.

### 5.1 Remove `FunctionOp` guard from `walk_ops`

**File:** `dgen/graph.py`

Before:
```python
for _, param in value.parameters:
    from dgen.dialects.builtin import FunctionOp
    if not isinstance(param, FunctionOp):
        visit(param)
```

After:
```python
for _, param in value.parameters:
    visit(param)
```

Also switch the `visited` set from `set[int]` (using `id(value)`) to `set[dgen.Value]` (using
value identity directly, relying on `eq=False` dataclass semantics):

```python
visited: set[dgen.Value] = set()
```

Remove the `vid = id(value)` / `if vid in visited` / `visited.add(vid)` pattern throughout.

### 5.2 Remove hard-coded exceptions from `verify.py`

**File:** `dgen/verify.py`

The `_is_cross_block_permitted` exception exists because the `walk_ops` guard suppresses
`FunctionOp` visits, so `FunctionOp` is NOT in `block.ops` → NOT in `valid`. Once Step 5.1 is
done, `FunctionOp` IS visited by `walk_ops` (it's an ambient op reachable via parameter edges),
so it IS in `valid`. The exception is then unnecessary.

Similarly, `LabelOp` is an ambient op (its body has no external block-arg dependencies) and
will be in `valid` via `walk_ops`. Remove `_is_cross_block_permitted` and its call sites.

Remove only AFTER Step 5.1 passes the full test suite.

### 5.3 Full test suite

```bash
pytest . -q
ruff format
ruff check --fix
ty check
```

All tests must pass. JIT output must be unchanged.

### 5.4 Commit

```bash
jj commit -m "walk_ops: remove FunctionOp guard, use value identity in visited set"
jj commit -m "verify: remove FunctionOp/LabelOp cross-block exceptions"
```

---

## Design Notes

### Block parameters vs block args

Block parameters (`block.parameters`) are structurally identical to block args (`block.args`)
— both are `BlockArgument` instances in scope in the same block. They differ in binding:
- **Block args**: passed by callers at every call/branch site.
- **Block parameters**: bound once by the op's lowering pass; callers never pass them.

In the use-def graph, both are leaves. The distinction matters only to lowering and codegen.

### `%self` auto-supply in lowering

When a lowering pass emits `llvm.br(%header, user_args)` and `%header`'s body block has a
`%self: llvm.Label` block parameter, the pass inserts `%header` into the actual argument list
for the `%self` slot before emitting the branch. Callers in the IR never name `%self`
explicitly — it is resolved structurally by the lowering pass.

### Ordering constraint

Phase 5 depends on ALL prior phases. Removing the `walk_ops` guard while implicit capture still
exists would cause `walk_ops` to surface ops across block boundaries incorrectly. Using
`verify_closed_blocks` as a post-pass assertion from Phase 2 onward catches regressions early,
before they become silent correctness bugs in Phase 5.

---

## File Summary

| File | Change |
|------|--------|
| `dgen/block.py` | Add `parameters: list[BlockArgument]` to `Block` |
| `dgen/graph.py` | Add `block.parameters` traversal; remove `FunctionOp` guard; use `value` in visited set |
| `dgen/verify.py` | Add block parameters to valid set; remove `_is_cross_block_permitted` exceptions |
| `dgen/asm/formatting.py` | Emit `block_name<%param: T>` syntax for blocks with parameters |
| `dgen/asm/parser.py` | Parse `<%param: T, ...>` in block headers |
| `dgen/dialects/builtin.dgen` | Add `then_args: List, else_args: List` to `if` op |
| `dgen/dialects/builtin.pyi` | Regenerate |
| `toy/dialects/affine.dgen` | Add `init_args: List` to `for` op |
| `toy/dialects/affine.pyi` | Regenerate |
| `dgen/passes/builtin_to_llvm.py` | Thread captures through `if` branch arg lists |
| `toy/passes/toy_to_affine.py` | Thread captures through `if` and `for` |
| `toy/passes/affine_to_llvm.py` | Produce closed label blocks with `%self` block parameter |
| `test/test_block_params.py` | New: block parameter round-trip and `walk_ops` tests |
| `toy/test/__snapshots__/` | Update affected snapshots |
