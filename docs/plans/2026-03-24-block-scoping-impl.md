# Block Scoping Implementation Plan

> **Design reference:** `docs/block-scoping.md`

**Goal:** Enforce the closed-block invariant, add `func.recursive`, introduce block parameters
(Option C `%self` mechanism), migrate ops with implicit capture, and simplify `walk_ops` to a
guard-free DAG walk.

**Current state (as of 2026-03-24):**
- `_stored_ops` removed; `Block` already uses `walk_ops` always ✓
- `PhiOp` removed from `llvm.dgen`; branch ops already have `args`/`true_args`/`false_args` ✓
- `verify.py` has `verify_closed_blocks()` but with hard-coded exceptions for `FunctionOp` and
  `LabelOp` — these exceptions are the structural problem to fix ✗
- `walk_ops` has the `FunctionOp` guard + `id()`-based visited set ✗
- No `func.recursive` op; `builtin.function` still produces `Nil` ✗
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
| 2 | `func.recursive` op | `builtin.dgen`, `builtin.pyi` |
| 3 | `builtin.if` per-branch args | `builtin.dgen`, `builtin.pyi`, `builtin_to_llvm.py` |
| 4 | `affine.for` explicit capture | `affine.dgen`, `affine.pyi`, `toy_to_affine.py` |
| 5 | `affine_to_llvm.py` closed blocks | `affine_to_llvm.py` |
| 6 | `walk_ops` + verifier cleanup | `graph.py`, `verify.py` |

Each phase ends with a passing full test suite (`pytest . -q`).

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

If a block has no params, the syntax is unchanged (params list elided). The first block name is
always printed when params are present (overriding the current carveout that elides it).

### 1.1 IR model: `Block.params`

**File:** `dgen/block.py`

Add `params: list[BlockArgument]` to `Block.__init__` (default `[]`). Params are structurally
identical to args — both are `BlockArgument` instances — but occupy a distinct list. The block's
full scope set is `block.params + block.args`.

Update `Block.__init__`:
```python
def __init__(
    self,
    *,
    result: dgen.Value,
    args: list[BlockArgument] | None = None,
    params: list[BlockArgument] | None = None,
) -> None:
    self.result = result
    self.args = args if args is not None else []
    self.params = params if params is not None else []
```

No change to `ops` property — params are leaves in `walk_ops`, not roots.

**File:** `dgen/graph.py`

In `walk_ops`, extend the block traversal to also visit param types:

```python
for _, block in value.blocks:
    for param in block.params: visit(param.type)
    for arg in block.args: visit(arg.type)
```

**File:** `dgen/verify.py`

In `_verify_block`, add params to the valid set alongside args:

```python
valid: set[int] = {id(arg) for arg in block.params} | {id(arg) for arg in block.args}
```

### 1.2 ASM formatter

**File:** `dgen/asm/formatting.py`

Locate the block formatting logic. When printing a block header:
- If `block.params` is non-empty: print `block_name<%param: T, ...>(%arg: T, ...):`
- The block name must be printed whenever params are non-empty (even for the first block).
- Format: `<%param1: T1, %param2: T2>` immediately after the block name, before `(`.

Current carveout: the first block's name is omitted. Change: if the first block has params,
its name is NOT omitted.

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
- Block with params, no args: `body<%self: Function<Index>>():`
- Block with params and args: `body<%self: llvm.Label>(%i: Index):`
- First block with params (name must be printed): verify formatter output
- Parser produces correct `block.params` list
- `walk_ops` treats params as leaves (does not descend into their values)
- `verify_closed_blocks` accepts param references within the same block

### 1.5 Commit

```bash
jj commit -m "Block: add params list, formatter/parser support for block parameter syntax"
```

---

## Phase 2: `func.recursive` Op

Add `func.recursive` to the builtin dialect. It produces a `Function<T>` value (the function's
own identity) and introduces `%self: Function<T>` as the first block parameter.

### 2.1 Update `builtin.dgen`

**File:** `dgen/dialects/builtin.dgen`

Add after the existing `function` op:
```
op recursive<result: Type>() -> Function<result>:
    block body
    has trait HasSingleBlock
```

The `result` parameter specifies the function's return type; the produced value is
`Function<result>`. The body block conventionally uses `%self: Function<result>` as its first
block parameter (populated by the lowering pass at call time).

### 2.2 Regenerate stubs

```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.pyi
```

### 2.3 Tests

**File:** `test/test_recursive.py` (new)

Write tests covering:
- Constructing a `func.recursive` op with a body block having a `%self` param
- ASM round-trip of the factorial example from `docs/block-scoping.md`
- The op's type is `Function<Index>`, not `Nil`
- `walk_ops` on the body's result: `%self` is a leaf (block param), no cycle

Example ASM to round-trip (from `docs/block-scoping.md` Section 3.1):
```
%factorial : Function<Index> = recursive<Index>() body<%self: Function<Index>>(%n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond)(%one: Index)(%n: Index, %self: Function<Index>):
        %base : Index = 1
    else(%n: Index, %self: Function<Index>):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %r : Index = call<%self>([%n1])
        %res : Index = llvm.mul(%n, %r)
```

Note: this test depends on Phase 3 (`builtin.if` with per-branch args). Write the `walk_ops`
and type tests first; defer the full factorial round-trip to after Phase 3.

### 2.4 Commit

```bash
jj commit -m "builtin: add func.recursive op producing Function<T>"
```

---

## Phase 3: `builtin.if` Per-Branch Args

Currently `if`'s then/else blocks take no arguments, enabling implicit capture. Under the
closed-block invariant, outer values used in branches must be threaded explicitly.

### 3.1 Update `builtin.dgen`

**File:** `dgen/dialects/builtin.dgen`

Change the `if` op to accept per-branch operand lists:
```
op if(cond: Index, then_args: List, else_args: List) -> Type:
    block then_body
    block else_body
```

`then_args` is a `List` of values passed as block arguments to `then_body`; similarly
`else_args` for `else_body`. The block argument types are derived from the passed values' types.

Callers that don't thread any values use empty lists: `if(%cond, [], [])`.

### 3.2 Regenerate stubs

```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.pyi
```

### 3.3 Update existing call sites

**File:** `dgen/passes/builtin_to_llvm.py`

Locate `IfOp` construction (in `_lower_if`). Thread any values from the enclosing scope that
are used in branch bodies through `then_args`/`else_args`, and declare them as block args on
the respective branch blocks. The lowering pass is responsible for identifying captured values.

The pattern:
```python
# Values used in then_body captured from enclosing scope:
then_captures = [...]   # list of Value objects
else_captures = [...]

then_args_pack = PackOp(values=then_captures, ...)
else_args_pack = PackOp(values=else_captures, ...)

# Branch blocks get corresponding BlockArguments:
then_body = Block(
    result=...,
    args=[BlockArgument(name=v.name, type=v.type) for v in then_captures],
)
```

**File:** `toy/passes/toy_to_affine.py`

Locate `IfOp` construction. Apply same threading pattern.

### 3.4 ASM formatter/parser

Update the `if` op formatter to emit per-branch arg lists and corresponding block arg lists:
```
%r : T = if(%cond)(%a: TA, %b: TB) (%a: TA, %b: TB):
    ...
else (%a: TA, %b: TB):
    ...
```

The first parenthesized group after the `if(cond)` is `then_args`; the `then_body` block header
follows; the `else` keyword precedes `else_body`'s block args.

This may require formatter/parser work to distinguish `then_args` (operands to the `if` op)
from the block argument lists on each branch. The cleanest approach is: after `if(%cond)`, emit
`(then_arg_list)` for the `then_body` block's args, and `else (else_arg_list)` for `else_body`.
The op's `then_args`/`else_args` fields control what values are passed; the block headers show
what they're bound as.

### 3.5 Tests

Update all existing `if`-containing IR tests to use the new syntax. Add a test that:
- Verifies `verify_closed_blocks` passes for an `if` op with explicit capture
- Verifies it fails for an `if` op with an outer-scope reference not in `then_args`

### 3.6 Commit

```bash
jj commit -m "builtin.if: per-branch arg threading for explicit capture"
```

---

## Phase 4: `affine.for` Explicit Capture

Currently `affine.for` takes no operands; the body block can freely reference outer values.
Under the closed-block invariant, external values must be passed as additional block args.

### 4.1 Update `affine.dgen`

**File:** `toy/dialects/affine.dgen`

Change the `for` op to accept an initial-args list:
```
op for<lo: Index, hi: Index>(init_args: List) -> Nil:
    block body
    has trait HasSingleBlock
```

The body block arguments are `(%i: Index, %captured_val: T, ...)` — induction variable first,
then one arg per value in `init_args`. The loop lowering pass is responsible for constructing
this.

### 4.2 Regenerate stubs

```bash
python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.pyi
```

### 4.3 Update `toy_to_affine.py`

**File:** `toy/passes/toy_to_affine.py`

Locate loop body construction. Identify all values from the enclosing scope used in the body.
Pass them in `init_args` and declare them as block args on the body block (after `%i`).

In the design, `affine.for<0, 4>(%src) (%i: Index, %src: llvm.Ptr):` means:
- `%src` is passed in `init_args`
- Body block args: `%i` (induction variable, first), `%src` (from init_args)

### 4.4 Update `affine_to_llvm.py`

**File:** `toy/passes/affine_to_llvm.py`

The lowering of `affine.for` to `llvm.label` blocks must thread `init_args` as block args
on the header label (alongside the loop variable).

### 4.5 Snapshots

```bash
pytest toy/test/test_toy_to_affine.py --snapshot-update -q
pytest toy/test/test_affine_to_llvm.py --snapshot-update -q
```

### 4.6 Commit

```bash
jj commit -m "affine.for: explicit capture via init_args operand list"
```

---

## Phase 5: `affine_to_llvm.py` Closed Blocks

With `affine.for` explicit capture in place, rewrite `affine_to_llvm.py` so that every
generated `llvm.label` body is closed: all outer-scope values are block args, no implicit
references.

**File:** `toy/passes/affine_to_llvm.py`

### 5.1 Rewrite `_lower_for`

The loop lowering creates header, body, exit labels. Each label body references only its own
block args and local ops. The header label takes `(%i: Index, %captured_val: T, ...)` as block
args (matching `affine.for`'s block arg order). The body label similarly receives any values it
needs.

The loop back-edge (`llvm.br` to header) passes `[%i_next, %captured_val, ...]` explicitly.
Entry branch passes initial values `[%lo, %captured_val, ...]`.

Use the `%self` block parameter convention for the loop header:
```
%header : llvm.Label = llvm.label() body<%self: llvm.Label>(%i: Index, %src: llvm.Ptr):
    ...
    %_ : Nil = llvm.br(%self, [%i_next, %src])   // back-edge via %self
```

Lowering populates `%self` automatically — callers pass only user-visible args.

### 5.2 Call `verify_closed_blocks` after the pass

In the pass entry point, call `verify_closed_blocks(module)` after lowering to assert
correctness. This catches any missed captures immediately rather than silently.

### 5.3 Update snapshots

```bash
pytest toy/test/test_affine_to_llvm.py --snapshot-update -q
pytest toy/test/test_end_to_end.py -q   # JIT output must be unchanged
```

### 5.4 Commit

```bash
jj commit -m "affine_to_llvm: closed label blocks with explicit capture and %self"
```

---

## Phase 6: `walk_ops` and Verifier Cleanup

With the closed-block invariant enforced throughout the pipeline, the structural special cases
in `walk_ops` and `verify.py` can be removed.

### 6.1 Remove `FunctionOp` guard from `walk_ops`

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

Also change the `visited` set from `set[int]` (using `id(value)`) to `set` (using `value`
directly, relying on `eq=False` dataclass identity):
```python
visited: set[dgen.Value] = set()
...
if value in visited:
    return
visited.add(value)
```

The `visit` function signature changes from `object` to `dgen.Value | list[dgen.Value]` (or
handle list dispatch at call sites).

### 6.2 Remove hard-coded exceptions from verifier

**File:** `dgen/verify.py`

Remove `_is_cross_block_permitted` and its two callers. `FunctionOp` and `LabelOp` are ambient
ops — they have no block-argument dependencies and are always valid to reference from any block.
The verifier should detect ambience structurally rather than by type check.

Update `_check_in_scope`: an op is permitted from any block if it is ambient (no transitive
block-argument dependencies). The simplest correct rule: an op is in-scope if it is in the
valid set (local ops + block args + block params). Ambient ops are already in the valid set
because they're reachable from `block.ops` (they're discovered by `walk_ops`).

Wait — are ambient ops actually in `block.ops`? Yes: `block.ops = walk_ops(block.result)`, and
`walk_ops` follows parameter edges into ambient ops. So `FunctionOp` (when referenced as a
`call` parameter) IS returned in the caller block's `walk_ops` result, hence it IS in `valid`.
The `_is_cross_block_permitted` exception is only needed because the current guard in `walk_ops`
SKIPS visiting `FunctionOp` — so it's NOT in `valid`. Once the guard is removed (Step 6.1),
the exception in the verifier is also unnecessary.

Remove `_is_cross_block_permitted` only AFTER Step 6.1, and verify tests pass.

### 6.3 Full test suite

```bash
pytest . -q
ruff format
ruff check --fix
ty check
```

All 110+ tests must pass. JIT output must be unchanged.

### 6.4 Commit

```bash
jj commit -m "walk_ops: remove FunctionOp guard, use value identity in visited set"
jj commit -m "verify: remove FunctionOp/LabelOp cross-block exceptions"
```

---

## Design Notes

### Block parameters vs block args

Block parameters (`block.params`) are structurally identical to block args (`block.args`) in
the IR — both are `BlockArgument` instances in scope. They differ in binding protocol:
- **Block args**: passed by callers at every branch/call site.
- **Block params**: bound once by the op's lowering pass (e.g., `%self` is the op's own
  identity); callers never pass them.

This distinction matters for codegen and lowering, not for the use-def graph. In the use-def
graph, both are leaves.

### `%self` auto-supply convention

For `llvm.label` bodies, `%self` (if present as a block param) is auto-supplied by lowering:
when emitting `llvm.br(%loop, [%i_next])`, the lowering pass inserts `%loop` as the `%self`
argument to the target block. Callers don't name it. This mirrors how `func.recursive` lowering
populates `%self` for recursive calls.

### `call` op remains unchanged

Under Option C, `call<callee: Function>(args: List)` is unchanged. Recursive calls use
`call<%self>([...])` where `%self` is a block arg (threaded from the block param). External
calls use `call<%factorial>([...])` where `%factorial` is an ambient op. The `FunctionOp` guard
in `walk_ops` is removed because ambient ops are now correctly included in the caller block's
op set (they were being suppressed before).

### Ordering constraint

Phase 6 depends on ALL prior phases. Removing the `walk_ops` guard while implicit capture still
exists would cause `walk_ops` to incorrectly surface ops across block boundaries. Run
`verify_closed_blocks` as a post-pass assertion throughout Phases 3–5 to catch regressions
early.

### `func.recursive` codegen

Codegen for `func.recursive` is out of scope for this plan. The op is introduced at the IR
level with round-trip support. Actual JIT compilation of recursive functions (including the
`%self` thunk mechanism) is a follow-on task that requires extending `dgen/codegen.py`.

---

## File Summary

| File | Change |
|------|--------|
| `dgen/block.py` | Add `params: list[BlockArgument]` to `Block` |
| `dgen/graph.py` | Add `block.params` traversal; remove `FunctionOp` guard; use `value` in visited set |
| `dgen/verify.py` | Add params to valid set; remove `_is_cross_block_permitted` exceptions |
| `dgen/asm/formatting.py` | Emit `block_name<%param: T>` syntax for blocks with params |
| `dgen/asm/parser.py` | Parse `<%param: T, ...>` in block headers |
| `dgen/dialects/builtin.dgen` | Add `recursive` op |
| `dgen/dialects/builtin.pyi` | Regenerate |
| `toy/dialects/affine.dgen` | Add `init_args: List` to `for` op |
| `toy/dialects/affine.pyi` | Regenerate |
| `dgen/passes/builtin_to_llvm.py` | Thread captures through `if` branch arg lists |
| `toy/passes/toy_to_affine.py` | Thread captures through `if` and `for` |
| `toy/passes/affine_to_llvm.py` | Produce closed label blocks with `%self` block param |
| `test/test_block_params.py` | New: block param round-trip and walk_ops tests |
| `test/test_recursive.py` | New: `func.recursive` construction and type tests |
| `toy/test/__snapshots__/` | Update affected snapshots |
