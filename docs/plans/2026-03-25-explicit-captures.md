# Plan: Explicit Captures on Blocks

## Context

The closed-block invariant (§2.1 of `docs/block-scoping.md`) requires every
external dependency to be threaded through block arguments. This is sound and
makes blocks self-describing, but the threading cost is high:

- **Loop lowering** (`affine_to_llvm.py`) creates 3 copies of every outer ivar
  (header/body/exit) and maintains an `_enclosing_loops` stack to thread `%self`
  references for nested loops.
- **Codegen** (`codegen.py`) must trace `%self` block args back through
  predecessor branches to discover which label they refer to, requiring a
  propagation loop.
- **Every op with blocks** reinvents capture threading: `ForOp` has `init_args`,
  `CondBrOp` has `true_args`/`false_args`, `PipelineOp` threads through its body
  arg. Each op defines its own convention for how outer values reach inner blocks.

The root cause: the current invariant conflates two kinds of block inputs — values
that **vary per entry** (loop induction variables, branch-selected values) and
values that are **constant across entries** (pointers to outer allocations, `%self`
label references). Both are threaded through the same `block.args` mechanism,
requiring phi nodes in the codegen for values that never change.

## Proposal

Add a `captures` list to `Block`, alongside the existing `args` and `parameters`:

```
%alloc = affine.alloc(...)
%header = llvm.label() body<%self: Label>(%iv: Index) captures(%alloc):
    // %alloc is available directly — no phi, no threading
    // %iv varies per iteration — comes from predecessor branches
    %val = affine.load(%alloc, [%iv])
    ...
    llvm.br<%self>([%next_iv])  // only %iv is passed; %alloc is captured
```

**Captures** are outer-scope values that a block references directly. They are:
- Declared on the block (explicit, verifiable)
- Available inside the block as the original value (no `BlockArgument` copy)
- Leaves in `walk_ops` (it stops at capture boundaries, same as block args)
- Not passed through branches (no phi nodes needed)

This gives blocks three kinds of inputs:
- **`parameters`**: compile-time values bound at block construction (`%self`)
- **`args`**: runtime values that vary per predecessor (loop ivars) — get phi nodes
- **`captures`**: runtime values from the enclosing scope that don't vary — no phi

## Design doc updates

### §2.1 — Closed Blocks: extend the invariant

Current: "every op reachable from B.result has dependencies entirely within B.args
(or is ambient)."

New: "every op reachable from B.result has dependencies entirely within
`B.parameters ∪ B.args ∪ B.captures` (or is ambient)."

The invariant remains structural and locally checkable. The only change is the
addition of `captures` to the valid-reference set.

### §2.2 — walk_ops: add capture boundary

Current: "walk_ops follows operands, parameter values, result type, and block
argument types."

New: walk_ops treats captured values as leaves — it does not follow edges from
ops inside a block to their captured values. This preserves the locality property:
`walk_ops(B.result)` produces only ops belonging to B (plus ambient nodes), never
ops from enclosing blocks reached through captures.

Implementation: when walk_ops encounters a value, check if it is in the current
block's `captures` set. If so, treat it as a leaf (like a `BlockArgument`). This
requires walk_ops to accept the block context, or captured values to be wrapped
in a leaf type (see implementation options below).

### §2.3 — Motivating example: update

The "after" example currently threads `%src` as a block argument:

```
%src = llvm.alloca<6>()
%5 = affine.for<0, 4>(%src) (%i: Index, %src: llvm.Ptr):
    %val = affine.load(%src, [%i])
```

With captures:

```
%src = llvm.alloca<6>()
%5 = affine.for<0, 4>() (%i: Index) captures(%src):
    %val = affine.load(%src, [%i])  // %src is the original value
```

### §3.2 — Label `%self`: simplify

`%self` remains a block parameter (compile-time, bound to the label). But values
that were previously threaded alongside `%self` through block args — outer ivars,
enclosing `%self` references, tensor pointers — become captures instead. The
back-branch passes only values that actually change (the incremented loop var):

```
%header = llvm.label() body<%self: Label>(%iv: Index) captures(%alloc, %outer_self):
    ...
    llvm.br<%self>([%next_iv])
```

No `_enclosing_loops` stack needed in the lowering — the block simply captures
whatever outer values it needs.

### §3.3 — Explicit Capture: rewrite

Currently describes threading through args as the only mechanism. Rewrite to
describe the three-part block interface (`parameters`, `args`, `captures`) and
the rules for each:

- **Parameters**: compile-time, bound by the op that owns the block, not passed
  by callers.
- **Args**: runtime, passed by every predecessor branch, generate phi nodes.
- **Captures**: runtime, reference outer-scope values directly, no phi nodes.

### §3.4 — Migrating Existing Ops: simplify

`affine.for` no longer needs `init_args` for captured values — only for values
that actually vary (if any). `builtin.if` doesn't need `then_args`/`else_args`
for captured values. The migration becomes simpler: identify which block inputs
actually vary per entry vs which are constant captures.

## Implementation

### Step 1: `Block.captures` field

Add `captures: list[dgen.Value]` to `Block.__init__`. Unlike `args` and
`parameters` (which are `BlockArgument` instances — fresh leaves), captures are
references to the original outer-scope values.

```python
class Block:
    result: dgen.Value
    args: list[BlockArgument]
    parameters: list[BlockArgument]
    captures: list[dgen.Value]
```

### Step 2: `walk_ops` — stop at capture boundaries

`walk_ops` needs to know which values are captures so it can treat them as leaves.
Two options:

**Option A: Pass a stop-set.**

```python
def walk_ops(root: dgen.Value, stop: set[dgen.Value] | None = None) -> list[dgen.Op]:
```

Callers (e.g. `Block.ops`) pass `set(block.captures)` as the stop set. The visit
function checks `if value in stop: return` before following edges. Simple, no new
types, but requires threading the stop set through every call.

**Option B: Capture wrapper type.**

Introduce `CapturedValue(Value)` that wraps a captured value. walk_ops treats it
as a leaf (not an Op, so it stops). Ops inside the block reference the
`CapturedValue`, not the original. This is structurally identical to
`BlockArgument` but semantically different (no phi). The wrapper carries the
original value for the codegen to read.

**Recommendation: Option A.** It's simpler, doesn't introduce a new Value
subclass, and Block.ops already calls walk_ops — it just needs to pass the
stop set. Option B makes captures more first-class in the type system but adds
a layer of indirection that complicates `replace_uses` and `inline_block`.

### Step 3: `verify_closed_blocks` — add captures to valid set

```python
valid: set[dgen.Value] = set(block.parameters) | set(block.args) | set(block.captures)
```

One line change.

### Step 4: ASM syntax — `captures(...)`

Extend block formatting and parsing:

```
body<%self: Label>(%iv: Index) captures(%alloc, %ptr):
```

- Formatter: after the `(args)` section, emit `captures(...)` if non-empty.
- Parser: after reading `(args)`, optionally read `captures(...)`.

### Step 5: `inline_block` — captures are already valid

```python
def inline_block(block: Block, args: list[Value]) -> Value:
    # Substitute args only — captures reference outer values directly.
    rewriter = Rewriter(block)
    for old_arg, new_val in zip(block.args, args):
        rewriter.replace_uses(old_arg, new_val)
    return block.result
```

No change needed — captures aren't `BlockArgument` instances, so `replace_uses`
doesn't touch them. They're already valid in the caller's scope.

### Step 6: Simplify `affine_to_llvm.py` lowering

- Remove `_enclosing_loops` stack.
- In `_lower_for`: outer ivars and enclosing `%self` values become captures on
  header/body/exit blocks instead of threaded block args. Only the loop induction
  variable is a block arg.
- Back-branch passes only `[%next_iv]`, not `[%next_iv, %outer_iv1, ..., %self0, %self1, ...]`.
- `cond_br` `true_args`/`false_args` shrink to just `[%header_iv]` for the body
  and `[]` for the exit.

### Step 7: Simplify `codegen.py`

- `label_of` propagation loop disappears. `%self` is a block parameter (directly
  in `label_of` seed). Enclosing `%self` values are captures — the codegen reads
  them directly from the block's captures list.
- `resolve_target`: branch target is either a `LabelOp` (direct) or a block
  parameter (in `label_of`). No body-arg tracing needed.
- Phi emission: emit phis only for `block.args`. Captures and parameters are
  skipped structurally (they're not in `args`), not by type-checking.

### Step 8: Regenerate snapshots

All LLVM IR snapshots change — blocks have fewer args, branches pass fewer
values, captures appear in the ASM syntax.

## What this does NOT change

- **`%self` block parameters**: Still needed to break DAG cycles. The header's
  `%self` is a block parameter bound to the label. This is orthogonal to captures.
- **`verify_dag`**: Unchanged. Cycles are still broken by `%self` (a
  `BlockArgument` leaf). Captures don't create cycles because walk_ops stops at
  them.
- **The DAG property**: Maintained. Captures are leaves in walk_ops, same as
  block args.
- **Staging**: Parameters are still the stage boundary. Captures are runtime
  values, like args.

## Migration path

This can be done incrementally:

1. Add `Block.captures` field + verify + walk_ops stop-set (core infra).
2. Update ASM formatter/parser for `captures(...)` syntax.
3. Migrate `affine_to_llvm.py` to use captures instead of threaded args.
4. Migrate `builtin_to_llvm.py` if/else lowering.
5. Update design doc sections.

Each step is independently testable. Steps 1–2 are backwards-compatible (empty
captures list = current behavior).
