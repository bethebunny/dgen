# Scope Paths: Framework-Maintained Captures

**Status: Proposal (design sketch). Derivation validated** — the
classification rule below is implemented as a reference in
`test/test_scope_paths.py` and reproduces the declared captures of real
lowered IR exactly (goto-lowered while/nested-for, dcc's token-threaded
nested loops). Running it as a parity check immediately found genuine
over-capture in the tree: `lower_for`/`lower_while` unconditionally
captured `%exit` in loop bodies that never break — dead dependency edges
the hand-maintained lists had been carrying. Fixed alongside the spike;
exactly the class of drift migration step 2 is designed to catch.

Goal: passes never read or write `.captures`. Mutation and construction APIs
maintain them, using scope information the pass framework already possesses.
Captures stay stored, explicit in ASM, and verifier-checked — the
check-usages-without-descending performance property is untouched. Only the
*authorship* of capture lists moves from pass authors to the framework.

Companion to `dgen/ir/uses.py` (ephemeral `Use` handles), which solved
capture maintenance for edge *rewiring*. This plan covers the two remaining
mutation classes: replacements that introduce new outer references, and
block construction.

## Why not explicit ownership

Captures cannot be derived from the graph alone: `block.values` is computed
*by stopping at captures*, so they define the scope partition rather than
reflect it. Automatic maintenance therefore needs scope information from
outside the block. Explicit ownership (`op.owner: Block`) would provide it
statically, always — but it taxes the IR's two best mechanics:

- **Zero-copy borrowing.** `staging._jit_evaluate` wraps a live subgraph in a
  throwaway `FunctionOp` on every stage-0 resolution. Ownership makes that
  illegal (copy on the hot path) or demands a borrow/view concept.
- **Wholesale restructuring.** `lower_while` builds new blocks whose values
  are largely the old block's ops. Any adoption scheme misclassifies them;
  explicit transfer ceremony lands in exactly the trickiest passes.

Scope information is only ever *needed* during a transform — and during a
transform, the pass framework already knows it, because it is walking the
block tree. Pay for scope context when mutating under a pass, not on every
mutation everywhere.

## Mechanism

### 1. `ScopePath` — the framework retains what it already computes

`Pass._lower_block` recursion is the path. Today each level iterates
`block.values` and discards it; instead, retain each level's value set:

```python
@dataclass(frozen=True)
class Scope:
    block: dgen.Block
    values: set[dgen.Value]        # block.values, materialized at descent
                                   # (args and parameters included)

class ScopePath:
    scopes: tuple[Scope, ...]      # root ... current

    def visible(self, value: dgen.Value) -> bool:
        return is_constant(value) or any(value in s.values for s in self.scopes)
```

The current path lives in a contextvar (the codebase already uses this shape:
`verify_passes`, codegen's `_emit_ctx`), pushed/popped by `_lower_block`.
Frontends, which maintain hand-rolled block stacks today (`dcc`'s `Parser`
scopes, `CLvalueToMemory._block_stack`), push the same contextvar via a
context manager: `with scope(block): ...`.

### 2. `derive_captures` — classification against the path

For a block under construction (or reconciliation), walk from `result`;
classify each reached value:

```python
def derive_captures(
    result: dgen.Value,
    args: Sequence[dgen.Value],
    parameters: Sequence[dgen.Value],
    path: ScopePath,
) -> list[dgen.Value]:
    declared = {*args, *parameters}
    captures, seen = [], set()

    def visit(value: dgen.Value) -> None:
        if value in seen or is_constant(value):
            return
        seen.add(value)
        if value in declared:
            return                              # local by declaration
        if isinstance(value, (BlockArgument, BlockParameter)):
            captures.append(value)              # boundary values are never
            return                              # implicitly local
        if path.visible(value):
            captures.append(value)              # pre-existing → external
            return
        # Fresh op created by the handler → local. Recurse: operands,
        # parameters, type, and child blocks' *declared captures* (their
        # dependencies must be satisfied here or captured onward).
        for edge in value.dependencies:
            visit(edge)

    visit(result)
    return captures
```

Two rules do the load-bearing work:

- **Freshness**: a value not in any path scope and not ambient must have been
  created by the current handler → local to the innermost new block that
  reaches it.
- **Boundary values are never implicitly local**: a `BlockArgument` /
  `BlockParameter` outside the new block's own declarations is always a
  capture. This classifies forward references to a parent-to-be — e.g.
  `merge_exit`, created fresh but destined to be the enclosing region's
  parameter — without knowing the parent exists yet.

### 3. Framework reconciliation — handlers just build

After a handler returns a replacement, and *before* recursing into its child
blocks (so nested lowering sees correct captures), the framework re-derives
captures for every block owned by the replacement, bottom-up:

```python
def _lower_block(self, block: dgen.Block) -> None:
    with self._path.entered(block):
        for v in block.values:
            result = self._dispatch_handlers(v)
            if result is not None:
                reconcile_captures(result, self._path)   # new: derive, bottom-up
            for _, child_block in (result or v).blocks:
                self._lower_block(child_block)
            if result is not None:
                block.replace_uses_of(v, result)
                prune_captures(block, retired_externals_of(v))  # new
```

With reconciliation in place, handlers stop declaring captures anywhere.
`dgen.Block(...)` keeps accepting an explicit `captures=` (the parser needs
it; escape hatch stays), but omitting it under a pass means "derive".

### 4. `Use.rebind` integration

`uses_of(..., within=...)` keeps its explicit root; under a pass the root
defaults to the current scope, and the "caller guarantees the new value is in
scope at the root" contract becomes a checked assertion: `path.visible(new)`.

## What each pain site becomes

**`lower_if`'s snapshot dance** (snapshot branch captures before
`_make_branch_label` mutates them, then splice into the region's list) —
deleted. The handler builds labels and region with no `captures=`;
reconciliation derives: branch bodies capture `merge_exit` (boundary rule),
the region captures `op.condition` plus the branches' external needs
(child-capture chaining), and `merge_exit` stays internal (declared as the
region's parameter).

**`redirect_to_exit`** (`block.captures.append(exit_param)`) — the append
goes; the branch it inserts references `exit_param`, and derivation adds the
capture via the boundary rule.

**`_resolve_jump_markers`** (returns a `needed: set[BlockParameter]` so
callers can patch child captures) — the return value and both call-site
patch loops go; same rule.

**`CLvalueToMemory._capture_alloca` + `_block_stack`** — the pass's private
block stack *is* a scope path; with the framework providing one, the
hoisted alloca's capture chain is derived wherever a buffer op references
it. The alloca-placement override remains (hoisting is policy, not capture
bookkeeping).

**`ThreadLoopMemory`** — already clean on `Use`; gains only the checked
rebind contract.

**Staging's borrow** — `Block(result=target)` in `_jit_evaluate` runs with
an *empty* path: every non-ambient reached value classifies as fresh/local,
captures derive to `[]`. That is exactly right for a stage-0 subgraph (no
block-argument dependencies by definition), so the borrow keeps working
unmodified.

## Migration

1. **Retain the path.** `ScopePath` + contextvar in `_lower_block`; no
   behavior change; expose `self.path` to handlers.
2. **Parity mode.** Implement `derive_captures`; in pass verification,
   derive for every constructed block and *assert equality* with the
   declared list across the whole suite. Divergences are either derivation
   bugs or latent capture bugs — both worth finding before flipping.
3. **Flip.** Reconciliation becomes authoritative; delete hand-written
   capture code pass by pass (`lower_if`, `redirect_to_exit`,
   `_resolve_jump_markers`, `raise_catch_to_goto`, `_capture_alloca`, toy's
   nest builders). ASM/parser captures stay explicit; the verifier is
   unchanged.
4. **Lint.** Direct writes to `.captures` outside `dgen/ir` /
   `dgen/passes` become a review smell (greppable).

## Costs and open questions

- **Retained value sets**: memory O(values on the current path); the
  materialization work already happens in `_lower_block`'s iteration.
  Reconciliation walks each new subtree once — the same order of work the
  closed-block verifier already does per pass in tests.
- **Fresh ops shared across sibling new blocks** classify as local to
  whichever block derivation reaches first, and double-ownership is caught
  by the existing verifier. No current pass does this; the `captures=`
  escape hatch covers a future one that must.
- **Ambient boundary**: `is_constant` is the classifier; staged type values
  (ops computing types) are correctly non-ambient and capture like any op.
- **Handlers that mutate matched blocks in place** (e.g. `lower_while`
  remapping args on `op.body` before wrapping it): reconciliation re-derives
  those blocks too since they're owned by the replacement — order of
  operations inside handlers stops mattering, which is the point.
- **Contextvar magic vs explicit parameter**: contextvar matches existing
  codebase practice and keeps `Block(...)` call sites clean; the explicit
  `scope=` parameter is worth offering for code running outside any pass.
