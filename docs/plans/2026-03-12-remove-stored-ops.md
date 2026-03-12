# Finish Removing `_stored_ops` from Block

## Context

`Block._stored_ops` is a backward-compatibility shim enabling gradual migration from a list-based IR model (where blocks store an explicit `list[Op]`) to a graph-based model (where blocks store only a `result` value and ops are derived via `walk_ops`). The destination is a pure graph-based `Block` with no `_stored_ops`, no `ops.setter`, and no `Block(ops=...)` constructor path.

**Current state — what's already done:**
- Task 1 (function-level imports): Done — `block.py` and `pass_.py` use top-level imports
- Task 2 (`_run_block` clears `_stored_ops`): Done — `pass_.py` line 135 sets `block._stored_ops = None`
- `optimize()` entry point: Already simplified to `return ToyOptimize().run(m)`

**Remaining `_stored_ops` callers:**
| Site | Usage |
|------|-------|
| `dgen/block.py` | Definition + `ops` property/setter |
| `dgen/passes/pass_.py:135` | `block._stored_ops = None` (the transition step) |
| `dgen/asm/parser.py:367,378` | `Block(ops=[], ...)` and `Block(ops=ops, ...)` |
| `dgen/staging.py:119,153,316,487` | `Block(ops=ops, ...)` construction + `func.body.ops = [...]` mutation |
| `toy/passes/toy_to_affine.py:48,113` | `Block(ops=ops, ...)` |
| `toy/passes/affine_to_llvm.py:55` | `Block(ops=ops, ...)` |
| `dgen/passes/builtin_to_llvm.py:36` | `Block(ops=ops, ...)` |
| `toy/parser/lowering.py:64` | `Block(ops=ops, ...)` |
| `test/test_peano.py:151,160` | `block.ops = ...` mutation |
| Various test files | `Block(ops=[...])` construction |

---

## Plan

### Step 1: Complete Task 3 — Remove `replace_op` from Rewriter

**File:** `dgen/passes/pass_.py`, `toy/passes/optimize.py`

In `optimize.py`, two handlers use `rewriter.replace_op`:
- `fold_constants`: change `rewriter.replace_op(op, new_op)` → `rewriter.replace_uses(op, new_op)`
- `simplify_reshape` (reshape-of-reshape case): change `rewriter.replace_op(op, new_op)` → `rewriter.replace_uses(op, new_op)`

In `pass_.py`, remove `replace_op` method and `_op_replacements` dict from `Rewriter`.

Note: The test IR in `test/test_pass.py` already uses chained returns (`return(%3)`, `return(%1)`, etc.) that make side effects reachable from `block.result`. No test IR updates needed — the snapshots in `test/snapshots/` should pass as-is after `_stored_ops` is cleared.

Run `pytest test/test_pass.py toy/test/test_optimize.py -q` to verify.

### Step 2: Complete Task 4 — Remove dead DCE code from `optimize.py`

**File:** `toy/passes/optimize.py`

Remove: `eliminate_dead_code`, `collect_uses`, `_remove_indices` functions.
Remove unused imports: `Sequence` (from `collections.abc`), `dgen` (if no longer used).

The test IR in `toy/test/test_optimize.py` needs to chain `return` to the last side-effecting op (as specified in the existing plan, Task 4 Step 1). Snapshot files will need updating for the DCE tests.

Run `pytest toy/test/test_optimize.py -q` to verify.

### Step 3: Task 5 — Lint/format checkpoint

```bash
ruff format && ruff check --fix && ty check
pytest . -q
jj commit -m "remove replace_op and dead DCE code (pass infra phase 1)"
```

### Step 4: Tasks 6–10 — Port remaining passes to Pass infrastructure

Following the detailed specifications in `docs/plans/2026-03-10-pass-review-fixes.md`:

- **Task 6**: Port `toy/passes/shape_inference.py`
- **Task 7**: Port `toy/passes/toy_to_affine.py` (converts `Block(ops=ops, ...)` sites at lines 48 and 113)
- **Task 8**: Port `toy/passes/affine_to_llvm.py` (converts line 55)
- **Task 9**: Port `dgen/passes/builtin_to_llvm.py` (converts line 36)
- **Task 10**: Final lint + validation

After Tasks 6–10: `toy/parser/lowering.py:64` still uses `Block(ops=ops, ...)`.

### Step 5: Convert `dgen/asm/parser.py` to `result=`

**File:** `dgen/asm/parser.py`

In `_read_block_body`:
- Line 378: `Block(ops=ops, args=args)` → `Block(result=ops[-1], args=args)`
- Line 367: `Block(ops=[], args=args)` — empty block (no body). This currently would raise `ValueError` from `Block.__init__` since neither `result=` nor non-empty `ops=` is provided. Need to decide: allow `Block(args=args)` with no result, or assert this case never occurs in valid input.

**Recommendation**: Extend `Block.__init__` to allow `result=None` with no ops (for arg-only blocks representing empty function bodies), setting `self.result` to a sentinel/None-typed value. Alternatively, check if this case is ever exercised in practice — if not, add an assertion.

### Step 6: Convert `toy/parser/lowering.py:64`

Change `dgen.Block(ops=ops, args=args)` to `dgen.Block(result=ops[-1], args=args)`.

### Step 7: Convert test file Block constructions

**Files:** `test/test_graph.py`, `test/test_type_values.py`, `toy/test/test_type_roundtrip.py`, `toy/test/test_toy_printer.py`

All use `Block(ops=[op1, op2, ...], args=[...])`. Change to `Block(result=last_op, args=[...])`.

### Step 8: Refactor `dgen/staging.py`

**File:** `dgen/staging.py`

Three mutation sites:
- Line 119: `dgen.Block(ops=ops, args=list(block_args))` → `Block(result=ops[-1], args=...)`
- Line 153: `func.body.ops = [o for o in func.body.ops if id(o) not in subgraph_ids]` — removes the subgraph ops after JIT. In graph model, once the field `op.field_name = const_op` is patched (line 160), the subgraph ops become unreachable from `block.result` and are naturally dead. This line can be removed.
- Line 316: `func.body.ops = new_ops` — replaces the ops list entirely. Need to trace what `new_ops` is and ensure `block.result` is updated to `new_ops[-1]`. Change to set `func.body.result = new_ops[-1]`.
- Line 487: `dgen.Block(ops=[pack, call_op, ret_op], args=thunk_args)` → `Block(result=ret_op, args=thunk_args)` (ret_op is the last op)

### Step 9: Refactor `test/test_peano.py`

**File:** `test/test_peano.py`

Lines 151, 160 use `block.ops = _lower_peano_ops(block.ops, replacements)`. This is hand-rolled lowering that mutates ops lists.

Two options:
1. Port the Peano lowering to Pass infrastructure (preferred, consistent with project direction)
2. Update to use `block.result = new_ops[-1]` (minimal change, avoids full pass port)

Option 2 is simpler: since `_lower_peano_ops` returns the full replacement list, set `block.result = result_ops[-1]` instead of `block.ops = result_ops`.

### Step 10: Remove `_stored_ops` from `Block`

**File:** `dgen/block.py`

Once all callers are converted:
- Remove `_stored_ops` field and all references to it
- Remove `ops.setter`
- Change `Block.__init__` to accept only `result=` (or no result for arg-only blocks)
- Remove `ops=` parameter from `Block.__init__`
- Update the `ops` property to always call `walk_ops(self.result)`
- Remove the `block._stored_ops = None` line from `pass_.py` (no longer needed)

---

## Critical Files

- `dgen/block.py` — Block definition (final target)
- `dgen/passes/pass_.py` — Rewriter, `_run_block`
- `toy/passes/optimize.py` — Step 1-2 changes
- `dgen/asm/parser.py` — Step 5
- `dgen/staging.py` — Step 8 (most complex)
- `test/test_peano.py` — Step 9
- `toy/passes/toy_to_affine.py`, `affine_to_llvm.py`, `dgen/passes/builtin_to_llvm.py` — Step 4

## Verification

After each step, run `pytest . -q` (110 tests, ~1s). Final state: all tests pass, `grep -r '_stored_ops\|Block(ops=' .` returns no results outside of `block.py` itself, then `block.py` is cleaned up.
