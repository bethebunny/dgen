# LLVM Dialect: Blocks-as-Values + Remove `_stored_ops`

## Context

`Block._stored_ops` is an escape hatch for lowering passes where control-flow ops (`BrOp`, `LabelOp`, `CondBrOp`) return `Nil` and have no downstream users, making them unreachable via `walk_ops`. The approved design (`docs/passes.md` §"Unstructured control flow: blocks as values") solves this:

- `LabelOp` gets a `body: Block` whose `body.args` are block arguments (replacing phi nodes)
- `BrOp`/`CondBrOp` get optional `arg`/`true_arg`/`false_arg` operands to pass values to the target's block args
- `PhiOp` is removed entirely — block args fill its role
- All ops are chained so they're reachable from `block.result`; `_stored_ops` can be removed

**The phi cycle problem** (`phi.b = next_op`, `next_op.lhs = phi_op`) disappears because `walk_ops` does not descend into nested `Op` blocks — the self-referencing `cond_br` inside `loop_label.body` is invisible to the top-level graph traversal.

**Correct walk_ops ordering for LLVM IR**: `BrOp.dest` remains a string parameter (not a value dep on `LabelOp`), so the chain `chain(alloca, chain(br_entry, chain(loop_label, chain(exit_label, ...))))` produces walk_ops order: alloca → br_entry → loop_label → exit_label → post-loop → ret — matching the expected LLVM IR basic-block order. This is the **flat continuation** model: exit label is a thin marker, post-loop ops continue flat in the function body after the exit label.

---

## Critical Files

- `dgen/dialects/llvm.dgen` — add `block body` to LabelOp; add `arg`/`true_arg`/`false_arg`; remove phi
- `dgen/dialects/llvm.py` — regenerate (never hand-edit)
- `dgen/codegen.py` — emit phi nodes from LabelOp block args; remove PhiOp case; add predecessor scan
- `toy/passes/affine_to_llvm.py` — rewrite ForOp lowering with correct chain order
- `dgen/passes/builtin_to_llvm.py` — rewrite IfOp lowering; merge label block arg as result
- `dgen/block.py` — remove `_stored_ops`, `ops=` constructor, `ops` setter
- `dgen/passes/pass_.py` — remove `block._stored_ops = None`
- `dgen/asm/parser.py` — `Block(ops=ops)` → `Block(result=ops[-1])`
- `dgen/staging.py` — three `Block(ops=...)` sites + `block.ops = ...` mutations
- `toy/parser/lowering.py`, `toy/passes/toy_to_affine.py` — `Block(ops=...)` callers
- `test/test_peano.py` — `block.ops = ...` → `block.result = new_ops[-1]`
- Various test files — `Block(ops=[...])` → `Block(result=last_op)`
- `toy/test/test_llvm_roundtrip.py` — update loop-pattern test
- `toy/test/__snapshots__/test_affine_to_llvm/` — regenerate

---

## Implementation Steps

### Step 1: Update `llvm.dgen` and regenerate `llvm.py`

**`dgen/dialects/llvm.dgen`** — make these changes:

```
# Add block body to LabelOp:
op label<label_name: String>() -> Nil:
    block body

# Add optional arg to BrOp (passes a value to target label's block arg):
op br<dest: String>(arg = Nil) -> Nil

# Add optional args to CondBrOp:
op cond_br<true_dest: String, false_dest: String>(cond: Int<1>, true_arg = Nil, false_arg = Nil) -> Nil

# REMOVE this line entirely:
# op phi<label_a: String, label_b: String>(a, b) -> Nil
```

Then regenerate:
```bash
python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.py
```

Verify: `LabelOp` has `body: Block` and `__blocks__ = ("body",)`; `BrOp` has `arg: Value | Nil = Nil()`; `CondBrOp` has `true_arg: Value | Nil = Nil()`, `false_arg: Value | Nil = Nil()`; `PhiOp` is gone.

### Step 2: Update `codegen.py`

**File**: `dgen/codegen.py`

**Add predecessor helper** — call it before the main emission loop to build `phi_preds: dict[str, list[tuple[str, dgen.Value | None]]]`:

```python
def _collect_phi_preds(
    ops: list[dgen.Op],
    current_block: str,
    out: dict[str, list[tuple[str, dgen.Value | None]]],
) -> None:
    for op in ops:
        if isinstance(op, llvm.BrOp):
            dest = string_value(op.dest)
            arg = None if isinstance(op.arg, builtin.Nil) else op.arg
            out.setdefault(dest, []).append((current_block, arg))
        elif isinstance(op, llvm.CondBrOp):
            for dest, val in [
                (string_value(op.true_dest), None if isinstance(op.true_arg, builtin.Nil) else op.true_arg),
                (string_value(op.false_dest), None if isinstance(op.false_arg, builtin.Nil) else op.false_arg),
            ]:
                out.setdefault(dest, []).append((current_block, val))
        elif isinstance(op, llvm.LabelOp):
            _collect_phi_preds(op.body.ops, string_value(op.label_name), out)
```

Call: `phi_preds: dict[str, list[tuple[str, dgen.Value | None]]] = {}` then `_collect_phi_preds(f.body.ops, "entry", phi_preds)`.

**Pre-scan**: extend the existing pre-scan `for op in f.body.ops` to also recurse into `LabelOp` bodies — register constant operands and types for nested ops.

**Main loop** — update `LabelOp` case:
```python
if isinstance(op, llvm.LabelOp):
    label_name = string_value(op.label_name)
    lines.append(f"{label_name}:")
    preds = phi_preds.get(label_name, [])
    for k, arg in enumerate(op.body.args):
        ty = types.get(arg, "i64")
        arg_name = tracker.track_name(arg)
        phi_parts = [
            f"[ {bare_ref(val)}, %{pred_name} ]"
            for pred_name, val in preds
            if val is not None
        ]
        if phi_parts:
            lines.append(f"  %{arg_name} = phi {ty} {', '.join(phi_parts)}")
    for body_op in op.body.ops:
        # emit body_op using same logic as the main loop
        ...
```

**Remove** the `elif isinstance(op, llvm.PhiOp)` case.

**Tracker setup**: before `tracker.register(f.body.ops)`, also call `tracker.register(label.body.ops)` for each `LabelOp` in `f.body.ops` (and recursively for nested labels). Also call `tracker.track_name(arg)` for each `label.body.args`.

### Step 3: Rewrite `affine_to_llvm.py` ForOp lowering

**File**: `toy/passes/affine_to_llvm.py`

**`_lower_for`**: Create a `BlockArgument` as the loop variable (replaces phi). Emit:

1. `init_op = ConstantOp(lo, Index())` — yield
2. Build loop body ops (hi, cmp, body ops, one, next)
3. Build `cond_br_op = CondBrOp(true_dest=exit_label, false_dest=header_label, cond=cmp, true_arg=Nil, false_arg=next_op)` — this is the body terminator
4. Build `exit_label_op = LabelOp(label_name=exit_name, body=Block(result=thin_placeholder, args=[]))`
5. Build `loop_label_op = LabelOp(label_name=header_name, body=Block(result=body_chain, args=[loop_var]))` where `body_chain` threads body ops + cond_br
6. `br_entry = BrOp(dest=header_name, arg=init_op)` — yield
7. Chain: yield `exit_label_op`, yield `loop_label_op`, yield `chain(prev_val, br_entry, ...)`, yield `chain(prev, loop_label_op, ...)`, yield `chain(prev, exit_label_op, ...)`

**Critical chain order** ensuring walk_ops visits in LLVM IR order:
```
chain(alloca, chain(br_entry, chain(loop_label_op, exit_label_op)))
```
→ walk_ops: alloca → br_entry → loop_label_op → exit_label_op → (post-loop)

`value_map[for_op]` = the final chain (type = alloca type / Ptr).

**`lower_function`**: change `Block(ops=ops, args=f.body.args)` → `Block(result=ops[-1], args=f.body.args)`.

### Step 4: Rewrite `builtin_to_llvm.py` IfOp lowering

**File**: `dgen/passes/builtin_to_llvm.py`

Create `merge_result_arg = BlockArgument(type=if_op.type)`. The merge LabelOp's body has `args=[merge_result_arg]`. Then/else BrOps pass their result values: `BrOp(dest="merge", arg=then_val)` / `BrOp(dest="merge", arg=else_val)`.

After building: `value_map[if_op] = merge_result_arg` (a `BlockArgument`, directly usable as a `Value` by downstream ops).

Chain order: `chain(cond_br_op, chain(then_label, chain(else_label, merge_label)))`.

`lower_function` → `Block(result=ops[-1], args=f.body.args)`.

### Step 5: Remove `_stored_ops` from `block.py`

**File**: `dgen/block.py`

```python
class Block:
    result: dgen.Value
    args: list[BlockArgument]

    def __init__(self, *, result: dgen.Value, args: list[BlockArgument] | None = None) -> None:
        self.result = result
        self.args = args if args is not None else []

    @property
    def ops(self) -> list[dgen.Op]:
        return walk_ops(self.result)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
```

### Step 6: Remove `_stored_ops = None` from `pass_.py`

**File**: `dgen/passes/pass_.py` — remove `block._stored_ops = None`.

### Step 7: Convert all remaining `Block(ops=...)` callers

Sites (from `docs/plans/2026-03-12-remove-stored-ops.md`):

| File | Change |
|------|--------|
| `dgen/asm/parser.py:367,378` | `Block(ops=ops)` → `Block(result=ops[-1])` |
| `dgen/staging.py:119,153,316,487` | See plan doc for per-site details |
| `toy/parser/lowering.py:64` | `Block(ops=ops)` → `Block(result=ops[-1])` |
| `toy/passes/toy_to_affine.py:48,113` | Same |
| `test/test_peano.py:151,160` | `block.ops = new_ops` → `block.result = new_ops[-1]` |
| Various test files | `Block(ops=[op1, op2])` → `Block(result=last_op)` |

### Step 8: Update tests and regenerate snapshots

```bash
pytest toy/test/test_affine_to_llvm.py --snapshot-update
pytest toy/test/test_llvm_roundtrip.py -v
```

`test_llvm_roundtrip.py` loop pattern: update to use `BrOp(arg=...)`, `CondBrOp(true_arg=..., false_arg=...)`, and `LabelOp` with body blocks and block args instead of `PhiOp`.

---

## Verification

```bash
# Step 1:
python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.py
pytest . -q

# After each step:
pytest . -q

# Final:
ruff format && ruff check --fix && ty check
pytest . -q  # all ~110 tests pass
```

Expected: `grep -r '_stored_ops\|Block(ops=\|PhiOp' .` returns no results (outside docs/plans).

---

## Branch

Work on branch: `claude/llvm-dialect-cleanup-cJfr8`
