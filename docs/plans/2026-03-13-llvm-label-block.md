# Plan: Add body block to llvm.LabelOp, connect control flow in use-def graph

## Context

The current `llvm.LabelOp` is a flat marker with no body — it produces a `Label` value but contains no code. Labels have no data dependencies, so they're invisible to the use-def graph walker. The ops "inside" a label's basic block are only associated by their position in a flat list (`_stored_ops`). This blocks the migration to a pure graph-based IR.

The design in `docs/passes.md` (lines 244–292) specifies: labels are ops (values) with body blocks, and branches reference labels as operands. This makes control flow edges into data edges in the use-def graph.

**Scope:** LabelOp gets a body block. BrOp/CondBrOp reference labels as operands (not parameters). PhiOp stays (the llvm dialect stays close to LLVM's model). All generated IR is fully use-def connected — label body blocks use `Block(result=...)` with chains for side effects.

## Changes

### 1. `dgen/dialects/llvm.dgen` — dialect definition

**Before:**
```
op br<label: Label>() -> Nil
op cond_br<true_dest: Label, false_dest: Label>(cond: Int<1>) -> Nil
op label() -> Label
op phi<label_a: Label, label_b: Label>(a, b) -> Nil
```

**After:**
```
op label() -> Label:
    block body

op br(target) -> Nil
op cond_br(cond: Int<1>, true_target, false_target) -> Nil
op phi<label_a: Label, label_b: Label>(a, b) -> Nil
```

- `label` gains `block body` (same pattern as `affine.for` in `toy/dialects/affine.dgen`)
- `br` — `target` becomes an operand (in `()`, not `<>`), making it a use-def edge
- `cond_br` — `true_target`, `false_target`, and `cond` are all operands
- `phi` — unchanged (labels referenced as parameters are already use-def visible via `walk_ops` visiting `parameters`)

### 2. `dgen/dialects/llvm.pyi` — regenerate

Run `python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.pyi`.

Expected generated classes:
```python
class LabelOp(Op):
    body: Block
    type: Type = Label()

class BrOp(Op):
    target: Value
    type: Type = Nil()

class CondBrOp(Op):
    cond: Value
    true_target: Value
    false_target: Value
    type: Type = Nil()

class PhiOp(Op):        # unchanged
    label_a: Value[Label]
    label_b: Value[Label]
    a: Value
    b: Value
    type: Type = Nil()
```

### 3. `toy/passes/affine_to_llvm.py` — loop lowering

Current approach: yields ops linearly with labels as flat markers.

New approach — two-phase in `lower_function`:

**Phase 1:** `lower_op` / `_lower_for` still yield ops linearly with LabelOps as boundary markers (no body yet). Minimal changes to the yield logic:
- `BrOp(target=header_label_op)` instead of `BrOp(label=header_label_op)`
- `CondBrOp(cond=cmp, true_target=body_label_op, false_target=exit_label_op)` instead of `CondBrOp(cond=cmp, true_dest=body_label_op, false_dest=exit_label_op)`
- PhiOp stays the same

**Phase 2:** New grouping logic in `lower_function` after collecting the flat list:
1. Split flat ops at LabelOp boundaries → entry_ops + (label, body_ops) groups
2. For each group, build `Block(result=_chain_body(body_ops))` as the label's body
3. Function body = `Block(ops=entry_ops + [label_ops...] + trailing_ops, args=...)`

**`_chain_body` helper** (chains side-effecting ops so all are use-def reachable from result):
```python
def _chain_body(ops: list[Op]) -> Value:
    """Make all body ops reachable from a single root value."""
    if not ops:
        raise ValueError("Empty body")
    terminator = ops[-1]  # br, cond_br, or return — always last
    side_effects = [op for op in ops[:-1]
                    if isinstance(op, (llvm.StoreOp, llvm.CallOp))]
    if not side_effects:
        return terminator
    # ChainOp(lhs=se, rhs=rest): walk_ops visits lhs first
    # Chaining in forward order: se1 emitted before se2 before terminator
    result = terminator
    for se_op in side_effects:
        result = builtin.ChainOp(lhs=se_op, rhs=result)
    return result
```

Walk order for `ChainOp(lhs=se1, rhs=ChainOp(lhs=se2, rhs=terminator))`: walk_ops visits lhs first → se1, se2, terminator. Correct.

### 4. `dgen/passes/builtin_to_llvm.py` — if/else lowering

Same two-phase pattern:

**Phase 1 changes to `_lower_if`:**
- `CondBrOp(cond=..., true_target=then_label, false_target=else_label)` (operands, not params)
- `BrOp(target=merge_label)` (operand, not param)

**Phase 2:** Same grouping + chaining in `lower_function`.

### 5. `dgen/codegen.py` — LLVM IR emission

The codegen currently iterates `f.body.ops` linearly. With labels having bodies, it needs to also iterate each label's body ops.

**Structural change to `_emit_func`:**

```python
# Separate entry ops and labels
entry_ops = []
label_ops = []
for op in f.body.ops:
    if isinstance(op, llvm.LabelOp):
        label_ops.append(op)
    else:
        entry_ops.append(op)

# Register all ops (entry + all label bodies) in SlotTracker
tracker.register(entry_ops)
for label_op in label_ops:
    tracker.track_name(label_op)
    for arg in label_op.body.args:
        tracker.track_name(arg)
    tracker.register(label_op.body.ops)

# Pre-scan types for entry ops + all label body ops
# ... existing type registration loop, but over all_ops ...

# Emit entry block
lines = [f"define {llvm_ret} @{func_name}({param_str}) {{", "entry:"]
for op in entry_ops:
    _emit_op(op, ...)

# Emit each label's body
for label_op in label_ops:
    name = tracker.track_name(label_op)
    lines.append(f"{name}:")
    for body_op in label_op.body.ops:
        _emit_op(body_op, ...)
```

**Op emission changes:**
- `BrOp`: `br label %{tracker.track_name(op.target)}` (was `op.label`)
- `CondBrOp`: `br i1 %{...}, label %{tracker.track_name(op.true_target)}, label %{tracker.track_name(op.false_target)}` (was `op.true_dest`, `op.false_dest`)
- `LabelOp`: Skip in op emission loop (handled structurally as block labels)
- `ChainOp`: Already skipped (`continue`)
- `PhiOp`: Unchanged

### 6. `dgen/asm/formatting.py` — no changes needed

The generic formatter already handles ops with blocks (lines 179–206 in `formatting.py`). LabelOp with body renders as:
```
%header : llvm.Label = llvm.label() ():
    %phi : Nil = llvm.phi<%entry, %body>(%init, %next)
    %cmp : llvm.Int<1> = llvm.icmp<"slt">(%phi, %hi)
    %_ : Nil = llvm.cond_br(%cmp, %body, %exit)
```

BrOp with operand renders as: `%_ : Nil = llvm.br(%header)`

### 7. `dgen/asm/parser.py` — no changes needed

The generic parser handles ops with blocks (via `__blocks__` in `op_expression`, line 244) and operands. No special logic needed.

### 8. Test updates

**`toy/test/test_llvm_roundtrip.py`:**
- Update ASM format in all tests: labels get bodies with indented ops, br/cond_br use operand syntax
- Example new format:
```
%loop_header : llvm.Label = llvm.label() ():
    %i : Nil = llvm.phi<%entry, %loop_body>(%init, %next)
    %cmp : llvm.Int<1> = llvm.icmp<"slt">(%i, %hi)
    %_ : Nil = llvm.cond_br(%cmp, %loop_body, %loop_exit)
%_ : Nil = llvm.br(%loop_header)
```

**`toy/test/__snapshots__/`:**
- Regenerate all affected snapshots with `--snapshot-update`

## Files to modify

| File | Change |
|------|--------|
| `dgen/dialects/llvm.dgen` | Add `block body` to label, change br/cond_br from params to operands |
| `dgen/dialects/llvm.pyi` | Regenerate from llvm.dgen |
| `toy/passes/affine_to_llvm.py` | Two-phase: yield linearly then group into label bodies with chains |
| `dgen/passes/builtin_to_llvm.py` | Same two-phase restructuring for if/else |
| `dgen/codegen.py` | Iterate label bodies, handle new operand names |
| `toy/test/test_llvm_roundtrip.py` | Update ASM format in all test cases |
| `toy/test/__snapshots__/*` | Regenerate |

## Key patterns to reuse

- `Block(result=value, args=[...])` — graph-based block construction (`dgen/block.py`)
- `builtin.ChainOp(lhs=side_effect, rhs=rest)` — chain side effects (`dgen/module.py`)
- `PackOp(values=[...], type=...)` — list values (`dgen/module.py`)
- `op.blocks` iteration — generic block handling (`dgen/op.py:28-31`)
- `SlotTracker.register(ops)` — already recurses into blocks (`dgen/asm/formatting.py:48-57`)

## Verification

1. `python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.pyi` — regenerate stubs
2. `pytest toy/test/test_llvm_roundtrip.py -q` — ASM round-trip
3. `pytest toy/test/test_affine_to_llvm.py -q --snapshot-update` — lowering + update snapshots
4. `pytest toy/test/test_end_to_end.py -q --snapshot-update` — full pipeline with JIT execution
5. `pytest . -q` — all tests pass
6. `ruff format && ruff check --fix` — formatting and linting
