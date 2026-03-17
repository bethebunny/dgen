# Closed Blocks Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make closed blocks an IR invariant. Remove `_stored_ops`. Remove `PhiOp`. Block args + branch args are the only mechanism for cross-block value passing.

**Architecture:** Every label body block is closed: ops inside only reference locally-defined values, block arguments, or constants. `walk_ops` from any block's result gives the correct op list — no `_stored_ops` override needed. Labels nest naturally in the graph where they're referenced. Codegen flattens to LLVM's flat basic-block model. Lowering passes produce closed blocks directly.

**Tech Stack:** Python, pytest, dgen IR framework, llvmlite

---

## Background

`Block` has a dual-mode `ops` property: return `_stored_ops` if set, otherwise `walk_ops(self.result)`. This exists because `walk_ops` gives wrong results for label bodies — phi operands and open-block references cross block boundaries. The fix is to make blocks closed (block args for all cross-block values) and remove phi nodes, so `walk_ops` is always correct and `_stored_ops` can be deleted.

## Concrete example

**Before (current):**
```
%f = function<Nil>() ():
    %alloc = llvm.alloca<3>()
    %init = 0
    %_ = llvm.br(%header)
    %header = llvm.label() ():
        %i = llvm.phi<%entry, %body>(%init, %next)
        %hi = 3
        %cmp = llvm.icmp<"slt">(%i, %hi)
        %_ = llvm.cond_br(%cmp, %body, %exit)
    %body = llvm.label() ():
        %ptr = llvm.gep(%alloc, %i)
        %val = llvm.load(%ptr)
        %_ = llvm.store(%val, %ptr)
        %one = 1
        %next = llvm.add(%i, %one)
        %_ = llvm.br(%header)
    %exit = llvm.label() ():
        %_ = return(())
```

Problems: `%body` references `%alloc` (entry) and `%i` (header). PhiOp references `%init` (entry) and `%next` (body). Blocks are open.

**After:**
```
%f = function<Nil>() ():
    %alloc = llvm.alloca<3>()
    %init = 0
    %header = llvm.label() (%i: Index, %p: Ptr):
        %hi = 3
        %cmp = llvm.icmp<"slt">(%i, %hi)
        %body = llvm.label() (%j: Index, %q: Ptr):
            %ptr = llvm.gep(%q, %j)
            %val = llvm.load(%ptr)
            %_ = llvm.store(%val, %ptr)
            %one = 1
            %next = llvm.add(%j, %one)
            %_ = llvm.br(%header, [%next, %q])
        %exit = llvm.label() ():
            %_ : Nil = ()
        %_ = llvm.cond_br(%cmp, %body, %exit, [%i, %p], [])
    %_ = llvm.br(%header, [%init, %alloc])
```

- No PhiOp. Block args on header receive `(%init, %alloc)` from entry and `(%next, %q)` from body — codegen emits phi from this.
- Every label body is closed. `walk_ops` from any body's result finds only local ops + referenced labels.
- Labels nest naturally where they're referenced. Codegen flattens when emitting LLVM IR.
- `%_ : Nil = ()` is the empty-block "pass" pattern (no `return(())`).

## File Structure

| File | Change |
|------|--------|
| `dgen/dialects/llvm.dgen` | Remove `phi`, add `args`/`true_args`/`false_args` to branch ops |
| `dgen/dialects/llvm.pyi` | Regenerate |
| `dgen/block.py` | Add `verify_closed()`, remove `_stored_ops` |
| `dgen/graph.py` | Remove `group_into_blocks`, `chain_body`, `placeholder_block`, `unwrap_chain` |
| `dgen/passes/pass_.py` | Remove `block._stored_ops = None` line |
| `dgen/asm/parser.py` | Fix empty-list-to-PackOp coercion; parse `%_ : Nil = ()` as constant |
| `toy/passes/affine_to_llvm.py` | Produce closed blocks directly with block args |
| `dgen/passes/builtin_to_llvm.py` | Produce closed blocks directly with block args |
| `dgen/codegen.py` | Recursively collect labels, emit phi from block args + predecessor scan |
| `test/test_llvm_roundtrip.py` | Update for new syntax |
| `toy/test/test_affine_to_llvm.py` | Update snapshots |
| `toy/test/test_end_to_end.py` | Verify unchanged JIT output |

---

## Task 1: Update llvm dialect — remove phi, add args to branches

**Files:**
- Modify: `dgen/dialects/llvm.dgen`
- Regenerate: `dgen/dialects/llvm.pyi`
- Modify: all files that reference `PhiOp`, `BrOp`, `CondBrOp`

- [ ] **Step 1: Update llvm.dgen**

Remove the `phi` line. Update branch ops:

```
op br(target, args: List) -> Nil
op cond_br(cond: Int<1>, true_target, false_target, true_args: List, false_args: List) -> Nil
```

- [ ] **Step 2: Regenerate stubs**

Run: `python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.pyi`

- [ ] **Step 3: Stub all broken call sites with empty packs**

Every existing `BrOp(target=x)` becomes `BrOp(target=x, args=empty_pack)`. Every `CondBrOp` gets `true_args=empty_pack, false_args=empty_pack`. Remove all `PhiOp` references (codegen, lowering passes). Use a module-level helper:

```python
_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))
```

Files: `toy/passes/affine_to_llvm.py`, `dgen/passes/builtin_to_llvm.py`, `dgen/codegen.py`.

Update existing roundtrip tests in `test/test_llvm_roundtrip.py` to include args in ASM: `llvm.br(%target, [])`, `llvm.cond_br(%cmp, %a, %b, [], [])`.

- [ ] **Step 4: Verify non-control-flow tests pass**

Run: `pytest test/test_type_roundtrip.py test/test_type_values.py test/test_graph.py -q`

- [ ] **Step 5: Commit**

```bash
jj commit -m "Remove PhiOp from llvm dialect, add args to branch ops"
```

---

## Task 2: Fix ASM parser for empty list coercion and pass pattern

**Files:**
- Modify: `dgen/asm/parser.py`
- Test: `test/test_llvm_roundtrip.py`

- [ ] **Step 1: Write failing tests**

Add to `test/test_llvm_roundtrip.py`:

```python
def test_roundtrip_br_with_args():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %init : Index = 0
        |     %header : llvm.Label = llvm.label() (%i: Index):
        |         %_ : Nil = ()
        |     %_ : Nil = llvm.br(%header, [%init])
        |     %_ : Nil = ()
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_cond_br_with_args():
    ir = strip_prefix("""
        | import llvm
        |
        | %f : Nil = function<Nil>() ():
        |     %init : Index = 0
        |     %cmp : Nil = llvm.icmp<"slt">(%init, %init)
        |     %body : llvm.Label = llvm.label() (%i: Index):
        |         %_ : Nil = ()
        |     %exit : llvm.Label = llvm.label() ():
        |         %_ : Nil = ()
        |     %_ : Nil = llvm.cond_br(%cmp, %body, %exit, [%init], [])
        |     %_ : Nil = ()
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest test/test_llvm_roundtrip.py::test_roundtrip_br_with_args test/test_llvm_roundtrip.py::test_roundtrip_cond_br_with_args -v`

- [ ] **Step 3: Fix parser**

Two fixes in `dgen/asm/parser.py`:

1. **Empty list coercion**: In `_coerce_operand`, add `or issubclass(field_type, builtin.List)` to the list-wrapping condition so empty `[]` becomes a PackOp.

2. **Pass pattern**: In `op_statement`, `()` after `=` should parse as a Nil constant. Add `"("` to `_LITERAL_START` or handle `()` as a special case in the constant literal path.

- [ ] **Step 4: Verify**

Run: `pytest test/test_llvm_roundtrip.py -v`

- [ ] **Step 5: Commit**

```bash
jj commit -m "Parser: empty list coercion for List-typed fields, Nil pass pattern"
```

---

## Task 3: Block.verify_closed() and remove _stored_ops

**Files:**
- Modify: `dgen/block.py`
- Modify: `dgen/passes/pass_.py`
- Modify: `dgen/graph.py`
- Modify: `dgen/asm/parser.py` (Block construction)
- Test: `test/test_graph.py`

- [ ] **Step 1: Write test for verify_closed**

Add to `test/test_graph.py`:

```python
def test_verify_closed_passes_for_local_refs():
    arg = BlockArgument(name="x", type=builtin.Index())
    one = ConstantOp(value=1, type=builtin.Index())
    add_op = llvm.AddOp(lhs=arg, rhs=one)
    block = Block(result=add_op, args=[arg])
    block.verify_closed()  # should not raise


def test_verify_closed_fails_for_external_ref():
    external_op = llvm.AllocaOp(elem_count=ConstantOp(value=3, type=builtin.Index()))
    gep = llvm.GepOp(base=external_op, index=ConstantOp(value=0, type=builtin.Index()))
    block = Block(result=gep, args=[])
    with pytest.raises(ValueError):
        block.verify_closed()
```

- [ ] **Step 2: Implement verify_closed**

Add to `Block` in `dgen/block.py`. An op's operand is allowed if it's in `self.ops` (local), in `self.args`, is a `Constant`, is a `Type`, or is an op with blocks (LabelOp — branch target, allowed until SoN migration). Check PackOp contents recursively.

- [ ] **Step 3: Remove _stored_ops from Block**

Simplify `Block.__init__` to only accept `result` and `args`. Remove `_stored_ops`, remove `ops=` parameter, make `ops` property always call `walk_ops(self.result)`. Remove `block._stored_ops = None` from `pass_.py:129`.

- [ ] **Step 4: Update all Block construction sites**

Every `Block(ops=..., ...)` becomes `Block(result=..., ...)`. The result is the last op (the terminator or chain root). Key files:
- `dgen/asm/parser.py:387` — `Block(ops=ops, args=args)` → `Block(result=ops[-1], args=args)`
- `toy/passes/affine_to_llvm.py` — updated in Task 4
- `dgen/passes/builtin_to_llvm.py` — updated in Task 5
- `dgen/staging.py` — update Block construction sites
- Test files — update Block construction

- [ ] **Step 5: Remove graph.py helpers that exist only for _stored_ops**

Remove `group_into_blocks`, `chain_body`, `placeholder_block`, `unwrap_chain` from `dgen/graph.py`. These exist to build `_stored_ops` lists. With closed blocks and graph-derived ops, they're unnecessary.

- [ ] **Step 6: Verify**

Run: `pytest . -q`

- [ ] **Step 7: Commit**

```bash
jj commit -m "Remove _stored_ops, Block.verify_closed(), simplify graph.py"
```

---

## Task 4: Rewrite affine_to_llvm to produce closed blocks

The pass builds closed label bodies directly. Every value used in a label body is either defined there, a block arg, or a constant. No two-phase grouping — labels are created with their bodies inline.

**Files:**
- Modify: `toy/passes/affine_to_llvm.py`
- Test: `toy/test/test_affine_to_llvm.py`

- [ ] **Step 1: Rewrite `_lower_for`**

The loop lowering creates header, body, exit labels with block args for the loop variable and any live values (allocas). Body ops reference their own block args. Branches pass values explicitly.

The pass tracks which values are live across block boundaries (allocas in `alloc_shapes`) and threads them as block args. `self.current_label` is no longer needed — block args replace the phi patching mechanism.

The result of each label body is the terminator (br or cond_br). Side-effecting ops (stores) are chained to the terminator via `ChainOp` so `walk_ops` discovers them.

- [ ] **Step 2: Rewrite `_lower_function`**

No more two-phase lowering. The pass yields ops that form a valid graph directly. The function body's result is the last op; labels nest naturally via branch target references.

- [ ] **Step 3: Call verify_closed on label bodies**

After building each label body, call `label_op.body.verify_closed()` to assert the invariant.

- [ ] **Step 4: Update snapshots**

Run: `pytest toy/test/test_affine_to_llvm.py -v --snapshot-update`

Review: block args on labels, no phi, closed bodies.

- [ ] **Step 5: Commit**

```bash
jj commit -m "affine_to_llvm: produce closed blocks with block args"
```

---

## Task 5: Rewrite builtin_to_llvm to produce closed blocks

Same approach for if/else lowering. The merge label gets a block arg for the result. Then/else branches pass their results via br args.

**Files:**
- Modify: `dgen/passes/builtin_to_llvm.py`

- [ ] **Step 1: Rewrite `_lower_if`**

Merge label has a block arg for the result value. Then/else each `br` to merge passing their result. No PhiOp.

- [ ] **Step 2: Rewrite `_lower_function`**

Same as Task 4 — no two-phase grouping.

- [ ] **Step 3: Verify**

Run: `pytest toy/test/test_affine_to_llvm.py toy/test/test_toy_to_affine.py -v`

- [ ] **Step 4: Commit**

```bash
jj commit -m "builtin_to_llvm: produce closed blocks for if/else"
```

---

## Task 6: Update codegen — phi from block args, recursive label collection

Codegen collects all labels recursively (they may be nested), flattens to LLVM's basic-block model, and emits phi instructions by scanning predecessor branches.

**Files:**
- Modify: `dgen/codegen.py`
- Test: `toy/test/test_end_to_end.py`

- [ ] **Step 1: Recursive label collection**

Replace the current flat split with a recursive walk:

```python
def _collect_labels(ops: list[dgen.Op]) -> list[llvm.LabelOp]:
    """Recursively collect all LabelOps from a list of ops."""
    labels: list[llvm.LabelOp] = []
    for op in ops:
        if isinstance(op, llvm.LabelOp):
            labels.append(op)
            labels.extend(_collect_labels(op.body.ops))
    return labels
```

Entry ops are non-label ops at the function body level (same as today).

- [ ] **Step 2: Build predecessor map**

For each label, scan all ops across all blocks to find branches targeting it. Map label → list of (predecessor_label, passed_arg_values). Use a dict keyed on the label op directly (not `id()`).

- [ ] **Step 3: Emit phi for block args**

At the start of each label's emission, for each block arg, emit a phi instruction using the predecessor map.

- [ ] **Step 4: Register block arg types**

Add label body block args to the type registration loop so they get LLVM type strings.

- [ ] **Step 5: Remove old PhiOp emission code**

Delete the `isinstance(op, llvm.PhiOp)` branches.

- [ ] **Step 6: Verify**

Run: `pytest toy/test/test_end_to_end.py -v`

Verify JIT output is unchanged:
```bash
python -m toy.cli toy/test/testdata/constant.toy
python -m toy.cli toy/test/testdata/transpose.toy
python -m toy.cli toy/test/testdata/multiply_transpose.toy
```

- [ ] **Step 7: Commit**

```bash
jj commit -m "codegen: recursive label collection, phi from block args"
```

---

## Task 7: Update tests and final verification

**Files:**
- Modify: `test/test_llvm_roundtrip.py`
- Update: `toy/test/__snapshots__/`

- [ ] **Step 1: Rewrite roundtrip tests**

Replace phi-based tests with block-arg versions. Add a full loop roundtrip test with nested labels and block args (matching the "After" example from this plan).

- [ ] **Step 2: Update snapshots**

Run: `pytest toy/test/ --snapshot-update -q`

- [ ] **Step 3: Full suite**

Run: `pytest . -q`

- [ ] **Step 4: Lint and type check**

Run: `ruff format && ruff check --fix && ty check`

- [ ] **Step 5: Commit**

```bash
jj commit -m "Closed blocks: all tests updated and passing"
```

---

## Design Decisions

### Closed blocks are an invariant, not a fixup

Lowering passes produce closed blocks directly. `Block.verify_closed()` asserts this. There is no `close_blocks()` utility — that would imply open blocks are a valid intermediate state.

### PhiOp removed from llvm dialect

PhiOp's purpose is to reference values from other blocks, which violates closure. Codegen reconstructs phi instructions from block args + predecessor scanning. The llvm dialect is "as close as is reasonable and ergonomic to LLVM," not a strict 1:1 mapping.

### Labels nest naturally

Labels appear in the graph where they're referenced (inside other label bodies). This is the natural graph structure from `walk_ops`. Codegen flattens to LLVM's model. The formatter prints the nesting. Both are display concerns.

### `_stored_ops` removed entirely

With closed blocks, `walk_ops` gives correct results for all blocks. The dual-mode `ops` property is gone. `Block` stores `result` and `args`; ops are always derived from the graph.

### `%_ : Nil = ()` is the pass pattern

Empty blocks (like loop exit) use `%_ : Nil = ()` instead of `return(())`. Clearer intent — "this block has no meaningful result."

## Relationship to Other Plans

- **`docs/plans/2026-03-12-remove-stored-ops.md`**: Superseded. `_stored_ops` is removed entirely in this plan.
- **`docs/plans/2026-03-13-llvm-label-block.md`**: Already implemented. This plan builds on it.
- **`docs/control-flow.md`**: This plan resolves problems (1) and (3) from the diagnosis. Problem (2) (branch targets as value references) remains — `walk_ops` still discovers target LabelOps in label bodies. This is acceptable because LabelOps are structural and consumers handle them naturally. The full fix is Sea-of-Nodes (future work).
