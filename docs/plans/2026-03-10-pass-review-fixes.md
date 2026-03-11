# Pass Infrastructure: Review Fixes + Pass Migration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address code review feedback — commit fully to the graph-based IR model in the pass infrastructure, removing list-based shims and rejected design elements — then port all remaining passes to the Pass base class.

**Architecture:** The design doc says Block stores a result value and derives ops via `walk_ops`. The current pass infrastructure hedges between list-based and graph-based models. Phase 1 (Tasks 1-5) removes list-based code from the pass infrastructure: `_run_block` clears `_stored_ops` after running so ops are graph-derived, `replace_op` is removed (design doc rejected it), `eliminate_dead_code` is removed (graph model makes DCE implicit), and `deepcopy` is removed (passes mutate in place). Phase 2 (Tasks 6-10) ports the remaining four passes (`shape_inference`, `toy_to_affine`, `affine_to_llvm`, `builtin_to_llvm`) from hand-written `isinstance` dispatch to `@lowering_for` handlers.

**Tech Stack:** Python, pytest, ruff, jj

**Key design constraints for pass migration:**
- Handlers call `replace_uses(old, new)` — operands of downstream ops are updated automatically, eliminating manual value maps.
- Side-effecting ops (ForOps that fill allocs, PrintOps, DeallocOps) must be chained to the replacement value via `ChainOp` so they're reachable from `block.result`.
- Handlers may use direct field mutation (`op.value = x`) for updating a single op without a global replace.
- Per-handler state (alloc metadata, loop counters) lives on `self` — the Pass is a class, handlers are methods.

---

### Task 1: Fix function-level imports

CLAUDE.md: "No function-level imports. The only acceptable exception is breaking a genuine circular dependency." There is no cycle between `block.py` → `graph.py` or `pass_.py` → `module.py`.

**Files:**
- Modify: `dgen/block.py`
- Modify: `dgen/passes/pass_.py`

**Step 1: Move imports to top level**

In `dgen/block.py`, add top-level import and remove the function-level import:

```python
# At top of file, after existing imports:
from dgen.graph import walk_ops

# In the ops property, remove:
#     from dgen.graph import walk_ops
```

In `dgen/passes/pass_.py`, add top-level import and remove the two function-level imports:

```python
# At top of file, after existing imports:
from dgen.module import Module, _walk_all_ops

# In verify_preconditions, remove:
#     from dgen.module import _walk_all_ops

# In verify_postconditions, remove:
#     from dgen.module import _walk_all_ops
```

Note: `_walk_all_ops` is already imported from `dgen.module`; `Module` is already imported. Just add `_walk_all_ops` to the existing import.

**Step 2: Run tests**

Run: `python -m pytest . -q`
Expected: all 440 tests pass, no behavior change

**Step 3: Commit**

```
jj commit -m "move function-level imports to top level in block.py and pass_.py"
```

---

### Task 2: Simplify `_run_block` — clear `_stored_ops` after pass

The current `_run_block` snapshots the ops list, dispatches handlers, tracks which ops were "replaced", then manually builds a new ops list by filtering + substituting. This is list-based logic that contradicts the graph model.

After handlers call `replace_uses`, the graph is already correct — old ops are unreachable, new ops are reachable from `block.result`. Clearing `_stored_ops` lets `block.ops` re-derive from the graph via `walk_ops`, giving implicit DCE.

**Prerequisite:** Test IR must use chains so side-effecting ops (print) are reachable from `block.result`. Without chains, `return(())` has no op dependencies and `walk_ops` would lose everything.

**Files:**
- Modify: `dgen/passes/pass_.py`
- Modify: `test/test_pass.py`

**Step 1: Update test IR in `test/test_pass.py` to use chains**

Every test that creates IR with `return(())` alongside side-effecting ops needs the return to reference the last side effect instead.

`test_rewriter_eager_replace`: No side effects, but ops must be reachable. Change `return(())` to `return(%2)` so `walk_ops` finds the add and its deps. Note: the return type stays `Nil` but value is the add op — this is valid since `ReturnOp.value` accepts any `Value`.

Input IR:
```
import llvm

%main : Nil = function<Nil>() ():
    %0 : Index = 1
    %1 : Index = 2
    %2 : Index = llvm.add(%0, %0)
    %_ : Nil = return(%2)
```

Expected output after replace_uses(%0, %1):
```
import llvm

%main : Nil = function<Nil>() ():
    %0 : Index = 1
    %1 : Index = 2
    %2 : Index = llvm.add(%1, %1)
    %_ : Nil = return(%2)
```

Note: `%0` (value 1) is still in the output because it's reachable from `return → add → %0 and %1`. After `replace_uses(%0, %1)`, the add references `%1` for both operands, but `%0` is no longer referenced by anything reachable from return. Wait — `return(%2)`, add references `%1, %1`. So `%0` is dead. But the test creates the Rewriter manually and doesn't clear `_stored_ops`. So this test still uses the stored ops list. The rewriter test doesn't go through `_run_block`, so it's unaffected by the _run_block changes. Keep the IR as-is but update the return for consistency.

Actually — this test creates a `Rewriter` directly and doesn't call `_run_block`. The block still has `_stored_ops` from the parser. So the formatted output includes all originally-parsed ops. The expected output should remain the same as the current test but with `return(%2)` in both input and output:

Input:
```
import llvm

%main : Nil = function<Nil>() ():
    %0 : Index = 1
    %1 : Index = 2
    %2 : Index = llvm.add(%0, %0)
    %_ : Nil = return(%2)
```

Expected:
```
import llvm

%main : Nil = function<Nil>() ():
    %0 : Index = 1
    %1 : Index = 2
    %2 : Index = llvm.add(%1, %1)
    %_ : Nil = return(%2)
```

`test_pass_run_eliminates_double_transpose`: Change return to reference print. After the pass clears `_stored_ops`, dead ops (%1 single transpose, %2 double transpose) are gone.

Input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
    %2 : toy.Tensor<[2, 3], F64> = toy.transpose(%1)
    %3 : Nil = toy.print(%2)
    %_ : Nil = return(%3)
```

Expected after pass:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

`test_pass_unregistered_ops_error`: Change return to reference print.

Input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

Behavior unchanged — still raises TypeError.

`test_pass_multiple_handlers_first_wins`: Change return to reference the constant so the handler fires.

Input:
```
%main : Nil = function<Nil>() ():
    %0 : Index = 42
    %_ : Nil = return(%0)
```

`test_pass_manager_verification_catches_range_violation`: Change return to reference print.

Input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
    %2 : Nil = toy.print(%1)
    %_ : Nil = return(%2)
```

**Step 2: Simplify `_run_block` in `dgen/passes/pass_.py`**

Remove the `replaced` set, the `_op_replacements` check, and the ops list splicing. After all handlers have run, clear `_stored_ops` so subsequent `block.ops` calls derive from the graph.

```python
def _run_block(self, block: dgen.Block) -> None:
    rewriter = Rewriter(block)
    for op in list(block.ops):  # snapshot — graph may change
        handlers = self._handlers.get(type(op), [])
        handled = False
        for handler in handlers:
            if handler(self, op, rewriter):
                handled = True
                break
        if not handled and not self.allow_unregistered_ops:
            raise TypeError(
                f"No handler for {type(op).__name__} in {type(self).__name__}"
            )
        if not handled:
            # Recurse into nested blocks for unhandled ops
            for _, child_block in op.blocks:
                self._run_block(child_block)
    # Graph takes over: derive ops from use-def walk
    block._stored_ops = None
```

**Step 3: Run tests**

Run: `python -m pytest test/test_pass.py -q`
Expected: all pass tests pass

Run: `python -m pytest . -q`
Expected: all 440 tests pass (optimize tests still use old `optimize()` wrapper with deepcopy + DCE, which constructs fresh blocks with `_stored_ops` set)

**Step 4: Commit**

```
jj commit -m "simplify _run_block to use graph-based ops derivation"
```

---

### Task 3: Remove `replace_op` from Rewriter

The design doc explicitly evaluates and rejects `replace_op` in favor of `replace_uses` only. With the graph-based `_run_block` from Task 2, `replace_uses` alone suffices — new ops are reachable via the graph, old ops are dead.

**Files:**
- Modify: `dgen/passes/pass_.py`
- Modify: `toy/passes/optimize.py`

**Step 1: Convert `optimize.py` handlers from `replace_op` to `replace_uses`**

In `fold_constants`:
```python
# Before:
rewriter.replace_op(op, new_op)

# After:
rewriter.replace_uses(op, new_op)
```

In `simplify_reshape` (reshape of reshape case):
```python
# Before:
rewriter.replace_op(op, new_op)

# After:
rewriter.replace_uses(op, new_op)
```

**Step 2: Remove `replace_op` and `_op_replacements` from Rewriter**

In `dgen/passes/pass_.py`, the Rewriter becomes:

```python
class Rewriter:
    """Manages in-place IR mutations during a pass."""

    def __init__(self, block: dgen.Block) -> None:
        self._block = block

    def replace_uses(self, old: dgen.Value, new: dgen.Value) -> None:
        """Eagerly replace all references to old with new in the block."""
        for op in self._block.ops:
            for name, operand in op.operands:
                if operand is old:
                    setattr(op, name, new)
            for name, param in op.parameters:
                if param is old:
                    setattr(op, name, new)
```

**Step 3: Run tests**

Run: `python -m pytest test/test_pass.py toy/test/test_optimize.py -q`
Expected: all pass

Run: `python -m pytest . -q`
Expected: all 440 tests pass

**Step 4: Commit**

```
jj commit -m "remove replace_op from Rewriter, use replace_uses only"
```

---

### Task 4: Remove `eliminate_dead_code` and `deepcopy` from `optimize.py`

With graph-based ops derivation in `_run_block` (Task 2), dead ops are implicitly excluded — they're unreachable from `block.result` so `walk_ops` doesn't find them. The explicit `eliminate_dead_code` function is redundant. Similarly, the design doc says "passes mutate the IR in place. No deepcopy."

**Prerequisite:** Test IR in `toy/test/test_optimize.py` must use chains so graph-based DCE works.

**Files:**
- Modify: `toy/passes/optimize.py`
- Modify: `toy/test/test_optimize.py`

**Step 1: Update test IR in `toy/test/test_optimize.py` to use chains**

Every test uses `return(())` alongside `toy.print`. Change return to reference the last side effect.

`test_transpose_elimination` input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
    %2 : toy.Tensor<[2, 3], F64> = toy.transpose(%1)
    %3 : Nil = toy.print(%2)
    %_ : Nil = return(%3)
```
Expected:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

`test_reshape_of_matching_constant` input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
    %2 : Nil = toy.print(%1)
    %_ : Nil = return(%2)
```
Expected:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

`test_constant_folding_reshape` input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[6], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
    %2 : Nil = toy.print(%1)
    %_ : Nil = return(%2)
```
Expected (original constant is dead — replaced by folded constant):
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

`test_dce` input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[2, 3], F64> = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    %2 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
    %3 : Nil = toy.print(%0)
    %_ : Nil = return(%3)
```
Expected (dead ops removed via graph walk):
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
    %_ : Nil = return(%1)
```

`test_full_pipeline` input:
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[2, 3], F64> = toy.reshape(%0)
    %2 : toy.Tensor<[6], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %3 : toy.Tensor<[2, 3], F64> = toy.reshape(%2)
    %4 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
    %5 : toy.Tensor<[3, 2], F64> = toy.transpose(%3)
    %6 : toy.Tensor<[3, 2], F64> = toy.mul(%4, %5)
    %7 : toy.Tensor<[3, 2], F64> = toy.transpose(%3)
    %8 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
    %9 : toy.Tensor<[3, 2], F64> = toy.mul(%7, %8)
    %10 : Nil = toy.print(%9)
    %_ : Nil = return(%10)
```
Expected (reshapes folded/eliminated, dead ops removed via graph walk):
```
import toy

%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %2 : toy.Tensor<[3, 2], F64> = toy.transpose(%1)
    %3 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
    %4 : toy.Tensor<[3, 2], F64> = toy.mul(%2, %3)
    %5 : Nil = toy.print(%4)
    %_ : Nil = return(%5)
```

**Step 2: Simplify `optimize()` function**

Remove `deepcopy`, `eliminate_dead_code`, and supporting functions (`collect_uses`, `_remove_indices`). The function becomes:

```python
def optimize(m: Module) -> Module:
    return ToyOptimize().run(m)
```

Remove unused imports: `Sequence`, `deepcopy`, `builtin` (if no longer used).

**Step 3: Run tests**

Run: `python -m pytest toy/test/test_optimize.py -v`
Expected: all 5 tests pass

Run: `python -m pytest . -q`
Expected: all tests pass. The end-to-end tests and CLI tests go through the full pipeline (lowering → optimize → shape inference → lower → codegen). These tests construct their IR via the parser/lowering which already threads chains, so they should be unaffected.

If any test fails with unexpected output, the expected output likely needs adjusting for graph-based op ordering (sequential numbering from 0, no gaps).

**Step 4: Commit**

```
jj commit -m "remove eliminate_dead_code and deepcopy from optimize — graph model handles DCE"
```

---

### Task 5: Lint, format, type-check (phase 1 checkpoint)

**Step 1: Format and lint**

Run: `ruff format`
Run: `ruff check --fix`
Run: `ty check`

Fix any issues.

**Step 2: Run full test suite**

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 3: Commit (if formatting changes)**

```
jj commit -m "fix formatting from ruff"
```

---

## Phase 2: Port Remaining Passes

All four remaining passes use the same hand-written pattern: iterate `f.body.ops`, `isinstance` dispatch per op type, yield replacement ops into a new list, maintain a manual value map, construct a new `FunctionOp` with `Block(ops=...)`. Porting each to the Pass infrastructure means: replace the `isinstance` ladder with `@lowering_for` handlers, replace the manual value map with `replace_uses`, and let `_run_block` handle the walk + graph derivation.

**Ordering rationale:** shape_inference is simplest (analysis pass, no op replacement). toy_to_affine and affine_to_llvm are lowering passes that need chain ergonomics. builtin_to_llvm has control flow lowering (IfOp → labels/phi) which is the most complex.

---

### Task 6: Port `shape_inference.py` to Pass infrastructure

Shape inference is an **analysis pass** — it mutates op types in place, no op replacement. This is the simplest pass to port because handlers don't call the rewriter at all; they just set `op.type` and return True.

**Files:**
- Modify: `toy/passes/shape_inference.py`
- Test: `toy/test/test_shape_inference.py` (existing tests, should pass without changes)

**Step 1: Create `ShapeInference(Pass)` class**

```python
class ShapeInference(Pass):
    allow_unregistered_ops = True  # skip ops we don't infer

    def __init__(self) -> None:
        self.type_of: dict[int, toy.Tensor] = {}
        self.func_map: dict[str, builtin.FunctionOp] = {}

    def run(self, module: Module) -> Module:
        """Override run to build func_map and seed block arg types."""
        self.func_map = {
            f.name: f for f in module.functions if f.name is not None
        }
        self.type_of = {}
        # Process main first (shapes derivable from constants)
        main = self.func_map.get("main")
        if main is not None:
            self._seed_and_run(main)
        # Process remaining functions
        for func in module.functions:
            if func.name != "main":
                self._seed_and_run(func)
        return module

    def _seed_and_run(self, func: builtin.FunctionOp) -> None:
        for arg in func.body.args:
            if isinstance(arg.type, toy.Tensor):
                self.type_of[id(arg)] = arg.type
        self._run_block(func.body)
```

**Step 2: Add `@lowering_for` handlers for each inference rule**

Each `isinstance` branch in the current `_infer_function` becomes a handler:

- `@lowering_for(ConstantOp)` — record type if Tensor
- `@lowering_for(toy.ReshapeOp)` — record type if Tensor
- `@lowering_for(toy.TransposeOp)` — infer reversed shape from input
- `@lowering_for(toy.MulOp)`, `@lowering_for(toy.AddOp)` — infer shape from lhs
- `@lowering_for(toy.ConcatOp)` — infer concatenated shape
- `@lowering_for(toy.TileOp)` — infer tiled shape
- `@lowering_for(builtin.CallOp)` — infer callee shapes, update return type

All handlers follow the pattern:
```python
@lowering_for(toy.TransposeOp)
def infer_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
    src = self.type_of.get(id(op.input))
    if src is not None:
        t = toy.Tensor(shape=shape_constant(list(reversed(src.unpack_shape()))))
        op.type = t
        self.type_of[id(op)] = t
    return True
```

Handlers return True unconditionally (even when no inference occurs) — this prevents the "unregistered op" error and tells the framework not to recurse into blocks (the pass handles recursion explicitly via CallOp).

**Step 3: Remove `_infer_function` and old `infer_shapes`**

Replace the module-level `infer_shapes` function:

```python
def infer_shapes(m: Module) -> Module:
    return ShapeInference().run(m)
```

Remove `_infer_function` and `_resolve_index_value` (move to method if needed). Remove `deepcopy` import — passes mutate in place.

**Step 4: Run tests**

Run: `python -m pytest toy/test/test_shape_inference.py -v`
Expected: all tests pass unchanged

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 5: Commit**

```
jj commit -m "port shape_inference to Pass infrastructure"
```

---

### Task 7: Port `toy_to_affine.py` to Pass infrastructure

This is a full lowering pass: every toy op is replaced with affine ops. The key challenge is that lowered ops include **side-effecting ForOps** (which fill allocated memory) that must be chained to the replacement value so they're reachable from `block.result`.

**Pattern for handlers that create side-effecting ops:**
```python
@lowering_for(toy.TransposeOp)
def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
    alloc_op = self._make_alloc(op.type.shape)
    # op.input is already the lowered value (replace_uses updated it)
    for_op = ...  # build nested for that fills alloc_op from op.input
    # Chain: for_op is side effect, alloc_op is the result
    result = builtin.ChainOp(lhs=alloc_op, rhs=for_op, type=alloc_op.type)
    rewriter.replace_uses(op, result)
    return True
```

**Files:**
- Modify: `toy/passes/toy_to_affine.py`
- Test: `toy/test/test_end_to_end.py`, `toy/test/test_shape_inference.py` (existing pipeline tests)

**Step 1: Create `ToyToAffine(Pass)` class**

```python
class ToyToAffine(Pass):
    allow_unregistered_ops = True  # ConstantOp, ReturnOp, ChainOp pass through

    def __init__(self) -> None:
        self.live_allocs: list[dgen.Value] = []
```

Note: `allow_unregistered_ops = True` means builtin ops (ConstantOp, ReturnOp, ChainOp, AddIndexOp) that appear in both domain and range pass through without handlers. Their operands are already correct from prior `replace_uses` calls. Switching to `False` with explicit pass-through handlers is a future refinement.

**Step 2: Add `@lowering_for` handlers**

Port each `isinstance` branch in the current `lower_op` to a handler. Key handlers:

- `@lowering_for(ConstantOp)` — for array constants, create new ConstantOp, track in `live_allocs`, `replace_uses`. For non-array constants, return False (pass through).
- `@lowering_for(toy.TransposeOp)` — create AllocOp + nested ForOps, chain the ForOp to the AllocOp, `replace_uses`.
- `@lowering_for(toy.MulOp)`, `@lowering_for(toy.AddOp)` — same pattern (alloc + nested for with binary op).
- `@lowering_for(toy.ReshapeOp)` — alias: `replace_uses(op, op.input)` (reshape is a no-op at the memory level).
- `@lowering_for(toy.PrintOp)` — create PrintMemrefOp, `replace_uses`.
- `@lowering_for(toy.DimSizeOp)` — create ConstantOp with the resolved dimension, `replace_uses`.
- `@lowering_for(toy.TileOp)` — alloc + nested for with offset arithmetic, chain, `replace_uses`.
- `@lowering_for(toy.ConcatOp)` — alloc + two nested fors (lhs copy, rhs copy with offset), chain both, `replace_uses`.
- `@lowering_for(builtin.ReturnOp)` — chain deallocs for all `live_allocs` to the return value via direct mutation (`op.value = chained_value`).

**Value map elimination:** The current pass uses `self.alloc_map` to remap operands. With `replace_uses`, this is automatic — when a handler replaces an op, all downstream ops' operands are updated in place. Handlers access the already-updated `op.input` (or `op.lhs`, `op.rhs`) directly.

The `alloc_map` entries that serve as metadata (tracking which allocs need deallocs) become `self.live_allocs`. The `alloc_map` entries for operand remapping are eliminated entirely.

**Step 3: Replace `lower_to_affine` entry point**

```python
def lower_to_affine(m: Module) -> Module:
    return ToyToAffine().run(m)
```

Remove the old `ToyToAffineLowering` class.

**Step 4: Run tests**

Run: `python -m pytest toy/test/ -q`
Expected: all tests pass

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 5: Commit**

```
jj commit -m "port toy_to_affine to Pass infrastructure"
```

---

### Task 8: Port `affine_to_llvm.py` to Pass infrastructure

Similar structure to Task 7 but lowers affine ops to LLVM ops. Additional complexity: ForOp lowering emits LLVM labels, branches, and phi nodes (unstructured control flow). Also tracks alloc metadata (`alloc_shapes`, `alloc_sizes`) for linearizing multi-dimensional indices and for `print_memref`'s size argument.

**Files:**
- Modify: `toy/passes/affine_to_llvm.py`
- Test: `toy/test/test_end_to_end.py` (existing pipeline tests)

**Step 1: Create `AffineToLLVM(Pass)` class**

```python
class AffineToLLVM(Pass):
    allow_unregistered_ops = True  # ConstantOp, ChainOp, etc. pass through

    def __init__(self) -> None:
        self.loop_counter = 0
        self.alloc_shapes: dict[int, list[int]] = {}  # id(value) -> shape
        self.alloc_sizes: dict[int, int] = {}  # id(value) -> total size
```

Note: metadata is keyed on `id(value)` rather than the value itself, since values are mutable and may be updated by `replace_uses`. When a handler creates a new op and wants to propagate metadata, it registers metadata under `id(new_op)`.

**Step 2: Add `@lowering_for` handlers**

Key handlers:

- `@lowering_for(affine.AllocOp)` — create `llvm.AllocaOp`, register metadata, `replace_uses`.
- `@lowering_for(affine.DeallocOp)` — no-op (stack alloc, no free). Return True to consume.
- `@lowering_for(affine.LoadOp)` — linearize indices, GEP + load, `replace_uses`.
- `@lowering_for(affine.StoreOp)` — linearize indices, GEP + store. StoreOp is a side effect — chain it.
- `@lowering_for(affine.ForOp)` — emit LLVM loop pattern (init, header label, phi, cmp, cond_br, body label, body ops, increment, br back, exit label). This is the most complex handler. The ForOp's nested body ops are lowered recursively by calling handler methods or `_run_block`. Chain the entire loop structure.
- `@lowering_for(affine.MulFOp)` — create `llvm.FmulOp`, `replace_uses`.
- `@lowering_for(affine.AddFOp)` — create `llvm.FaddOp`, `replace_uses`.
- `@lowering_for(affine.PrintMemrefOp)` — look up `alloc_sizes` for the input, create PackOp + CallOp, `replace_uses`.
- `@lowering_for(ConstantOp)` — register shape metadata if applicable, return True (op passes through).

**ForOp lowering challenge:** The current `_lower_for` emits a sequence of LLVM ops (labels, branches, phi nodes) into a flat list. These are inherently sequential/side-effecting. In the graph model, they need to be chained. The labels/branches create a control flow graph that isn't captured by the data-flow use-def graph. This is the "label problem" from the design doc.

Pragmatic approach: the ForOp handler emits the complete loop structure, chains it to the result via ChainOp, and calls `replace_uses`. The labels/branches/phi form an internal chain within the loop that's connected to the replacement value. The ForOp itself is consumed by the handler.

**Step 3: Replace `lower_to_llvm` entry point**

```python
def lower_to_llvm(m: Module) -> Module:
    return AffineToLLVM().run(m)
```

Remove the old `AffineToLLVMLowering` class.

**Step 4: Run tests**

Run: `python -m pytest toy/test/test_end_to_end.py -v`
Expected: all end-to-end tests pass

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 5: Commit**

```
jj commit -m "port affine_to_llvm to Pass infrastructure"
```

---

### Task 9: Port `builtin_to_llvm.py` to Pass infrastructure

Lowers builtin ops (AddIndexOp, SubtractIndexOp, EqualIndexOp, IfOp, CallOp) to LLVM ops. IfOp lowering is the complex part — it creates labels, branches, phi nodes, and processes nested blocks (then/else bodies).

**Files:**
- Modify: `dgen/passes/builtin_to_llvm.py`
- Test: `test/test_peano.py` (existing tests that exercise IfOp lowering)

**Step 1: Create `BuiltinToLLVM(Pass)` class**

```python
class BuiltinToLLVM(Pass):
    allow_unregistered_ops = True  # LLVM ops, ConstantOp, PackOp pass through

    def __init__(self) -> None:
        self.if_counter = 0
        self.current_label = "entry"
```

**Step 2: Add `@lowering_for` handlers**

Key handlers:

- `@lowering_for(builtin.AddIndexOp)` — create `llvm.AddOp`, `replace_uses`.
- `@lowering_for(builtin.SubtractIndexOp)` — create `llvm.SubOp`, `replace_uses`.
- `@lowering_for(builtin.EqualIndexOp)` — create `llvm.IcmpOp` + `llvm.ZextOp`, `replace_uses`.
- `@lowering_for(builtin.IfOp)` — the most complex handler. Creates cond_br, then/else labels, processes nested blocks, creates merge label + phi. The result is the phi op. Chain the entire control flow structure. Call `replace_uses(if_op, phi_op_or_chain)`.
- `@lowering_for(builtin.CallOp)` — create PackOp + `llvm.CallOp`, `replace_uses`.
- `@lowering_for(builtin.ReturnOp)` — return True (pass through; operands already updated by `replace_uses`).

**IfOp lowering challenge:** The handler needs to lower ops inside `then_body` and `else_body`. Since the handler "owns the op entirely, including its nested blocks" (per design doc), it processes the bodies itself rather than relying on framework recursion. The handler iterates each body's ops, calls the appropriate lowering for each, and collects the results.

**Step 3: Replace `lower_builtin_to_llvm` entry point**

```python
def lower_builtin_to_llvm(m: Module) -> Module:
    return BuiltinToLLVM().run(m)
```

Remove the old `BuiltinToLLVMLowering` class.

**Step 4: Run tests**

Run: `python -m pytest test/test_peano.py -v`
Expected: all Peano tests pass

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 5: Commit**

```
jj commit -m "port builtin_to_llvm to Pass infrastructure"
```

---

### Task 10: Final validation + lint

**Step 1: Format and lint**

Run: `ruff format`
Run: `ruff check --fix`
Run: `ty check`

Fix any issues.

**Step 2: Run full test suite**

Run: `python -m pytest . -q`
Expected: all tests pass

**Step 3: Verify no old pass patterns remain**

Check that the old pass classes are removed:
- `ToyToAffineLowering` class in `toy_to_affine.py` — removed
- `AffineToLLVMLowering` class in `affine_to_llvm.py` — removed
- `BuiltinToLLVMLowering` class in `builtin_to_llvm.py` — removed
- `_infer_function` in `shape_inference.py` — removed
- `eliminate_dead_code` in `optimize.py` — removed

Check that no manual value maps remain in the ported passes (no `self.value_map`, no `self.alloc_map`).

**Step 4: Commit (if formatting changes)**

```
jj commit -m "final lint and validation for pass infrastructure migration"
```
