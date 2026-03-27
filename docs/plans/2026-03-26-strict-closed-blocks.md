# Strict closed blocks: no ambient ops

## Problem

The current closed-block invariant allows "ambient" ops: ops defined in a parent
scope that are reachable from a child block's result via `walk_ops`. This means
an op can appear in multiple blocks' `block.ops`, creating ambiguity about which
LLVM basic block it belongs to. The codegen needs complex scheduling logic
(claimed sets, depth computation) to resolve this.

## Design

Remove the ambient op carveout. Every value referenced by a block that isn't
defined within that block must be an explicit capture. This means:

- `block.ops` (from `walk_ops(result, stop=captures)`) contains only ops
  defined locally — all external paths go through captures.
- Each op belongs to exactly one block. No scheduling needed.
- The codegen emits each block's ops in topological order. Period.

### What changes

**Verifier** (`dgen/verify.py`): The valid set is already
`parameters + args + captures + block.ops`. No verifier change needed — the
invariant is the same. The change is that passes must provide correct captures
so that `block.ops` doesn't reach through to parent-scope ops.

**`_compute_captures`** (`toy/passes/control_flow_to_goto.py`): Already
collects operand and parameter references. Needs to also collect ops reachable
from the block result that are defined outside the block. In practice, this
means: walk the block result, and for any op encountered that isn't in the
block's locally-defined set, add it as a capture.

Actually — the simpler formulation: compute captures from `walk_ops(result)`
WITHOUT the stop set. Any op in that walk that isn't a local definition
(block arg, block parameter, or an op whose operands are all in-scope) must
be captured. But this is circular.

The practical approach: `_compute_captures` already works by scanning operands
and parameters of each op in `block.ops`. The issue is that `block.ops` itself
is computed with the stop set, so ambient ops already appear in it. If we add
them as captures, the next `block.ops` computation would stop at them, and
they'd no longer appear as ambient ops. This is the fix: after constructing
a block, compute its captures, set them, and `block.ops` becomes clean.

**`ControlFlowToGoto`**: Already calls `_compute_captures` after construction.
The captures function needs to be complete (catch all external refs).

**`StructuredToLLVM`**: Manually constructs capture lists. These need to be
complete. This pass is being replaced, so this is low priority.

**`BuiltinToLLVMLowering`**: Constructs label bodies for IfOp. Needs capture
computation for then/else/merge labels.

**Codegen** (`dgen/codegen.py`): Replace the `_collect_labels` + `claimed` +
`op_to_label` machinery with the simple `_walk_all_blocks` recursive walk.
Each block's ops are disjoint (no overlap), so no dedup needed.

**`walk_ops`** (`dgen/graph.py`): No change. Already stops at captures. With
correct captures, `block.ops` naturally excludes parent-scope ops.

### Key insight

The captures mechanism already exists and works. The "ambient op" issue is just
that some passes don't set captures completely. Fixing capture computation in
the passes makes the codegen trivially simple.

## Implementation

1. Write `walk_ops` tests and docs (clarify the contract)
2. Make `_compute_captures` complete: check all ops, operands, parameters,
   and child block captures — and add any value not locally defined
3. Update `ControlFlowToGoto` to use it (already mostly done)
4. Update `BuiltinToLLVMLowering` to compute captures for IfOp labels
5. Replace codegen with simple `_walk_all_blocks` (no claimed set)
6. Run full test suite
7. Update CLAUDE.md block documentation

## Testing

- `test_nested_loop_codegen`: the minimal reproducer should pass with
  correct LLVM IR (no ops after terminators, correct phi predecessors)
- All existing tests should pass
