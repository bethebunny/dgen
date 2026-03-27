# Codegen: Three-Phase LLVM IR Emission

The codegen (`dgen/codegen.py`) emits LLVM IR text from dgen's graph-based IR.
The core function is `_emit_func`, which transforms a `FunctionOp` into LLVM IR
basic blocks via three phases.

## Overview

dgen's IR is a use-def graph where execution order is determined by data
dependencies. LLVM IR is a flat list of basic blocks with explicit control flow.
The codegen bridges these models in three phases:

1. **Separate** — split mixed blocks (labels + non-label ops) into pure groups
2. **Linearize** — flatten the nested label tree into a flat list of `LinearBlock`s
3. **Emit** — walk the list, emit LLVM IR lines (phis, ops, terminators)

## Phase 1: Separate

A dgen block can contain both label ops and non-label ops. LLVM basic blocks
cannot — each must be pure. `_separate(block)` groups the block's ops by which
label ops they transitively depend on (via `_label_deps`):

- **No-dependency group**: ops that don't depend on any label → emitted first
- **Single-label groups**: ops depending on one label → emitted after that label
- **Multi-label groups**: ops depending on 2+ labels → emitted last

Each group becomes a synthetic `LabelOp` (named `_blk0`, `_blk1`, ...) that the
linearizer treats like a real label block.

## Phase 2: Linearize

`_linearize(block, name)` recursively flattens the separated groups into a list
of `LinearBlock(name, ops, label)`:

- Real labels recurse into their body blocks
- Synthetic labels carry their ops directly
- `%exit` parameters generate empty blocks (fall-through after loop headers)

### IfOp expansion

`control_flow.IfOp` is structured control flow — it doesn't need goto labels.
`_linearize_ops` expands it inline during linearization:

```
[ops_before, IfOp, ops_after]

→ block_N:   ops_before; icmp + br i1 %cond, %then, %else
  then_N:    then_body ops; br %merge
  else_N:    else_body ops; br %merge
  merge_N:   phi [then_result, %then], [else_result, %else]; ops_after
```

The IfOp's result value is aliased to the phi in the merge block.

## Phase 3: Emit

Walk the `LinearBlock` list and emit LLVM IR:

- **Phi nodes**: for real labels, build from the predecessor map (which labels
  branch to this one, with what values)
- **If/else phis**: for merge blocks, emit from `if_phis` state
- **Ops**: `_emit_op` dispatches on op type to produce LLVM IR instructions.
  Unrecognized ops raise `ValueError`.
- **Terminators**: explicit branches (`goto.BranchOp`, `goto.ConditionalBranchOp`),
  if/else merge branches, fall-through `br label %next`, or `ret`.

## Predecessor map

Built after linearization from the flat block list. For each branch op, records
(source_block, argument_values) → target_label. Also adds fall-through
predecessors when a block without a terminator precedes a label block.

## Current architecture

All three phases plus their shared state live inside `_emit_func`, a ~500-line
closure. The function-local state includes: `tracker` (SSA numbering),
`constants`/`types` (value → LLVM representation), `predecessors` (phi
construction), and `if_blocks`/`if_phis`/`if_merge_targets` (IfOp expansion).

`_emit_op` dispatches on op type via an isinstance chain to produce LLVM IR
instructions. Unrecognized ops raise `ValueError`.

The `%exit` block convention relies on `param.name.startswith("exit")` — a
naming contract between `control_flow_to_goto.py` and the codegen.
