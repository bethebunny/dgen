# Codegen simplification: topological emission

## Problem

The codegen's `_emit_func` uses a complex block-membership model:
- `_collect_labels` recursively walks label body blocks
- `claimed` set deduplicates ops by first-walk order
- `op_to_label` maps ops to labels based on body block membership
- `resolve_target` handles `%self` → label indirection

This breaks for the flat exit continuation design: ops that follow a nested
loop in the outer body block get claimed by the outer body, but LLVM needs
them after the inner exit label (which terminates the outer body block).

## Design

Replace block-membership partitioning with topological emission. The key
insight: `walk_ops` already returns ops in topological order (dependencies
before dependents). Labels are ops. If we emit ops in walk_ops order,
inserting LLVM label headers when we encounter a LabelOp and phi nodes
when we encounter a label's block args, the output is naturally correct.

### Emission rules

Walk `f.body.ops` (topological order). For each op:

1. **LabelOp**: emit `{name}:` (LLVM label header). Then emit phi nodes
   for the label's block args, derived from predecessor branches.
2. **BranchOp / ConditionalBranchOp**: emit `br` (LLVM terminator).
3. **ChainOp / ConstantOp / PackOp**: skip (sugar ops, not emitted).
4. **Everything else**: emit as LLVM instruction in the current block.

The "current block" is implicitly defined by the last label header emitted.
Ops before any label are in the entry block. LLVM fall-through between
blocks is automatic.

### Predecessor map

For phi nodes, we still need to know which branches target each label.
Scan all ops for BranchOp/ConditionalBranchOp, record
`(target_label, source_label, args)`. The source label is the last
LabelOp that appeared before this branch in topological order — same
implicit current-block tracking.

### What changes

- Delete `_collect_labels`, `claimed` set, `label_body_ops`, `op_to_label`
- Delete `resolve_target` and the `label_of` parameter map
- Replace with a single walk emitting ops in order
- `_needs_ret`: check if the last real op in each block section is a branch

### What stays the same

- `SlotTracker` registration (still needs emission-order numbering)
- `emit_op` (individual op emission)
- `typed_ref`, `bare_ref`, constant/type registration
- `_branch_edges` helper

## Why this fixes nested loops

In topological order, the inner loop's labels and ops come between the
outer body's entry branch and the outer body's continuation. The emission
naturally places the continuation after the inner exit label:

```
loop_body0:           ← outer body label
  %j0 = phi ...
  br label %header1   ← inner loop entry (terminates body0)
loop_header1:          ← inner header
  ...
loop_body1:            ← inner body
  ...
loop_exit1:            ← inner exit (empty, fall-through)
  %next = add %j0, 1  ← outer continuation (after inner exit)
  br label %header0    ← outer back-edge
```
