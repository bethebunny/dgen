# Control Flow Representation

## Design

dgen uses **closed blocks with MLIR-style block arguments**, not a flat basic-block model.
Within a single block scope, the use-def graph is the execution model: unordered ops may
execute in any order, and side effects are made explicit via chains.

### Closed Blocks

Blocks are closed: an op inside a block may only reference values defined within that block
(local ops, block arguments, or the block's own result). Any value that must flow from an
enclosing scope into a block must be threaded explicitly as a block argument. This is the
same invariant as MLIR.

```
# Correct: outer value threaded in as block argument
%body = llvm.label() (%i: Index, %ptr: Ptr):
    %val = llvm.load(%ptr)
    %next = llvm.add(%i, %one)
    %_ = llvm.br(%header, [%next, %ptr])

# Wrong: inner block captures outer value directly
%body = llvm.label() (%i: Index):
    %val = llvm.load(%ptr)   ← %ptr is from enclosing scope — invalid
```

Block arguments at each block boundary serve the same role as MLIR block arguments and phi
nodes. Codegen reconstructs phi instructions by scanning predecessor branches.

### Within-Block: Sea-of-Nodes Semantics

Within a single block, the IR is a **Sea-of-Nodes** graph: the use-def graph is the
execution model. There is no implicit ordering between ops that are not connected by
use-def or chain edges.

**Pure ops** (no side effects) may execute in any order with respect to each other, as long
as data dependencies are satisfied. Optimizations may freely reorder, hoist, sink, or
eliminate them.

**Side-effecting ops** must be connected to the block's use-def graph via `ChainOp` to be
reachable from the block result. Chain edges encode both liveness (the op will execute) and
ordering (the chain spine defines the sequence). See `passes.md` for the chain convention.

**All ops in a block must be reachable from the block's `result` via `walk_ops`.** An op
not reachable from the result is dead and will not execute. The block's `result` is the
use-def root; `walk_ops` on `result` gives the canonical op list.

### What `ChainOp` Is

A `ChainOp(lhs=side_effect, rhs=next)` encodes a **control edge disguised as a data edge**:
`lhs` must execute before `next`, and `lhs` must not be eliminated. Chains are the only
ordering guarantee within a block. Two side-effecting ops that are not connected on the
same chain spine have no guaranteed execution order relative to each other.

```
# Store executes before return; both are live
%_ = chain(%store_op, return(()))
```

The chain spine is the schedule for a block's side effects.

---

## Transitional Exceptions

Two kinds of values may currently cross block boundaries without being threaded as block
arguments. Both are transitional: they will be resolved via a symbol/forward-declare
mechanism in a future design.

### Label Ops

`llvm.LabelOp` values (branch targets) may be referenced from any block. A branch
inside a loop body may target a label defined in an outer scope:

```
%header = llvm.label() (%i: Index):
    ...
    %body = llvm.label() (%j: Index):
        %_ = llvm.br(%header, [%next])   ← %header is from enclosing scope
```

This is a structural reference (branch target), not a data dependency. It is permitted
because label ops define control-flow structure, not computed values. Once a symbol/
forward-declare design is in place, branch targets will be referenced by symbol rather
than by value, removing this exception.

### Function References

`builtin.FunctionOp` values used as call targets may similarly cross block boundaries.
Same rationale as labels: structural reference, not a data dependency. Same eventual
resolution via symbols.

---

## Possible Extension: Constants

Values that are pure and have no side effects could in principle be referenced across
block boundaries without threading — equivalent to duplicating the constant at the use
site. This is not part of the current design. If adopted, it would require a clear
definition of "constant" (likely: `ConstantOp` only, not arbitrary pure ops) and would
need to be explicitly represented in the parser's scoping rules.

---

## Relationship to Sea-of-Nodes

The within-block execution model is Sea-of-Nodes. The between-block structure is
MLIR-style closed blocks with explicit block arguments. This is not a migration toward a
fully Sea-of-Nodes IR (no block containers, explicit `ctrl` fields, floating ops across
regions). Full SoN would require:

- Side-effecting ops carrying explicit control inputs
- Regions derived from control edges rather than containers
- A scheduler as the exit pass

These are not planned. The current hybrid — closed MLIR blocks containing SoN use-def
graphs — is the intended permanent design. It preserves MLIR's structural clarity at the
block level while enabling within-block optimization freedom.

---

## Block Argument Conventions

Block arguments follow the same conventions as MLIR:

- A block's arguments are in scope for all ops in the block.
- Branch ops pass values to target block arguments positionally:
  `llvm.br(%target, [%val0, %val1])` passes `%val0` → first arg of `%target`'s body,
  `%val1` → second arg.
- Codegen emits phi nodes by scanning all branches targeting a given label and collecting
  the (predecessor, value) pairs for each block argument position.
- The entry block of a function has the function's parameters as its block arguments.

---

## Symbols (Future Work)

The two current exceptions (labels, function refs) both share the same structure: they
are named entities referenced by multiple users, potentially before definition. The
correct abstraction is a **symbol**: a named, globally-addressable value that can be
referenced before it is defined, without creating a use-def edge.

A symbol design would introduce `forward_declare`/`link` or similar, replacing direct
value references to labels and functions with symbol lookups. This resolves both
exceptions cleanly and also enables mutual recursion and irreducible CFGs.
