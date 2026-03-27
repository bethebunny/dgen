# Control Flow Representation

## Design

dgen uses **closed blocks with explicit captures**, not a flat basic-block model.
Within a single block scope, the use-def graph is the execution model: unordered ops may
execute in any order, and side effects are made explicit via chains.

### Closed Blocks

Blocks are closed: an op inside a block may only reference values that are in scope:
local ops, block arguments, block parameters, or declared captures. Any value from an
enclosing scope must be declared in the block's `captures` list.

```
# Correct: outer value declared as a capture
%body = goto.label([]) body(%i: Index) captures(%ptr):
    %val = llvm.load(%ptr)
    %one = 1
    %next = algebra.add(%i, %one)

# Wrong: inner block references outer value without capturing it
%body = goto.label([]) body(%i: Index):
    %val = llvm.load(%ptr)   ← %ptr not in scope — invalid
```

### Three Kinds of Block Inputs

- **`args`** — runtime values that vary per entry. Branch ops pass values to target
  block args positionally. Codegen emits phi nodes for these.
- **`parameters`** — compile-time values bound at block construction. Used for structural
  references like `%self` (back-edges) and `%exit` (loop exit labels). Declared with
  `<name: Type>` syntax in ASM.
- **`captures`** — outer-scope values referenced directly. No phi, no copy — just a
  declared dependency. The block doesn't own the captured value; it's a boundary in
  `walk_ops`.

### Within-Block: Sea-of-Nodes Semantics

Within a single block, the IR is a **Sea-of-Nodes** graph: the use-def graph is the
execution model. There is no implicit ordering between ops that are not connected by
use-def or chain edges.

**Pure ops** (no side effects) may execute in any order with respect to each other, as long
as data dependencies are satisfied. Optimizations may freely reorder, hoist, sink, or
eliminate them.

**Side-effecting ops** must be connected to the block's use-def graph via `ChainOp` to be
reachable from the block result. Chain edges encode both liveness (the op will execute) and
ordering (the chain spine defines the sequence).

**All ops in a block must be reachable from the block's `result` via `walk_ops`.** An op
not reachable from the result is dead and will not execute. The block's `result` is the
use-def root; `walk_ops` on `result` gives the canonical op list.

### What `ChainOp` Is

A `ChainOp(lhs, rhs)` encodes a **control edge disguised as a data edge**: `rhs` must
execute and must not be eliminated, and the chain's result is `lhs`'s runtime value.
Dataflow dependencies are the only ordering guarantee within a block; `ChainOp` is how
side-effecting ops that produce no useful value are injected into the dataflow graph so
they participate in that ordering.

```
# %val is passed through; %store_op is kept live and must execute
%_ = chain(%val, %store_op)
```

The chain spine is the schedule for a block's side effects.

### What `walk_ops` Follows

`walk_ops(root, stop)` returns all ops that are transitive dependencies of the root,
in the same block. The dependency edges are: operands, parameters, types, block
captures, and block argument types. These are exactly the edges that connect a value
to the values it depends on within a single block scope.

It does NOT descend into nested block bodies. Each block is its own walk scope.
Captures are boundaries — the walk visits them as dependencies but doesn't traverse
past them into their own dependency subgraphs (they're in the stop set).

---

## Control Flow Dialects

### `goto` dialect — Unstructured control flow

Used for loops (which need back-edges). Labels, branches, conditional branches.

**Label-as-expression model**: A `goto.label` is not a jump target — it's an expression
block that runs when control reaches it in use-def order. No explicit entry branch is
needed. The label's `initial_arguments` provide first-iteration values for its block args.

```
%header = goto.label([0]) body<%self: Label, %exit: Label>(%iv: Index):
    %cmp = algebra.less_than(%iv, %limit)
    %body = goto.label([]) body(%jv: Index) captures(%self):
        %next = algebra.add(%jv, 1)
        goto.branch<%self>([%next])
    goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])
```

Key conventions:
- `%self` parameter enables back-edges (breaks use-def cycles)
- `%exit` parameter: codegen emits a fall-through label after the header block

### `control_flow` dialect — Structured control flow

Higher-level ops that lower to `goto` (loops) or are emitted directly by codegen (if/else).

- **`control_flow.for`** and **`control_flow.while`** — lowered to `goto.label` by
  `ControlFlowToGoto` pass. See `toy/passes/control_flow_to_goto.py` docstring.
- **`control_flow.if`** — NOT lowered to goto. Codegen expands it inline during
  linearization into `cond_br → then → else → merge(phi)`. This avoids the fundamental
  difficulty of representing merge labels and phi values in a label-as-expression model.

---

## Relationship to Sea-of-Nodes

The within-block execution model is Sea-of-Nodes. The between-block structure is
closed blocks with explicit arguments, parameters, and captures. This is not a migration
toward a fully Sea-of-Nodes IR (no block containers, explicit `ctrl` fields, floating ops
across regions).

The current hybrid — closed blocks containing SoN use-def graphs — is the intended
permanent design. It preserves structural clarity at the block level while enabling
within-block optimization freedom.

---

## Function References

Function references (`function.FunctionOp`) are currently referenced by value across
block boundaries. Blocks should explicitly capture function references they depend on,
just like any other outer-scope value.
