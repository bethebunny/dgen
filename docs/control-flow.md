# Control Flow Representation

## Problem

Adding body blocks to `llvm.LabelOp` — making labels into values with nested code — exposed a fundamental tension in the IR's representation of control flow. The implementation required:

1. A new `unwrap_chain` function in `graph.py` because `walk_ops` crosses label boundaries through branch operands, discovering ops that belong to other basic blocks.
2. `Block` accepting both `result=` and `ops=` simultaneously, because the data-flow root (`result`) and the schedule (`_stored_ops`) answer different questions about the same code.
3. Cycle-safe guards in `SlotTracker.register`, `op_asm`, and `_walk_all_ops` because labels reference each other through branch operands, creating reference cycles.
4. A two-phase lowering pattern: first emit ops linearly, then group them into label bodies — because the graph structure and the schedule diverge.

These are not incidental bugs. They are symptoms of a design conflict: **blocks conflate data flow (use-def reachability) with scheduling (containment and ordering)**.

## The Tension: Data Flow vs Scheduling

A `Block` has two faces:

```
Block:
  result: Value       ← "what is the data-flow root?"
  _stored_ops: [Op]   ← "what ops are scheduled here?"
```

For straight-line code these agree — the ops reachable from `result` via `walk_ops` are exactly the ops that should be scheduled in the block. But control flow breaks this correspondence:

**Data flow crosses block boundaries.** A `CondBrOp` references label values as operands. Following those operand edges in `walk_ops` pulls in the label and all its contents — ops that belong to other basic blocks.

```
entry block:
  %cmp = icmp ...
  %body = label() ():          ← label is a value
      %next = add(%i, %one)    ← belongs to body, not entry
      br(%header)              ← belongs to body, not entry
  %exit = label() (): ...
  cond_br(%cmp, %body, %exit)  ← operand edges reach into body and exit
```

Walking `cond_br`'s operands discovers `%body` and `%exit`, whose bodies contain ops like `%next` and `br`. From the use-def graph's perspective, these are all reachable from `cond_br`. From the scheduling perspective, they belong to different basic blocks.

**Scheduling doesn't follow data edges.** An op's "home block" is determined by which label it appears under — a positional relationship in the flat list, not a data dependency. A `LoadOp` inside a label body may only depend on a pointer from the entry block. There's no data edge from the label to the load. The load is "in" the body only because it was placed there by the lowering pass.

This is why `Block` needs `_stored_ops` — the graph structure alone cannot recover which ops belong where.

## What `ChainOp` Really Is

The chain mechanism (documented in `passes.md`) threads side-effecting ops into the use-def graph so they're reachable from the block's result:

```python
# ChainOp(lhs=side_effect, rhs=rest_of_chain)
chain = ChainOp(lhs=store_op, rhs=ChainOp(lhs=print_op, rhs=return_op))
```

This is described as a data edge, but it encodes **scheduling information**. The store has no data-flow relationship with the rest of the chain — it's linked in because it must be emitted. The chain preserves two things:

1. **Liveness** — the store is reachable from the result, so DCE won't remove it.
2. **Ordering** — the chain spine defines the emission order: store, then print, then return.

A chain edge is a **control edge disguised as a data edge**. It answers the scheduling question ("this op must run, in this order") using the data-flow mechanism ("this op is reachable from the result"). This works because the IR has only one edge type. Sea-of-Nodes (discussed below) makes the distinction explicit.

The `unwrap_chain` function exists precisely because the chain mixes concerns — it recovers the schedule (the linear sequence of ops in a block) from a structure that pretends to be data flow:

```python
def unwrap_chain(result: dgen.Value) -> list[dgen.Op]:
    """Walk only the chain spine, returning the ops that were
    chained together — without following transitive operands
    that may cross block boundaries."""
    from dgen.dialects.builtin import ChainOp

    ops: list[dgen.Op] = []
    current: dgen.Value = result
    while isinstance(current, ChainOp):
        if isinstance(current.lhs, dgen.Op):
            ops.append(current.lhs)
        current = current.rhs
    if isinstance(current, dgen.Op):
        ops.append(current)
    return ops
```

This is not a general graph walk. It's spine-only: follow `rhs`, collect `lhs`. It deliberately does *not* follow data dependencies, because those cross block boundaries. It recovers the *schedule* from the chain structure.

## Approaches Considered

### 1. Symbols

**Idea:** Labels become named references (symbols) rather than direct value references. Branches reference labels by name, not by use-def edge. `walk_ops` never follows a symbol reference, so it stays within block boundaries.

```
%header = forward_declare<"header">()
%body_impl = label { ...; br(%header) }    ← symbol reference, not use-def
link(%header, %body_impl)
```

**What it solves:** `walk_ops` no longer crosses block boundaries. Each block's ops are exactly the ops reachable from its result within its scope.

**What it doesn't solve:** The `_stored_ops` / `result` duality persists. Side-effecting ops still need chains to be reachable. The chain is still a control edge disguised as a data edge. Symbols add indirection without addressing the fundamental conflation.

**Verdict:** Useful for breaking reference cycles (mutual label references, mutual recursion), but orthogonal to the data-flow vs scheduling tension. The symbol design in `passes.md` stands on its own merits for the forward-reference problem.

### 2. Closed Blocks / Capture Lists

**Idea:** Blocks are banned from referencing values defined outside their scope. Any external value must be passed as a block argument (like a function parameter). This makes block membership = reachability: if you start from a block's result and walk the graph, you stay within the block, because external references are severed at the block boundary.

```
# Closed block: all inputs are explicit block arguments
%body = label(%i: Index, %ptr: Ptr) ():
    %val = load(%ptr)
    %next = add(%i, 1)
    br(%header, %next, %ptr)      ← passes values to target
```

**What it solves:** `walk_ops` from a block's result finds only ops within that block. No cross-boundary leakage. `_stored_ops` becomes redundant — the graph structure is the schedule.

**What it doesn't solve:** Verbose. Every value used across a block boundary must be threaded through block arguments. For a loop with many live variables, the capture list grows large. This is the MLIR model (block arguments replace phi nodes), and it works, but it's a significant ergonomic cost.

**Relationship to current design:** The current label body blocks are *open* — they freely reference values from outer scopes (the entry block's alloca, constants, etc.). Closing them would require a refactoring pass that inserts block arguments for every cross-scope reference.

### 3. Do We Need Blocks At All?

If blocks can't tell us which ops belong where (because the graph crosses boundaries), and the schedule is recovered from the chain spine (not the block structure), then what are blocks *for*?

Blocks serve three roles in the current IR:

1. **Scope for block arguments** — function parameters, loop induction variables. These are values that have no defining op; they're defined by the block itself.
2. **Container for the ASM formatter** — the formatter indents ops within a block. Without blocks, there's no indentation structure.
3. **Container for codegen** — the codegen emits ops grouped by their basic block. Without blocks, it needs another way to group ops.

Roles 2 and 3 are **scheduling** — they're about how to present and emit ops, not about what the ops compute. Role 1 (block arguments) is semantic, but it could be modeled differently.

This leads to the key question: **should the IR represent scheduling explicitly, or should scheduling be derived from the graph at emit time?**

## Recommendation: Sea-of-Nodes

### The Core Idea

Sea-of-Nodes (Click, 1993) answers the question directly: **the IR has two kinds of edges, and they are explicit.**

1. **Data edges** — use-def dependencies. An `add` depends on its two operands. These are the edges `walk_ops` follows.
2. **Control edges** — scheduling constraints. An op carries a control input that says "I am pinned to this region." These are the edges that determine which basic block an op "lives in."

Pure ops (add, mul, load from immutable memory) have *no* control input — they float freely in the graph. The scheduler can place them anywhere that respects their data dependencies. Side-effecting ops (store, call, branch) have a control input that pins them to a specific region.

```
                     ┌───────────────────────────────┐
 Data edges          │     Control edges              │
 (use-def):          │     (scheduling):              │
                     │                                │
 %a ──→ add ←── %b   │     entry ──→ store            │
         │           │       │                        │
         ▼           │       ▼                        │
       mul ←── %c    │     branch ──→ header          │
                     │       │         │              │
                     │       ▼         ▼              │
                     │     cond_br   phi              │
                     └───────────────────────────────┘
```

A **region** (the Sea-of-Nodes analogue of a basic block) is the set of ops whose control input traces to the same control node. It's not a container — it's derived from the graph. No `_stored_ops`. No explicit block object holding a list.

### How It Maps to dgen

The existing IR is *almost* Sea-of-Nodes already:

| Sea-of-Nodes | dgen today |
|---|---|
| Data edges | Operand references (use-def) |
| Control edges | `ChainOp` spine |
| Region | Block (via `_stored_ops`) |
| Scheduling | ASM formatter's topo sort, codegen's block iteration |
| Floating pure ops | Ops reachable via `walk_ops` |
| Pinned side-effect ops | Ops linked via `ChainOp` |

`ChainOp` is already a control edge — it pins side-effecting ops to a schedule. The ASM formatter already performs scheduling — it topologically sorts ops from a block's result to determine emission order. The codegen already iterates label bodies separately, emitting ops in the order determined by the block structure. The transformation to Sea-of-Nodes is not a revolution; it's making the existing implicit structure explicit.

### What Changes

**Control input on ops:** Side-effecting ops gain a `ctrl` field that points to their control region (a `RegionOp` or similar). Pure ops have no `ctrl` — they float.

```python
@dataclass(eq=False, kw_only=True)
class StoreOp(Op):
    ctrl: Value        # control input: which region this store lives in
    value: Value       # data input
    ptr: Value         # data input
    type: Type = Nil()
```

**Regions replace blocks:** A region is identified by a control node (analogous to a basic block label). Ops that reference this control node are "in" the region. No container object needed.

```python
# A region is just a control value that other ops point to
@dataclass(eq=False, kw_only=True)
class RegionOp(Op):
    """A control flow region (basic block header)."""
    type: Type = Control()   # produces a control token

# Branching produces control tokens for target regions
@dataclass(eq=False, kw_only=True)
class CondBranchOp(Op):
    cond: Value              # data input
    true_region: Value       # data input (a RegionOp)
    false_region: Value      # data input (a RegionOp)
    type: Type = Control()   # produces control for the next region
```

**`ChainOp` becomes unnecessary.** Currently ChainOp encodes "this op must be emitted, in this order." With explicit control edges, the control input serves both purposes — the op is reachable (pinned to a region) and ordered (the region's control sequence determines the schedule).

**`_stored_ops` becomes unnecessary.** The set of ops in a region is derived from the graph: all ops whose `ctrl` traces to a given `RegionOp`. No need for a separate list.

**`unwrap_chain` becomes unnecessary.** The chain existed to recover the schedule from a structure that mixed data and control. With separate edge types, the schedule is directly readable from control edges.

**`walk_ops` stays the same.** It follows data edges only. It never follows control edges (which are a different field). Cross-boundary leakage disappears because control edges are not data edges.

### Scheduling Is the Exit Pass

The formatter and codegen already perform scheduling — they take a graph and produce a linear sequence. In a Sea-of-Nodes IR, scheduling is explicitly the responsibility of the **exit pass**: the pass that converts the floating-op graph into a linear basic-block representation for emission.

The current codegen in `codegen.py` already does this:

```python
# Separate entry ops from label ops
entry_ops: list[dgen.Op] = []
label_ops: list[llvm.LabelOp] = []
for op in f.body.ops:
    if isinstance(op, llvm.LabelOp):
        label_ops.append(op)
    else:
        entry_ops.append(op)

# Emit entry block ops
for op in entry_ops:
    emit_op(op, lines)

# Emit each label's body
for label_op in label_ops:
    label_name = tracker.track_name(label_op)
    lines.append(f"{label_name}:")
    for body_op in label_op.body.ops:
        emit_op(body_op, lines)
```

This is a scheduler. It determines which ops go in which basic block and what order they're emitted. In a Sea-of-Nodes IR, this logic becomes the *only* place that cares about block structure. All optimization passes work on the graph — they never think about blocks or ordering. Only the final pass (codegen or ASM emission) linearizes.

The ASM formatter in `formatting.py` is also a scheduler — `SlotTracker.register` walks ops to assign stable names, and `op_asm` produces a linear text representation. Both are scheduling concerns that belong at the exit, not in the IR structure.

### ASM Round-Trip

A natural question: if regions are implicit (derived from control edges), how does the ASM format work? The same way it works today — the formatter *schedules* ops into a linear representation:

```
%entry = region():
%cmp = icmp<"slt">(%i, %n)
%body = region():
%exit = region():
%_ = cond_br(%cmp, %body, %exit)
%next = add [ctrl=%body] (%i, %one)
%_ = br [ctrl=%body] (%header)
%_ = ret [ctrl=%exit] (())
```

The `[ctrl=%body]` annotation is the control input — it tells the reader (and the parser) which region the op is pinned to. Pure ops (like `icmp`, `add` when not pinned) have no control annotation and float freely.

Alternatively, the formatter could group ops by region with indentation, exactly as it does today with label blocks:

```
%entry = region():
    %cmp = icmp<"slt">(%i, %n)
    %_ = cond_br(%cmp, %body, %exit)
%body = region():
    %next = add(%i, %one)
    %_ = br(%header)
%exit = region():
    %_ = ret(())
```

This is a display choice, not a semantic one. The underlying graph is the same either way. Both formats round-trip correctly because the parser builds the graph from operand references, not from indentation.

### Example: Loop

Current representation (with label body blocks):

```
%f : Nil = function<Nil>() ():
    %0 : Nil = llvm.alloca<3>()
    %init : Index = 0
    %_ : Nil = llvm.br(%loop_header)
    %loop_header : llvm.Label = llvm.label() ():
        %i0 : Nil = llvm.phi<%entry, %loop_body>(%init, %next)
        %hi : Index = 3
        %cmp : Nil = llvm.icmp<"slt">(%i0, %hi)
        %_ : Nil = llvm.cond_br(%cmp, %loop_body, %loop_exit)
    %loop_body : llvm.Label = llvm.label() ():
        %one : Index = 1
        %next : Nil = llvm.add(%i0, %one)
        %_ : Nil = llvm.br(%loop_header)
    %loop_exit : llvm.Label = llvm.label() ():
        %_ : Nil = return(())
```

Sea-of-Nodes representation (regions are control values, ops carry control inputs):

```
%f : Nil = function<Nil>() ():
    %entry : Control = region()
    %0 : Ptr = alloca<3>() [ctrl=%entry]
    %init : Index = 0
    %_ : Nil = br(%header) [ctrl=%entry]

    %header : Control = region()
    %i0 : Int<64> = phi<%entry, %body>(%init, %next) [ctrl=%header]
    %hi : Index = 3
    %cmp : Int<1> = icmp<"slt">(%i0, %hi) [ctrl=%header]
    %_ : Nil = cond_br(%cmp, %body, %exit) [ctrl=%header]

    %body : Control = region()
    %one : Index = 1
    %next : Int<64> = add(%i0, %one) [ctrl=%body]
    %_ : Nil = br(%header) [ctrl=%body]

    %exit : Control = region()
    %_ : Nil = ret(()) [ctrl=%exit]
```

The key difference: ops explicitly declare their region via `[ctrl=...]`. The formatter groups them for readability, but the grouping is derived from the control edges, not from container objects.

### Example: Floating Pure Ops

In Sea-of-Nodes, pure ops (no side effects) can float between regions. The scheduler decides where to place them:

```
# In the graph (no control pinning):
%x = add(%a, %b)        # pure — floats freely
%y = mul(%x, %c)        # pure — floats freely
%_ = store(%y, %ptr) [ctrl=%entry]   # pinned to entry

# Scheduler might place %x and %y in any dominating region
# GVN can discover that %x is computed identically in two branches
# and hoist it above the branch — this is automatic because %x has no
# control input constraining it
```

This is how HotSpot's C2 compiler (Click & Paleczny, 1995) achieves powerful optimizations with a simple framework: floating ops naturally enable code motion, common subexpression elimination, and loop-invariant code motion without dedicated passes for each.

## Design Summary

| Aspect | Current (MLIR-style blocks) | Sea-of-Nodes |
|---|---|---|
| Block membership | `_stored_ops` list | Derived from control edges |
| Side effects | ChainOp (control disguised as data) | Explicit control input |
| `walk_ops` | Crosses block boundaries | Follows data edges only |
| Scheduling | Implicit in block structure | Explicit exit pass |
| Pure op placement | Fixed to defining block | Floats; scheduler decides |
| Cross-block data flow | Direct value references (leaks) | Direct value references (safe — control edges are separate) |

## Relationship to Other Designs

### Symbols (`passes.md`)

Symbols solve the forward-reference problem (mutual recursion, irreducible CFGs). They are orthogonal to Sea-of-Nodes vs blocks. In a Sea-of-Nodes IR, symbols would still be needed for `forward_declare`/`link` patterns. The two designs compose cleanly.

### Pass framework (`passes.md`)

The pass framework's `replace_uses` and use-def walking work *better* with Sea-of-Nodes. Currently, handlers must be careful about block boundaries and chain threading. With explicit control edges:

- `replace_uses` on a data edge has no control-flow side effects.
- Side-effecting replacement ops carry their own control input — no manual chain threading.
- DCE is still implicit (unreferenced ops disappear) and safe (control edges keep pinned ops alive from the region's perspective).

### Chains (`passes.md`)

Chains become unnecessary. The chain mechanism exists because the IR has only data edges, so control information must be encoded as data. With two edge types, each concern has its own mechanism.

## Migration Path

The current label-with-body implementation is **incrementally closer** to Sea-of-Nodes:

1. Labels are values (ops that produce `Label` type) — this is a control value.
2. Branches reference labels as operands — these are control edges (though currently encoded as data edges).
3. The codegen already separates scheduling (iterate label bodies) from data flow.
4. `unwrap_chain` already distinguishes the chain spine (control) from data dependencies.

A migration could proceed:

1. **Introduce a `Control` type** — labels already produce `Label`, which is semantically a control type.
2. **Add `ctrl` field to side-effecting ops** — initially redundant with `_stored_ops`, but establishes the pattern.
3. **Derive block membership from `ctrl`** — the formatter and codegen use control edges instead of `_stored_ops` to determine grouping.
4. **Remove `_stored_ops`** — blocks become projections over the graph, not containers.
5. **Remove `ChainOp`** — control edges replace chain threading.

Each step is independently testable and preserves ASM round-trip correctness.

## References

- Click, "From Quads to Graphs: An Intermediate Representation's Journey" (1993) — introduces Sea-of-Nodes
- Click & Paleczny, "A Simple Graph-Based Intermediate Representation" (1995) — Sea-of-Nodes in HotSpot's C2 JIT compiler
- Kelsey, "A Correspondence between Continuation Passing Style and Static Single Assignment Form" (1995) — SSA ↔ CPS equivalence
- Appel, *Compiling with Continuations* (1992) — CPS as compiler IR
- Braun et al., "Simple and Efficient Construction of Static Single Assignment Form" (2013) — SSA construction without dominators
- Leißa et al., "A Graph-Based Higher-Order Intermediate Representation" (2015) — Thorin/AnyDSL, continuation-based IR
- Stanier & Watson, "Intermediate Representations in Imperative Compilers: A Survey" (2013) — comprehensive IR taxonomy
- Graal/Truffle (Oracle Labs) — modern Sea-of-Nodes JIT, successor to C2's approach
- Cliff Click's blog, "The Sea of Nodes and the HotSpot JIT" — accessible introduction to the ideas
- FIRM (libFirm) — another Sea-of-Nodes IR (Karlsruhe Institute of Technology), used in the cparser C compiler

## Open Questions

### Control edge representation

Should control edges be a dedicated field (`ctrl: Value[Control]`) or a separate edge type in the graph infrastructure? A dedicated field is simpler and matches the current dataclass-field-driven op definition. A separate edge type would require changes to the `Op` base class and field introspection.

### Phi nodes vs block arguments

Sea-of-Nodes typically uses phi nodes (as LLVM does). The current MLIR-style design in `passes.md` favors block arguments. Both work; the choice affects ergonomics. Phi nodes are natural in Sea-of-Nodes because there are no block containers to hold arguments. Block arguments could be modeled as explicit projection ops from the region's control token.

### Floating op scheduling strategy

When a pure op can be placed in multiple valid positions, the scheduler must choose. Options include:
- **As early as possible** — place at the dominating position (reduces register pressure in loops)
- **As late as possible** — place just before use (reduces live ranges)
- **Profile-guided** — place on the hot path only

This is a codegen/exit-pass concern, not an IR concern. The IR should not constrain the choice.

### Interaction with structured control flow

Structured ops (`IfOp`, `ForOp`) have nested blocks. In a Sea-of-Nodes IR, should these become region-based (ops inside carry control inputs pinning them to the structured region)? Or should structured ops remain as containers, with Sea-of-Nodes only for unstructured control flow? The cleanest design is uniform: everything is Sea-of-Nodes, and structured ops are sugar that the formatter recognizes for pretty-printing.
