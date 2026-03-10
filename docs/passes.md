# Pass Framework Design

## Problem

The codebase has four passes with three distinct implementation patterns:

1. **Lowering passes** (`toy_to_affine`, `affine_to_llvm`, `builtin_to_llvm`) — walk functions, `isinstance` dispatch per op, yield replacement ops, manually maintain a value map
2. **Optimization pass** (`optimize`) — in-place rewrite on a deepcopy, function-level transforms composed sequentially
3. **Analysis pass** (`shape_inference`) — in-place mutation of types, no op replacement

All three are "dispatch on op type, do something per-op" with manual boilerplate. There is no shared infrastructure for value mapping, verification, or pass composition.

## Design

### Core Concepts

**Pass**: declares what it can rewrite (handlers keyed on op type) and what IR it expects (domain/range sets). A pass does not control how it is walked — that is the framework's responsibility.

**Rewriter**: passed to handlers. Exposes one mutation primitive:

- `replace_uses(old_value, new_value)` — eagerly walk all ops that reference `old_value` and update them to reference `new_value` instead. This is an immediate, in-place mutation — no deferred resolution, no value map to consult later.

Passes mutate the IR in place. No deepcopy.

**Handler**: `(self, op: Op, rewriter: Rewriter) -> bool`. Registered via `@lowering_for(OpType)`. Multiple handlers may be registered per op type; they are tried in registration order. Returns whether it acted.

**Readiness guarantee**: handlers are only ever invoked on ops whose operands are *ready* — all compile-time values are resolved, all dependencies are satisfied. The staging system manages JIT compilation via the pass pipeline to ensure this. For IR that is already structured so all values are ready (the common case), this degrades gracefully to the normal MLIR conception of a pass. This means handlers do not need `isinstance` guards to check whether their operands have the expected types — they can assume correctness.

### Pass Declaration

```python
class ToyToAffine(Pass):
    op_domain = {toy.TransposeOp, toy.MulOp, toy.AddOp, toy.ReshapeOp,
                 toy.PrintOp, ConstantOp, builtin.ReturnOp, ...}
    op_range = {affine.AllocOp, affine.ForOp, affine.LoadOp, affine.StoreOp,
                affine.PrintMemrefOp, ConstantOp, builtin.ReturnOp, ...}
    type_domain = {toy.Tensor}
    type_range = {affine.MemRef, toy.Tensor}
    allow_unregistered_ops = False

    @lowering_for(toy.TransposeOp)
    def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        alloc_op = affine.AllocOp(shape=op.type.shape)
        # ... build nested for ops ...
        rewriter.replace_uses(op, alloc_op)
        return True
```

### Walk Behavior

The pass manager walks the use-def graph from each block's result value, visiting ops in dependency order:

- **Op has registered handler(s):** call them in registration order until one returns `True`. The handler owns the op entirely, including its nested blocks. The pass manager does **not** recurse into the op's blocks.
- **Op has no registered handler, `allow_unregistered_ops=True`:** recurse into the op's nested blocks (applying the pass within them), leave the op in place.
- **Op has no registered handler, `allow_unregistered_ops=False`:** error.

This means `allow_unregistered_ops` determines the walk mode:

- `True` → **transparent pass.** Unknown ops pass through; their blocks are recursed into. Good for partial lowerings and optimizations.
- `False` → **manual pass.** Every op must have a handler. Good for full dialect-to-dialect lowerings.

Since `replace_uses` is eager (it immediately updates all referencing ops), the pass manager does not need to remap operands — they are already correct by the time downstream ops are visited.

### The IR is a graph, not a list

A `Block` stores its **result value**, not a list of ops. The ops in a block are the transitive dependencies of the result value — they are discovered by walking the use-def graph. This means:

- **Passes** traverse the use-def graph and mutate it in place. When a handler creates new ops and calls `replace_uses(old, new)`, the new ops are automatically part of the graph (they're reachable from whatever references `new`). No splicing, no insertion ordering.
- **Formatting** (ASM printing) walks the use-def graph from the block's result, topologically sorts the ops, and emits them in linear order. The topological sort is a display concern, not a pass concern.
- **Parsing** reads linear ASM and builds the graph. The parser already does this — ops reference each other via SSA names, and the parser resolves these to value references.

```
Block:
  result: Value    ← the root of the use-def graph
  args: list[BlockArgument]

  ops (derived):   ← walk use-def from result, topological sort
```

This eliminates deepcopy (passes mutate in place), eliminates manual op list management (ops exist by being reachable), and makes DCE implicit (unreachable ops simply aren't in the graph).

**Dependency on chains:** for this to work, side-effecting ops must be reachable from the block's result. The chain mechanism (threading side effects through the use-def graph to the return value) is required. Without chains, side-effecting ops like `print` would be unreachable and lost. See [Side effects and the chain mechanism](#side-effects-and-the-chain-mechanism).

### Verification

- **Pre-pass:** walk all ops, assert each is in `op_domain`. Walk all types, assert each is in `type_domain`.
- **Post-pass:** walk all ops, assert each is in `op_range`. Walk all types, assert each is in `type_range`.

Verifiers are **debug assertions** — the pass manager decides whether to run them, and usually will not for performance. Subclasses may override verification methods to add custom checks, and should call `super().verify_preconditions()` / `super().verify_postconditions()` to retain the base domain/range checks:

```python
class ToyToAffine(Pass):
    def verify_preconditions(self, module: Module) -> None:
        super().verify_preconditions(module)
        # custom: every tensor must have a known shape
        for op in walk(module):
            if isinstance(op.type, toy.Tensor):
                assert op.type.shape is not None

    def verify_postconditions(self, module: Module) -> None:
        super().verify_postconditions(module)
        # custom: no MemRef leaks (every alloc has a matching dealloc)
        ...
```

### Walk Strategy

The pass declares handlers. The framework decides execution strategy:

- **Single forward pass** (default) — walk ops once, apply handlers
- **Fixed-point iteration** — repeat until no handler returns `True`

The pass does not know or care which strategy is used. This is possible because the pass API (multiple handlers per op, `bool` return) supports both without changes.

### Handler Registration

Multiple handlers per op type, tried in registration order:

```python
class Optimize(Pass):
    allow_unregistered_ops = True

    @lowering_for(toy.ReshapeOp)
    def fold_constants(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        new_op = ConstantOp(value=op.input.memory.to_json(), type=...)
        rewriter.replace_uses(op, new_op)
        return True

    @lowering_for(toy.ReshapeOp)
    def simplify_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        new_op = toy.ReshapeOp(input=op.input.input, type=op.type)
        rewriter.replace_uses(op, new_op)
        return True
```

`fold_constants` is tried first. If it acts, `simplify_reshape` is not tried on this op. If it doesn't act, `simplify_reshape` gets a chance. Note: no `isinstance` guards are needed — the readiness guarantee means `op.input` has its expected type when the handler is called.

## Design Discussion

This section records the reasoning behind key design choices, including alternatives that were considered and why they were rejected. It is written for future readers who need to understand not just *what* was decided but *why*.

### Why `replace_uses` instead of `replace_op`

The first design considered had handlers returning replacement ops:

```
# Approach 1: handler returns replacement ops
def lower_transpose(self, op) -> Iterator[Op]:
    yield alloc_op
    yield for_op
    # framework splices these in, rewrites uses of op -> ???
```

This has two problems. First, the handler must decide which of the yielded ops is the "replacement value" that downstream ops should reference. Second, the handler is responsible for linearizing the new ops into the correct order, which is error-prone and redundant with information already present in the use-def graph.

The second design considered a combined primitive:

```
# Approach 2: explicit splice + rewrite
rewriter.replace_op(old_op, new_ops=[alloc_op, for_op], new_value=alloc_op)
```

This is better but still requires the handler to enumerate all new ops. Since ops reference their operands (the use-def graph), the framework can discover the full set of new ops by walking from the replacement value.

The final design uses only `replace_uses(old, new)`:

```
# Approach 3: replace_uses only
alloc_op = affine.AllocOp(shape=...)
for_op = affine.ForOp(body=Block(ops=[...load, store...], ...))
rewriter.replace_uses(op, alloc_op)
```

The handler builds the new ops (which are automatically part of the use-def graph via their operand references) and calls `replace_uses` to redirect downstream references. The new ops are reachable from the block's result because downstream ops now point to `alloc_op`. No splicing, no insertion ordering — the graph is the IR.

### Why use-def discovery works: blocks and visibility

A key invariant makes use-def discovery clean: **ops defined inside a block are never visible outside that block.** The only externally-visible result of a block is the return value of the enclosing block op.

```
┌─ ForOp ──────────────────────────┐
│  body block:                     │
│    load = LoadOp(memref=alloc)   │  ← not visible outside ForOp
│    store = StoreOp(val=load)     │  ← not visible outside ForOp
│    return(store)                 │
└──────────────────────────────────┘
         │
         ▼
  ForOp itself is visible (it's the value downstream ops reference)
```

This means when the framework walks the use-def graph from a replacement value, it only finds top-level ops. Ops nested inside a `ForOp.body` or `IfOp.then_body` are not reachable from outside — they're encapsulated. The framework collects the top-level new ops, topologically sorts them, and splices them into the parent block.

Without this invariant, use-def discovery would pull interior ops into the parent block, breaking the nesting structure.

### Side effects and the chain mechanism

In a use-def graph where ops are only kept alive by references from other ops, pure computation is straightforward — unused ops are dead and can be eliminated. But side-effecting ops (stores, prints, I/O) must be preserved even if no other op references their result.

The design requires all side-effecting ops to be linked into the use-def graph via a **chain** mechanism — a data dependency that threads through side-effecting ops and is ultimately `return`ed. This serves two purposes:

1. **DCE safety**: since side effects are in the use-def graph, a general DCE pass (remove unreferenced ops) is safe for all ops, with no special-casing for side effects.
2. **Use-def discovery**: when a handler produces side-effecting ops, the chain links them into the replacement value's dependency graph, so the framework discovers them automatically.

```
  chain_in ──→ StoreOp ──→ PrintOp ──→ ReturnOp
                  │            │
                  ▼            ▼
              (store is      (print is
               kept alive     kept alive
               by chain)      by chain)
```

The ergonomics of threading chains through pass handlers need further design work — it must be natural to write handlers that preserve the chain without boilerplate.

### The label problem: linearization vs structured control flow

The current `builtin_to_llvm` pass lowers `IfOp` (structured, with `then_body`/`else_body` blocks) into a flat sequence of LLVM-dialect ops:

```
  CondBrOp → then_label
  LabelOp("then")
  ... then ops ...
  BrOp → merge_label
  LabelOp("else")
  ... else ops ...
  BrOp → merge_label
  LabelOp("merge")
  PhiOp(then_val, else_val)
```

This is **premature linearization** — it converts structured control flow into a flat basic-block representation at the IR level. Labels are not values: nothing depends on a label, and a label depends on nothing. They are ordering markers in a linear sequence.

This is incompatible with use-def-based op discovery. When the framework walks the use-def graph from a replacement value, it cannot find labels — they have no data dependencies. The first op after a label may have no data dependencies at all (it's reachable only via control flow).

```
  CondBrOp ──data──→ cond_value
  LabelOp("then")              ← no data edges in or out
  LoadOp(ptr=...)              ← may only depend on a ptr from before the branch
```

This motivated a deeper investigation into how control flow should be represented in the IR.

### Unstructured control flow: blocks as values

Rather than giving up unstructured control flow (which would limit what programming models dgen users can target), the design introduces control flow ops where **labels are ops, and therefore values**, making them visible in the use-def graph.

Three ops form an `unstructured_cf` dialect:

```python
@dialect.op("label")
class LabelOp(Op):
    body: Block      # the basic block's code

@dialect.op("branch")
class BranchOp(Op):
    target: Value    # a LabelOp

@dialect.op("cond_branch")
class CondBranchOp(Op):
    cond: Value
    true_target: Value   # a LabelOp
    false_target: Value  # a LabelOp
```

`LabelOp` is an op (and therefore a value). `BranchOp` references it as an operand. Control flow edges are now data edges in the use-def graph.

This is essentially **continuations**: each label is "a computation that can be jumped to," and branching is "invoke this continuation." Block arguments (parameters of the label's block) replace phi nodes — a branch passes values to the target label.

```
# Simple if/else as unstructured control flow:

%then = label { %x = add(%a, %b); return(%x) }
%else = label { %y = sub(%a, %b); return(%y) }
%result = cond_branch(%cmp, %then, %else)
```

The pass framework discovers `%then` and `%else` via use-def from `%result`. Their bodies are blocks — interior ops aren't visible outside (existing invariant). No model changes required.

**Structured and unstructured coexist.** `IfOp` (structured, blocks as fields) and `label`/`branch` (unstructured, blocks as values) are different dialects. A lowering pass can convert `IfOp` → `label` + `cond_branch` when needed. Or a dialect can stay structured all the way to codegen. User's choice.

**Self-reference (loops):** a label's body can reference the label itself:

```
%loop = label(%i) {
    %next = add(%i, 1)
    %done = icmp(%next, %n)
    cond_branch(%done, %exit, %loop, %next)   ← self-reference
}
```

This is natural scoping — a label is in scope within its own body, just as a function can call itself by name. It's recursion, not a data dependency cycle (the label value exists before its body executes).

**Mutual reference (irreducible CFGs):** two labels referencing each other:

```
%B = label { ...; branch(%C) }    ← references %C
%C = label { ...; branch(%B) }    ← references %B
```

Neither can be defined before the other. This is a genuine cycle in the value graph. The solution is **symbols** (see next section).

**Connection to prior art:**
- Blocks-as-values = continuations. `br(label, args)` = continuation invocation.
- SSA and CPS are formally equivalent: Kelsey, "A Correspondence between Continuation Passing Style and Static Single Assignment Form" (1995).
- Appel, *Compiling with Continuations* (1992) — CPS as compiler IR.
- Kennedy, "Compiling with Continuations, Continued" (2007) — practical CPS IR design.
- Leißa et al., "A Graph-Based Higher-Order Intermediate Representation" (2015) — Thorin/AnyDSL, a continuation-based IR where functions and continuations are values.

### Symbols and forward references

Mutual reference between labels (and mutual recursion between functions) creates cycles in the value graph. **Symbols** break these cycles by introducing a level of indirection: a name that can be referenced before its target is defined.

Two paths:

**Direct symbol** (definition known at creation, pure op):

```
%add_fn = function { ... }
%add: Function<[Index, Index], Index> = symbol<"add">(%add_fn)
```

`symbol` is a pure op — it binds a name to a value. No mutation, no side effects.

**Forward declaration + link** (definition comes later):

```
%B: Label = forward_declare<"B">()
%C: Label = forward_declare<"C">()

%B_impl = label { ...; branch(%C) }
%C_impl = label { ...; branch(%B) }

%chain1: Nil = link(%chain0, %B, %B_impl)
%chain2: Nil = link(%chain1, %C, %C_impl)
```

`forward_declare` creates a typed, unresolved symbol. `link` resolves it. `link` is a side effect (it mutates the symbol binding), so it is chained. `forward_declare` and `link` must be in the **same block** — this ensures resolution is verifiable locally.

**The value graph is a DAG:**

```
forward_declare("B") ──→ %B ──→ branch (inside %C_impl's body)
forward_declare("C") ──→ %C ──→ branch (inside %B_impl's body)

%B ─────┐
%B_impl ┼──→ link ──→ %chain1 ──→ link ──→ %chain2 ──→ return
%chain0 ┘    (B)                   (C)
%C ─────┘
%C_impl ┘
```

No cycles. `%B` and `%C` are values created by `forward_declare` before their implementations exist. The label bodies reference these symbols, not the implementations. `link` connects everything through the chain. The use-def graph remains a DAG.

The symbol value is a **constant** — the system guarantees it is resolvable at compilation time, similar to `ConstantOp`. Passes can treat symbols like any other constant value.

**Verification:** at block boundary, every `forward_declare` must have a corresponding `link`. An unresolved `forward_declare` without a `link` in the same block is a verifier error. (External symbols — values defined in another module — are a separate concept to be designed with the module system.)

**How this solves the letrec problem:** symbols provide forward references. The `forward_declare`/`link` pattern is essentially the "allocate then backpatch" implementation of `letrec` (Scheme) or `let rec ... and ...` (ML), made explicit as IR ops rather than hidden in the parser or runtime.

**Prior art for symbols in IRs:**
- **LLVM/MLIR**: functions and globals are symbols. `declare @foo(...)` is a forward declaration; `define @foo(...)` is the definition. MLIR formalizes this with `SymbolTable`, `Symbol` trait, and `SymbolRefAttr`. Symbols are looked up by string, outside the SSA graph.
- **Scheme `letrec`**: mutually recursive bindings. Implementation: allocate boxes, then backpatch. Names in scope for all definitions.
- **ML `let rec ... and ...`**: the compiler allocates closures with dummy values and backpatches.
- **C/C++ forward declarations**: typed symbol declarations resolved by the linker.

**Open question:** the string name in `symbol<"name">` is a pragmatic starting point. A richer design with modules as namespaces may replace it, but the `symbol`/`forward_declare`/`link` mechanics don't depend on how names work.

### Reducible and irreducible control flow

Structured control flow ops (`IfOp`, `ForOp`, `WhileOp`) always produce **reducible** control flow graphs — every cycle has a single entry point (a "loop header"), and the graph can be decomposed into nested regions.

An **irreducible** CFG has a cycle with multiple entry points:

```
        A
       / \
      ▼   ▼
      B → C
      ▲   │
      └───┘
```

Both B and C are reachable from outside the cycle (from A) and from inside (from each other). No structured nesting of if/for/while can express this. It requires `goto` or computed jumps.

With the symbols design, irreducible CFGs are expressible via `forward_declare`/`link`:

```
%B = forward_declare<"B">()
%C = forward_declare<"C">()
%B_impl = label { ...; branch(%C) }
%C_impl = label { ...; branch(%B) }
link(%B, %B_impl)
link(%C, %C_impl)
cond_branch(%cmp, %B, %C)
```

Reducible CFGs (loops) need only self-reference, which works without symbols — a label in scope within its own body.

**The Relooper alternative:** the Relooper algorithm (Zakai 2011, used by Emscripten and WebAssembly) converts arbitrary CFGs to structured form by introducing helper variables and loop constructs. This avoids the need for unstructured control flow in the IR entirely. It is a proven approach but can produce suboptimal code for complex irreducible cases. With dgen's symbol mechanism, the Relooper is unnecessary — both reducible and irreducible CFGs are directly expressible.

**References:**
- Hecht & Ullman, "Flow Graph Reducibility" (1972) — original definition
- Dragon Book (Aho, Lam, Sethi, Ullman), Chapter 9 — textbook treatment of reducibility, dominators, natural loops
- Zakai, "Emscripten: An LLVM-to-JavaScript Compiler" (2011) — Relooper algorithm
- Click, "From Quads to Graphs" (1993) — Sea-of-Nodes introduction
- Click & Paleczny, "A Simple Graph-Based Intermediate Representation" (1995) — Sea-of-Nodes in the HotSpot JVM
- Stanier & Watson, "Intermediate Representations in Imperative Compilers: A Survey" (2013) — IR survey including control flow representation tradeoffs

### Op ordering: topological sort is a display concern

Since the IR is a graph (not a list), there is no inherent ordering of ops within a block. Ordering only matters for display (ASM printing) and codegen. The formatter walks the use-def graph from the block's result and topologically sorts the ops to produce a valid linear order:

```
  stride_const = ConstantOp(value=2)
  mul = MulOp(lhs=idx, rhs=stride_const)    ← depends on stride_const
  add = AddOp(lhs=acc, rhs=mul)             ← depends on mul

  Topological order for display: stride_const, mul, add
```

Passes never think about ordering. They build ops (which reference each other via operands) and call `replace_uses`. The graph structure captures all ordering constraints implicitly.

## Paths Not Taken

### Functional replacement model (`op -> list[Op]`)

Handlers return replacement ops directly; the framework splices them in. Simpler, but can't express in-place mutations (type annotation changes in shape inference) without escape hatches. Also forces the handler to manage op ordering. See [Why `replace_uses` instead of `replace_op`](#why-replace_uses-instead-of-replace_op) for the full discussion.

### Separate base classes for transparent/manual passes

A `TransparentPass` and `ManualPass` with different base classes. Rejected because the distinction is already captured by `allow_unregistered_ops` — adding a class hierarchy for it is unnecessary.

### MLIR-style PatternRewriter with worklist

Full worklist-based fixpoint: pop ops, try patterns, add modified ops back to worklist, repeat until empty. Powerful (patterns can enable each other across distant ops) but heavyweight. The current design supports fixpoint iteration as a walk strategy without the worklist machinery. Can be added later if needed — the pass API (multiple handlers per op, `bool` return) is compatible.

### `erase(op)` as a separate primitive

Explicit op deletion. Not needed as a rewriter primitive — unreferenced ops can be cleaned up by a general DCE pass. Since side-effecting ops are linked into the use-def graph via the `chain` mechanism, DCE is safe for all ops without special-casing.

### Multiple blocks per region (LLVM/MLIR model)

Functions (or any op with regions) would contain a list of basic blocks, with control flow as the block graph and data flow as use-def within/across blocks. Rejected because it complicates the data model — every piece of code that walks ops must now handle multiple blocks per region. The blocks-as-values approach achieves the same expressiveness without changing the core data model.

### Control tokens (Sea-of-Nodes style)

Make control flow edges explicit data edges by threading "control token" values through every op: branches produce tokens, ops consume them. This is the pure Sea-of-Nodes approach. Rejected because it's verbose — every op in a basic block needs a control input/output even if it's pure computation. The blocks-as-values approach is less intrusive: only branch/label ops participate in control flow edges.

### List-based Block with deepcopy and value maps

The initial design sketch had the pass framework deepcopy the module, walk the op list, maintain a value map (`id(old) → new`), and provide a `rewriter.resolve(value)` method to look up remapped values. New ops were discovered by walking the use-def graph from the replacement value, topologically sorted, and spliced into the op list.

Rejected because: deepcopy is expensive, value maps add indirection (every value access goes through `resolve`), and topological sort in the pass is redundant work that the formatter already needs to do. The graph-based Block with eager `replace_uses` is simpler — passes just mutate the graph in place, and ops exist by being reachable from the block's result.

### Keeping structured control flow only (no unstructured support)

Restrict dgen to `IfOp`/`ForOp`/`WhileOp` and push linearization to codegen. Simpler, but prescriptive — limits what programming models users can target. Languages with `goto`, state machines, and hand-optimized control flow would require the Relooper algorithm (which can produce suboptimal code). Rejected in favor of offering both structured and unstructured as dialect choices.

## Open Questions

### Chain ergonomics in pass handlers

The chain mechanism ensures side effects are in the use-def graph, but handlers that produce side-effecting ops need to thread the chain naturally. How does a handler receive the current chain, and how does it return the updated chain? This affects the handler signature and the rewriter API.

### Dead code elimination strategy

Two options for cleaning up unreferenced ops after `replace_uses`:

1. **Reference counting** in the rewriter — immediate cleanup on replacement
2. **General DCE pass** — periodic cleanup, simpler rewriter

Punted for now. Both work. The chain mechanism ensures DCE is safe for all ops.

### Type domain/range granularity

Currently four sets: `op_domain`, `op_range`, `type_domain`, `type_range`. May need refinement — e.g., a type might appear both as an op's result type and as a nested type parameter. The verification walk needs to decide how deep to check.

### Symbol naming and modules

The string name in `symbol<"name">` is a pragmatic starting point. A richer design with modules as namespaces may replace it. The `symbol`/`forward_declare`/`link` mechanics don't depend on how names work — the naming scheme can evolve independently.

### Interaction between symbols and staging

Symbol values are constants (resolvable at compilation time). How this interacts with the staging system (compile-time vs runtime resolution) needs further design work.

## Examples

### Lowering pass (toy -> affine)

```python
class ToyToAffine(Pass):
    op_domain = {*toy.ops, ConstantOp, builtin.ReturnOp, builtin.AddIndexOp}
    op_range = {*affine.ops, ConstantOp, builtin.ReturnOp, builtin.AddIndexOp}
    type_domain = {toy.Tensor, builtin.Index, builtin.F64}
    type_range = {affine.MemRef, toy.Tensor, builtin.Index, builtin.F64}
    allow_unregistered_ops = False

    @lowering_for(toy.TransposeOp)
    def lower_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        in_alloc = op.input  # operands are already remapped by the framework
        alloc_op = affine.AllocOp(shape=op.type.shape)
        # ... build ForOp with load/store body ...
        rewriter.replace_uses(op, alloc_op)
        return True
```

### Optimization pass (within toy dialect)

```python
class ToyOptimize(Pass):
    op_domain = {*toy.ops, ConstantOp, builtin.ReturnOp}
    op_range = {*toy.ops, ConstantOp, builtin.ReturnOp}
    type_domain = {toy.Tensor, builtin.Index, builtin.F64}
    type_range = {toy.Tensor, builtin.Index, builtin.F64}
    allow_unregistered_ops = True  # pass through ops it doesn't rewrite

    @lowering_for(toy.TransposeOp)
    def eliminate_double_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        # Readiness guarantee: op.input has its expected type
        rewriter.replace_uses(op, op.input.input)
        return True

    @lowering_for(toy.ReshapeOp)
    def fold_constants(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        new_op = ConstantOp(value=op.input.memory.to_json(), type=...)
        rewriter.replace_uses(op, new_op)
        return True

    @lowering_for(toy.ReshapeOp)
    def simplify_reshape(self, op: toy.ReshapeOp, rewriter: Rewriter) -> bool:
        new_op = toy.ReshapeOp(input=op.input.input, type=op.type)
        rewriter.replace_uses(op, new_op)
        return True
```

### Analysis pass (shape inference)

```python
class ShapeInference(Pass):
    op_domain = {*toy.ops, ConstantOp, builtin.ReturnOp, builtin.CallOp}
    op_range = {*toy.ops, ConstantOp, builtin.ReturnOp, builtin.CallOp}
    type_domain = {toy.Tensor, toy.InferredShapeTensor, builtin.Index}
    type_range = {toy.Tensor, builtin.Index}  # all shapes resolved
    allow_unregistered_ops = True

    @lowering_for(toy.TransposeOp)
    def infer_transpose(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
        # Readiness guarantee: op.input.type is a concrete Tensor
        op.type = toy.Tensor(shape=reversed(op.input.type.unpack_shape()))
        return True

    def verify_postconditions(self, module: Module) -> None:
        super().verify_postconditions(module)
        # No unresolved shapes remain
        for op in walk(module):
            assert not isinstance(op.type, toy.InferredShapeTensor)
```

### Unstructured control flow (irreducible CFG)

```python
# Forward-declare mutually-referencing labels
%B: Label = forward_declare<"B">()
%C: Label = forward_declare<"C">()

# Define label implementations (can reference each other via symbols)
%B_impl = label {
    ...
    branch(%C)
}
%C_impl = label {
    ...
    branch(%B)
}

# Resolve forward declarations (chained, same block)
%chain1: Nil = link(%chain0, %B, %B_impl)
%chain2: Nil = link(%chain1, %C, %C_impl)

# Entry point
cond_branch(%cmp, %B, %C)
```
