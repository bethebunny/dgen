# dgen IR Design: Scope, Capture, and Use-Def Cycles

*dgen Compiler Framework — IR Design Series*

---

## Abstract

This document describes the recommended approach to scope, value capture, and use-def cycle
management in the dgen IR. The central problems are: (1) use-def cycles that arise from recursive
functions and loop back-edges, and (2) the definition and enforcement of a capture discipline that
keeps block semantics well-defined. The recommendations are grounded in established compiler theory
— Sea of Nodes, CPS, and SSA — and are designed to support dgen's meta-compilation goals,
hash-consing, and eventual tiered scheduling.

---

## 1. Background and Problems

### 1.1 The IR Model

The dgen IR is structured around **ops** and **blocks**. An op may have zero or more blocks
attached to it. A block has zero or more typed arguments and exactly one result. The use-def graph
connects op outputs to op inputs. Blocks are not explicitly tracked as containers of ops; an op
belongs to a block by virtue of being reachable from that block's result or argument uses in the
use-def graph.

### 1.2 Problem: Use-Def Cycles

A naive IR representation of recursive functions and loops produces cycles in the use-def graph.
Consider a simple loop:

```
// Hypothetical (creates use-def cycle):
%loop : llvm.Label = llvm.label() (%i: Index):
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
    %_ : Nil = llvm.cond_br(%cond, %body, %exit, [], [])

%body : llvm.Label = llvm.label() ():
    %one : Index = 1
    %i_next : Index = subtract_index(%i, %one)
    %_ : Nil = llvm.br(%loop, [%i_next])    // back-edge: %i_next feeds back to %i
```

The use-def edges form a cycle: `%i` → `%i_next` → `llvm.br` → `%loop` → `%i`. This complicates analysis,
scheduling, and hash-consing, all of which prefer or require a DAG structure.

### 1.3 Problem: Scope and Capture

Without a formal capture discipline, it is ambiguous which block an op "belongs to." An op with
inputs drawn from two different block argument boundaries has an ill-defined scope. This makes
block-level analysis and inlining unsound, and prevents each block from being treated as a closed,
independently-meaningful term.

### 1.4 The Symbol Table Temptation

A common solution is a global or module-level symbol table: functions and labels are identified by
name strings, and cross-block references are resolved by lookup. LLVM IR and early MLIR designs use
this approach. The problems are:

- Names are not values. You cannot pass a symbol to a higher-order function, specialize over it, or
  reason about it in the use-def graph.
- Name resolution is a separate, implicit mechanism layered on top of the use-def graph, making the
  IR's full dependency structure non-local.
- Recursion and cross-references are handled via strings rather than typed edges, weakening the
  IR's invariants.

The recommendation is to **not use a symbol table at all**. All references are explicit value
edges. Recursion is handled by dedicated op forms described in Section 2.

---

## 2. Recommendations

### 2.1 `func.recursive`: Eliminating Use-Def Cycles in Recursive Functions

A `func.recursive` op introduces `%self` as its first block argument, typed with the function's
own declared signature. The body may call `%self` to recurse. Crucially, `%self` is a fixed input
— it is defined at the block boundary, not fed back from the recursive call's result. The recursive
call produces a fresh return value. Therefore:

- Execution cycles (the function calls itself).
- The use-def graph remains a DAG (`%self` is never redefined by the recursion).

```
// %self is defined once at the block boundary; the recursive call
// produces a fresh %r. No use-def cycle.
%factorial : Nil = func.recursive<Index>() (%self: Function<Index>, %n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond) ():
        %base : Index = 1
    else ():
        %n1 : Index = subtract_index(%n, %one)
        %r : Index = call(%self, [%n1])    // uses %self (fixed input), produces fresh %r
        %res : Index = llvm.mul(%n, %r)
```

The type of `%self` is simply the function's declared signature, resolved at definition time. This
is **nominal isorecursion**: the declared type is the fixpoint, and the self-reference is broken by
the declaration boundary rather than by an explicit roll/unroll.

#### Mutual Recursion via Nesting

Mutual recursion does not require a separate `func.letrec` primitive. One function may be defined
inside the other, capturing the outer function's `%self` as an explicit argument:

```
%even : Nil = func.recursive<Index>() (%self: Function<Index>, %n: Index):
    // odd is defined inside even; it captures %self as %even_ref
    %odd : Nil = function<Index>() (%even_ref: Function<Index>, %n: Index):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %res : Index = call(%even_ref, [%n1])
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"eq">(%n, %zero)
    %result : Index = if(%cond) ():
        %t : Index = 1
    else ():
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %res : Index = call(%odd, [%self, %n1])    // pass %self so odd can call back
```

`odd` is not self-recursive, so it does not need `func.recursive`. The mutual recursion cycle is
broken by nesting: `even` owns `odd`, and `odd` calls back through the captured reference. The
use-def graph is a DAG throughout.

The trade-off is asymmetry: one function must be nominated as the owner. For larger mutual
recursion groups (A → B → C → A), all inner functions are nested under the root with captured
references threaded explicitly. A `func.letrec` group op can be added later as syntactic sugar for
the symmetric case if warranted.

### 2.2 Label `%self`: Eliminating Use-Def Cycles in Loops

The loop back-edge problem is symmetric to the recursion problem, and the solution is symmetric.
Every block receives `%self : llvm.Label` as its first argument. A loop is expressed as a block
that calls `%self` with updated arguments:

```
// No back-edge in the use-def graph. %self is a fixed input.
// %i_next is an argument to the call, not fed back into a definition.
%loop : llvm.Label = llvm.label() (%self: llvm.Label, %i: Index):
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
    %body : llvm.Label = llvm.label() ():
        %one : Index = 1
        %i_next : Index = subtract_index(%i, %one)
        %_ : Nil = llvm.br(%self, [%i_next])    // recursive call through %self, not a back-edge
    %_ : Nil = llvm.cond_br(%cond, %body, %exit, [], [])
```

Comparing this to the back-edge version from Section 1.2: the use-def cycle
`%i → %i_next → llvm.br → %i` is eliminated. `%self` is fixed at the block boundary, and
`%i_next` is an argument at the call site, not a definition of `%i`.

#### Sibling Blocks and Mutual Block Recursion

When two sibling blocks mutually recurse (as in a loop split across blocks), the capture discipline
requires the label to be threaded explicitly. The pattern is identical to the `even`/`odd` nesting:

```
%loop : llvm.Label = llvm.label() (%self: llvm.Label, %i: Index):
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
    // pass %self to %body so it can call back
    %_ : Nil = llvm.cond_br(%cond, %body, %exit, [%self, %i], [])

%body : llvm.Label = llvm.label() (%loop_label: llvm.Label, %i: Index):
    %one : Index = 1
    %i_next : Index = subtract_index(%i, %one)
    %_ : Nil = llvm.br(%loop_label, [%i_next])
```

`%loop` passes `%self` to `%body` as an explicit argument `%loop_label`. `%body` calls back through
`%loop_label`. The entire graph is a DAG.

#### Blocks and Functions: A Unified View

With `%self : llvm.Label` on every block, the distinction between a `func.recursive` and a looping
block largely collapses: both are callable things that receive `%self` and can recurse through it.
The remaining distinction is that a `func` has a return type, while a block is a continuation — it
does not return; control transfers via `llvm.br`. This is explored further in Section 3.

### 2.3 Explicit Capture

The explicit capture discipline is: **a block's free variables must be declared as its arguments**.
More precisely: if op `V` is used inside block `B`, and `V` transitively depends on the arguments
of some block `B′` that is not an ancestor of `B` in the block nesting tree, then `V` must be
threaded through the argument lists on the path from `B′` to `B`.

This makes each block a **closed term**: its meaning is determined entirely by its arguments.
Benefits:

- Block-level analysis, inlining, and partial evaluation are sound in isolation.
- The full data dependencies of each block are manifest in its argument list.
- No implicit environment or dynamic scope to track.
- Aligns with scheduling requirements: the scheduler can determine a block's inputs by inspection.

### 2.4 Ambient Nodes

The explicit capture discipline has a natural exemption: ops that have no value dependencies at
all. We call these **ambient nodes**. An ambient node cannot transitively depend on any block's
arguments — there is nothing in its transitive input set that could cause a cross-block capture
violation. Therefore, ambient nodes may be referenced from any block without declaration.

The canonical ambient nodes are:

- Constants and integer/float literals (`%c : Index = 0`, `%f : F64 = 3.14`).
- Global references (immutable global values known at compile time).
- Function labels referenced as first-class values (when not recursive).
- Type metadata and other purely static values.

More generally, an op's **scope** is determined by the deepest block whose arguments it transitively
depends on. Ambient nodes have no such block — they float above the entire block structure. This is
not a special case; it is the natural consequence of the capture discipline applied to nodes with an
empty dependency set.

Ambient node sharing is the foundation for **hash-consing**: two ambient subgraphs with identical
structure and inputs are semantically equivalent and can be unified to a single node. This supports
implicit common subexpression elimination without a separate CSE pass — a desirable property for
dgen's meta-compilation workloads.

### 2.5 Scheduling Shared Pure Subgraphs

After explicit capture is enforced, some ops are still "free-floating": their scope is not pinned
to a specific block because all of their dependencies are ambient. These ops have a legal scheduling
window:

- **Earliest**: the block that dominates all of the op's dependencies. For ambient ops, this is the
  function entry.
- **Latest**: the block that post-dominates all of the op's uses. You cannot defer past the point
  where any consumer needs the value.

Any placement within `[earliest, latest]` is **sound**. Soundness requires only the use-def graph
and the dominator relation; no loop analysis is needed.

#### Heuristics

Within the legal window, placement heuristics optimize for performance:

- **Rematerialization** (for trivially cheap ops): constants and literals are emitted at each use
  site. Their live range is zero; no register allocation pressure. This is the degenerate "schedule
  at every use site" case.
- **Loop hoisting** (for non-trivial pure ops): an op whose uses are all inside a loop but whose
  dependencies are all outside it should be placed at the loop's pre-header — computed once, not on
  every iteration. This is the LCA-of-uses heuristic: place at the LCA of all use sites in the
  dominator tree, but walk upward toward `earliest` while still exiting the outermost loop.
- **Placement at LCA** (default): for ops that cannot be further hoisted, the LCA of all use sites
  is the natural placement — the latest point that still dominates all consumers.

#### Loop Detection

Loop detection is required only for the hoisting heuristic, not for soundness. With the
`%self : llvm.Label` design, loops are structurally manifest: a block is a loop if and only if its
body contains a `llvm.br(%self, ...)`. Loop nesting depth is read directly from block nesting. No
dominator analysis is needed to identify loop structure; it is visible in the use-def graph via
`%self` values.

#### Division of Labor with LLVM

dgen currently targets LLVM IR as its exit dialect. LLVM's optimization pipeline (LICM, GVN, loop
vectorization, instruction scheduling, register allocation) provides a strong baseline. The
recommended strategy for dgen's scheduler is therefore:

- Produce **sound** LLVM IR: legal placement, correct use-def threading, nothing more required.
- Defer loop-invariant code motion, register-pressure-aware scheduling, and other optimizations to
  LLVM.
- Revisit dgen-side scheduling when dgen's higher-level semantic knowledge (purity by construction,
  specialization decisions) can demonstrably improve on what LLVM can infer from the lowered form.

### 2.6 Tension with `builtin.call` and the Callee Parameter

The `func.recursive` and `%self` designs require calling a value passed as a block argument. This
creates tension with the current `builtin.call` design, where the callee is a compile-time
parameter:

```
op call<callee: Function>(args: List) -> Type    // callee in <> = compile-time parameter
```

ASM: `%result : T = call<%fn_ref>([%arg1, %arg2])`

The callee must be a `FunctionOp` value resolved at compile time. This works for static calls
between module-level functions. It does not work for:

1. **Recursive calls**: `%self` is a block argument, not a module-level `FunctionOp`.
2. **Higher-order functions**: a function value passed as a block argument cannot appear in the
   parameter position.

For example, calling a function argument:

```
%apply : Nil = function<Index>() (%f: Function<Index>, %n: Index):
    // Cannot write call<%f>([%n]) — %f is a block argument, not a compile-time FunctionOp.
    %result : Index = call(%f, [%n])    // callee must be an operand here
```

**Option A: Move callee to operand.** Change `call` to `call(callee, args)` with callee as a
runtime `Value`. This handles all cases uniformly. For static calls to module-level functions, a
`func_ref` ambient op wraps the reference:

```
%fn_ref : Function<Index> = func_ref<%my_function>()    // ambient — no block-arg dependencies
%result : Index = call(%fn_ref, [%arg])
```

`walk_ops` reaches `%fn_ref` and stops there — it is an ambient op. It never descends into
`%my_function`'s body. The `FunctionOp` guard in `walk_ops` (see Section 2.8) becomes unnecessary.

**Option B: Keep callee as parameter; add a separate call form.** Calls through block-argument
function values use a `call_dynamic` or `call_self` op. The static `call` op retains compile-time
callee semantics. The downside is two call forms with similar but not identical behavior.

**Option C: Accept the `walk_ops` guard.** Keep the current design. `walk_ops` retains its
`FunctionOp` special case. The cost is ongoing fragility and a non-structural exception.

Option A is the most uniform and eliminates the structural special case in `walk_ops`. The
`func_ref` ambient op cleanly wraps module-level references, and the `%self` recursive call is
expressed as a plain operand edge to a block argument.

### 2.7 Implicit Capture in the Current IR

The current IR has two ops that produce blocks with implicit capture: `builtin.if` and
`affine.for`. Both allow their body blocks to reference values from the enclosing scope without
threading them through block arguments.

**`builtin.if`**: The `then_body` and `else_body` blocks take no arguments. Any outer-scope value
used inside a branch is implicitly captured:

```
// Current (implicit capture):
%alloc : llvm.Ptr = llvm.alloca<6>()
%result : F64 = if(%cond) ():
    %loaded : F64 = llvm.load(%alloc)    // %alloc captured implicitly from outer scope
else ():
    %c : F64 = 0.0
```

Under explicit capture, outer values must be declared as block arguments:

```
// Explicit capture:
%alloc : llvm.Ptr = llvm.alloca<6>()
%result : F64 = if(%cond) (%alloc: llvm.Ptr):
    %loaded : F64 = llvm.load(%alloc)
else (%alloc: llvm.Ptr):
    %c : F64 = 0.0
```

**`affine.for`**: The body block receives the loop induction variable as a block argument, but not
the external memory references its body uses:

```
// Current (implicit capture):
%src : llvm.Ptr = llvm.alloca<6>()
%5 : Nil = affine.for<0, 4>() (%i: Index):
    %val : F64 = affine.load(%src, [%i])    // %src captured implicitly
```

Under explicit capture, external references appear as additional block arguments preceding the
induction variable:

```
// Explicit capture:
%src : llvm.Ptr = llvm.alloca<6>()
%5 : Nil = affine.for<0, 4>() (%src: llvm.Ptr, %i: Index):
    %val : F64 = affine.load(%src, [%i])
```

The lowering passes that generate `builtin.if` and `affine.for` nodes (`builtin_to_llvm.py`,
`toy_to_affine.py`) must be updated to thread all referenced outer values through the block
argument lists.

**Verifier.** A `Block.verify_closed()` check can be run after each lowering pass to enforce the
closed-block invariant. It walks each block's op set and flags any `Value` reference that is:
- not an op reachable from `block.result` via the use-def graph, and
- not a block argument of the block, and
- not an ambient node (i.e., it transitively depends on some block's arguments).

Running this verifier as a post-pass assertion makes capture regressions visible immediately rather
than silently producing unsound IR.

### 2.8 `walk_ops`: Consequences for Graph Traversal

`walk_ops(root)` traverses the use-def graph from `root` and returns ops in topological order. It
is the canonical op iterator for a block's contents. Its current implementation has two special
cases that exist because the IR uses implicit capture.

**The `FunctionOp` guard.** When visiting an op's parameters, `walk_ops` skips parameters that are
`FunctionOp` instances:

```python
for _, param in value.parameters:
    from dgen.dialects.builtin import FunctionOp   # circular-import workaround
    if not isinstance(param, FunctionOp):
        visit(param)
```

This prevents descending into a module-level function's body when it appears as a `call` callee.
Without the guard, `walk_ops` would include every op in the callee's body as part of the current
block's op list — mixing module-level and block-local ops.

**Block argument type traversal.** `walk_ops` visits the types of block arguments of each nested
block:

```python
for _, block in value.blocks:
    for arg in block.args:
        visit(arg.type)
```

This is needed when a block argument's type is itself an SSA value defined in an outer block
(a dependent type under implicit capture). Without it, those type dependencies would be invisible
to the walk.

**Under explicit capture**, both cases simplify:

- If callee moves to an operand (Option A in Section 2.6), `FunctionOp` never appears as a
  parameter. The guard and its circular import disappear; `walk_ops` follows all operands and
  parameters uniformly.
- If all blocks are closed, block argument types are either ambient (no block-arg dependencies) or
  already explicit block arguments. The type traversal remains correct but no longer needs special
  cross-scope handling.

The end state is a simple recursive DAG walk: visit operands, visit parameter values, visit the
result type, emit the op. No `isinstance` guards, no special cases — the structure is guaranteed by
the capture discipline rather than enforced by ad hoc checks.

---

## 3. Relationship to CPS

The design described above is closely related to Continuation-Passing Style (CPS). The
correspondences are direct:

| dgen IR concept                        | CPS equivalent                              |
|----------------------------------------|---------------------------------------------|
| `%self : llvm.Label`                   | Continuation argument                       |
| `llvm.br(%self, [%i_next])`            | Tail call to a continuation                 |
| Block with explicit capture            | Closed lambda abstraction                   |
| `function<R>` with return type `R`     | Function taking explicit return continuation|
| `llvm.br` to another label            | Tail call to another continuation           |
| Block argument list                    | Lambda parameter list                       |

### 3.1 Is CPS Always Flat?

"Flat" in CPS refers to the *dynamic* structure: there is no call stack, because every call is a
tail call and continuations are just jumps. It does not imply that the *lexical* structure is flat.
CPS-based IRs (MLton, SML/NJ's FLINT, Thorin/MimIR) have lexical nesting for scope and capture;
the flatness property means that dynamically, control never returns — it always transfers forward.

dgen's IR is flat in the CPS sense (`llvm.br` is a jump, not a stack-pushing call) but lexically
nested (blocks can be nested inside ops). This combination is valid and useful: dynamic flatness
gives efficient implementation, and lexical nesting makes scope manifest.

### 3.2 The `func` / `block` Distinction

In fully continuation-passing style (as in Thorin/MimIR), even the return from a function is
modeled as calling a return continuation argument `%return : llvm.Label`. This collapses the
`func`/`block` distinction entirely: everything is a continuation. The only distinction between a
"function" and a "basic block" is calling convention and ABI.

dgen does not go this far today: `func` retains a return type rather than an explicit return
continuation. The design is compatible with a future collapse to full CPS if desired.

### 3.3 Thorin/MimIR

Thorin (CGO 2015) and its successor MimIR (arXiv 2411.07443, POPL 2025) take the full CPS approach
described above. In Thorin, a function is a continuation; there are no back-edges in the IR; loops
are recursive continuations; and the IR is a DAG by construction. dgen's design is converging on
the same structural properties through the `%self` mechanism, while retaining a more familiar
block/function surface syntax.

---

## 4. Alternatives Considered

### 4.1 Symbol Tables (Rejected)

Using a symbol table for function and label references is the conventional approach (LLVM IR, early
MLIR). It is rejected here because it makes references non-values, prevents use-def graph
completeness, and requires a separate name resolution mechanism. See Section 1.4.

### 4.2 `func.letrec` Group Op

A `func.letrec` op that introduces a set of mutually recursive functions simultaneously, with all
signatures in scope for all bodies, is the symmetric alternative to the nesting approach for mutual
recursion. It avoids nominating a root function. It is not adopted as a primitive today because
nesting + `func.recursive` covers the same ground and is simpler. `func.letrec` can be added as
syntactic sugar later.

### 4.3 Equirecursive Types for `%self`

The type of `%self` could in principle be an equirecursive type: `T = (T, ...) → R`. This makes
`T` equal to its own unfolding and requires coinductive type equality, which is expensive and
unusual. It is rejected in favor of nominal isorecursion (see Section 2.1), where the declared type
is the fixpoint and self-reference is broken by the declaration.

### 4.4 Implicit Capture (Dominance-Based)

MLIR's `func.func` regions use dominance-based implicit capture: a value defined in an outer region
is automatically visible in all nested regions without declaration. This is convenient for
sequential code but makes each block's dependency set non-local, complicating independent analysis.
Explicit capture is preferred for dgen's analysis and meta-compilation goals.

### 4.5 Sea of Nodes

Sea of Nodes (Click & Paleczny, PLDI 1995) delays all scheduling to the backend by giving pure
nodes no control edges, letting the scheduler place them anywhere in the legal window. dgen's
ambient node design achieves the same effect for ops with no block-argument dependencies, while
using explicit capture to pin non-ambient ops. The scheduling heuristics in Section 2.5 are derived
directly from the SoN literature.

---

## 5. References

Click, C. and Paleczny, M. (1995). A simple graph-based intermediate representation. *ACM SIGPLAN
Workshop on Intermediate Representations (IR '95)*.

Kennedy, A. (2007). Compiling with continuations, continued. *ACM SIGPLAN International Conference
on Functional Programming (ICFP '07)*.

Leissa, R., Boesche, N., Hack, S., et al. (2015). Graph-based higher-order intermediate
representation. *CGO 2015*. (Thorin)

Müller, M., et al. (2025). MimIR: A Higher-Order Intermediate Representation Based on
Continuation-Passing Style. *POPL 2025*. arXiv:2411.07443.

Appel, A. W. (1992). *Compiling with Continuations*. Cambridge University Press.

Lattner, C. and Adve, V. (2004). LLVM: A Compilation Framework for Lifelong Program Analysis and
Transformation. *CGO 2004*.

MLIR Language Reference. https://mlir.llvm.org/docs/LangRef/
