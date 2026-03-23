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
%loop = block(%i : i64) {
    %cond = gt(%i, 0)
    cond_br %cond, %body, %exit
}
%body = block {
    %i_next = sub(%i, 1)
    br %loop(%i_next)     // back-edge: %i_next feeds back to %i
}
```

The use-def edges form a cycle: `%i` → `%i_next` → `br` → `%i`. This complicates analysis,
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
// produces a fresh %result. No use-def cycle.
func.recursive @factorial(%self : fn(i64) -> i64, %n : i64) -> i64 {
    %one  = const 1
    %cond = le(%n, %one)
    cond_br %cond, %base, %rec

    %base = block { ret %one }

    %rec = block {
        %n1  = sub(%n, %one)
        %r   = call %self(%n1)    // uses %self (fixed input), produces fresh %r
        %res = mul(%n, %r)
        ret %res
    }
}
```

The type of `%self` is simply the function's declared signature, resolved at definition time. This
is **nominal isorecursion**: the declared type is the fixpoint, and the self-reference is broken by
the declaration boundary rather than by an explicit roll/unroll.

#### Mutual Recursion via Nesting

Mutual recursion does not require a separate `func.letrec` primitive. One function may be defined
inside the other, capturing the outer function's `%self` as an explicit argument:

```
func.recursive @even(%self : fn(i64) -> bool, %n : i64) -> bool {
    // odd is defined inside even; it captures %self as %even_ref
    func @odd(%even_ref : fn(i64) -> bool, %n : i64) -> bool {
        %one  = const 1
        %n1   = sub(%n, %one)
        %res  = call %even_ref(%n1)
        ret %res
    }
    %zero = const 0
    %cond = eq(%n, %zero)
    cond_br %cond, %base_true, %rec

    %base_true = block { %t = const true; ret %t }
    %rec       = block {
        %one = const 1
        %n1  = sub(%n, %one)
        %res = call %odd(%self, %n1)    // pass %self so odd can call back
        ret %res
    }
}
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
Every block receives `%self : Label<(...)>` as its first argument, typed with its own argument
types. A loop is expressed as a block that calls `%self` with updated arguments:

```
// No back-edge in the use-def graph. %self is a fixed input.
// %i_next is an argument to the call, not fed back into a definition.
%loop = block(%self : Label<(i64)>, %i : i64) {
    %zero = const 0
    %cond = gt(%i, %zero)
    cond_br %cond, %body, %exit

    %body = block {
        %one    = const 1
        %i_next = sub(%i, %one)
        br %self(%i_next)    // recursive call through %self, not a back-edge
    }
}
```

Comparing this to the back-edge version from Section 1.2: the use-def cycle
`%i → %i_next → br → %i` is eliminated. `%self` is fixed at the block boundary, and `%i_next` is
an argument at the call site, not a definition of `%i`.

#### Sibling Blocks and Mutual Block Recursion

When two sibling blocks mutually recurse (as in a loop split across blocks), the capture discipline
requires the label to be threaded explicitly. The pattern is identical to the `even`/`odd` nesting:

```
%loop = block(%self : Label<(i64)>, %i : i64) {
    %zero = const 0
    %cond = gt(%i, %zero)
    // pass %self to %body so it can call back
    cond_br %cond, %body(%self, %i), %exit
}

%body = block(%loop_label : Label<(i64)>, %i : i64) {
    %one    = const 1
    %i_next = sub(%i, %one)
    br %loop_label(%i_next)
}
```

`%loop` passes `%self` to `%body` as an explicit argument `%loop_label`. `%body` calls back through
`%loop_label`. The entire graph is a DAG.

#### Blocks and Functions: A Unified View

With `Label<%self>` on every block, the distinction between a `func.recursive` and a looping block
largely collapses: both are callable things that receive `%self` and can recurse through it. The
remaining distinction is that a `func` has a return type, while a block is a continuation — it does
not return; control transfers via `br`. This is explored further in Section 3.

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

- Constants and integer/float literals (`const 0u8`, `const 3.14`).
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
`%self : Label` design, loops are structurally manifest: a block is a loop if and only if its body
contains a `br %self(...)`. Loop nesting depth is read directly from block nesting. No dominator
analysis is needed to identify loop structure; it is visible in the use-def graph via `%self`
values.

#### Division of Labor with LLVM

dgen currently targets LLVM IR as its exit dialect. LLVM's optimization pipeline (LICM, GVN, loop
vectorization, instruction scheduling, register allocation) provides a strong baseline. The
recommended strategy for dgen's scheduler is therefore:

- Produce **sound** LLVM IR: legal placement, correct use-def threading, nothing more required.
- Defer loop-invariant code motion, register-pressure-aware scheduling, and other optimizations to
  LLVM.
- Revisit dgen-side scheduling when dgen's higher-level semantic knowledge (purity by construction,
  specialization decisions) can demonstrably improve on what LLVM can infer from the lowered form.

---

## 3. Relationship to CPS

The design described above is closely related to Continuation-Passing Style (CPS). The
correspondences are direct:

| dgen IR concept                   | CPS equivalent                              |
|-----------------------------------|---------------------------------------------|
| `%self : Label<(i64)>`            | Continuation argument `cont(i64)`           |
| `br %self(%i_next)`               | Tail call to a continuation                 |
| Block with explicit capture       | Closed lambda abstraction                   |
| `func` with return type `R`       | Function taking explicit return continuation|
| `br` to another label             | Tail call to another continuation           |
| Block argument list               | Lambda parameter list                       |

### 3.1 Is CPS Always Flat?

"Flat" in CPS refers to the *dynamic* structure: there is no call stack, because every call is a
tail call and continuations are just jumps. It does not imply that the *lexical* structure is flat.
CPS-based IRs (MLton, SML/NJ's FLINT, Thorin/MimIR) have lexical nesting for scope and capture;
the flatness property means that dynamically, control never returns — it always transfers forward.

dgen's IR is flat in the CPS sense (`br` is a jump, not a stack-pushing call) but lexically nested
(blocks can be nested inside ops). This combination is valid and useful: dynamic flatness gives
efficient implementation, and lexical nesting makes scope manifest.

### 3.2 The `func` / `block` Distinction

In fully continuation-passing style (as in Thorin/MimIR), even the return from a function is
modeled as calling a return continuation argument `%return : Label<(R)>`. This collapses the
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
