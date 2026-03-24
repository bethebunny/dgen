# dgen IR Design: Scope, Capture, and Use-Def Cycles

*dgen Compiler Framework — IR Design Series*

---

## Abstract

This document describes the recommended approach to scope, value capture, and use-def cycle
management in the dgen IR. The central thesis is that the current IR lacks a formal invariant
around value capture: ops inside a block may freely reference values from enclosing blocks without
declaring them, creating implicit dependencies that are invisible in the use-def graph. This
**implicit capture** leads to a fragile `walk_ops` implementation (patched with `isinstance`
guards and circular imports) and complex lowering passes that must manually track cross-block
dependencies. The recommendation is a **closed-block invariant**: every block is a closed term
whose free variables are explicitly declared as block arguments. With this invariant, `walk_ops`
becomes a simple DAG walk with no special cases, and block-level analysis, inlining, and
scheduling become sound in isolation. The document also addresses use-def cycles from recursive
functions and loop back-edges, which are eliminated by a `%self` mechanism at the block boundary.

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

The use-def edges form a cycle: `%i` → `%i_next` → `llvm.br` → `%loop` → `%i`. This complicates
analysis, scheduling, and hash-consing, all of which prefer or require a DAG structure.

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
edges. Recursion is handled by dedicated op forms described in Section 3.

### 1.5 Problem: Fragile IR Implementation

The absence of a formal capture invariant creates fragility throughout the implementation. The
clearest symptom is `walk_ops`:

```python
for _, param in value.parameters:
    from dgen.dialects.builtin import FunctionOp   # circular-import workaround
    if not isinstance(param, FunctionOp):
        visit(param)
```

This guard exists because `call<callee: Function>(args)` stores the callee—a module-level
`FunctionOp`—as a compile-time parameter. Without the guard, `walk_ops` would follow the parameter
edge into the callee's entire body, surfacing that function's internal ops as members of the
calling block's op set. This is incorrect: those ops belong to a different block's scope, but the
IR provides no structural way to detect the boundary.

The same fragility appears in lowering passes. `toy_to_affine.py` and `affine_to_llvm.py` generate
loop body blocks that reference outer-scope values (array pointers, loop bounds) without threading
them through block arguments. This works today only because the lowering and execution paths happen
to process the implicit captures correctly. Any new pass that analyses these blocks for
dependencies—a scheduler, an inliner, a verifier—must independently reconstruct the implicit
capture set, or add yet another special case.

The root cause is the absence of a checked invariant: there is no mechanism that enforces block
closure, so violations accumulate silently and are patched one by one.

---

## 2. Proposed Invariants

This section states the desired end state: a set of invariants that, if enforced, make the IR
well-behaved by construction. Section 3 describes how to achieve this state.

### 2.1 Closed Blocks and the DAG Property

**Closed-block invariant.** A block B is *closed* if every op reachable from B.result satisfies
one of the following:

1. It is an **ambient node**: it has no transitive dependencies on any block's arguments
   (constants, global references, function definitions with no external operands, etc.).
2. Its transitive block-argument dependencies are entirely within B.args.

Equivalently: if op V is reachable from B.result and V transitively depends on the arguments of
some block B′, then B′ must be B itself. Cross-scope references—ops that depend on arguments of
a *different* block—must be threaded explicitly through the argument lists on the path between
blocks.

**DAG property.** The use-def graph is always a directed acyclic graph. Back-edges from
recursion and loops are broken at block-argument boundaries: a recursive call passes `%self` (a
block argument, a leaf in the graph) as the callee, not a direct edge back to the defining op.

### 2.2 walk_ops: The Clean Invariant

With the closed-block and DAG invariants enforced, `walk_ops` on any well-formed IR value has a
simple, exception-free specification:

> Given value V, `walk_ops(V)` visits ops by following **operands**, **parameter values**, the
> **result type** (if it is an SSA value), and **block argument types**. No special cases. No
> `isinstance` guards. The traversal terminates because the use-def graph is a DAG.

**The locality property.** `walk_ops(B.result)` produces exactly:
- Ops whose transitive block-argument dependencies are entirely within B.args, and
- Ambient ops (no block-argument dependencies) reachable via the use-def graph.

It will never surface ops whose scope belongs to a different block. This is what makes block-level
analysis, inlining, and scheduling sound in isolation: a block's complete dependency set is
readable directly from its argument list.

### 2.3 Motivating Example: Before and After

The following pair shows a loop under the current implicit-capture style and under the
closed-block invariant.

**Current (implicit capture):**

```
%src : llvm.Ptr = llvm.alloca<6>()
%5 : Nil = affine.for<0, 4>() (%i: Index):
    %val : F64 = affine.load(%src, [%i])    // %src captured implicitly from outer block
```

`walk_ops` on the `affine.for` result does not see `%src` as a dependency of the loop body—yet
the loop body functionally depends on it. The `llvm.alloca` op belongs to the outer block, and
the loop body references it without declaration. This dependency is invisible to any tool that
inspects block structure.

**Under the closed-block invariant:**

```
%src : llvm.Ptr = llvm.alloca<6>()
%5 : Nil = affine.for<0, 4>() (%src: llvm.Ptr, %i: Index):
    %val : F64 = affine.load(%src, [%i])
```

`walk_ops` on this block's result immediately encounters `%src` and `%i` as block arguments
(leaves). There is nothing to special-case: the block's complete dependency set is manifest in
its argument list. Any tool that operates on this block—a scheduler, a verifier, an inliner—reads
its inputs directly from the argument list without reconstructing implicit capture.

---

## 3. Recommendations

### 3.1 `func.recursive`: Eliminating Use-Def Cycles in Recursive Functions

A `func.recursive` op introduces `%self` as its first block argument, typed as `Function<T>` where
`T` is the function's return type. The body may call `%self` to recurse. Crucially, `%self` is a
fixed input defined at the block boundary, not fed back from the recursive call's result. The
recursive call produces a fresh return value. Therefore:

- Execution cycles (the function calls itself).
- The use-def graph remains a DAG (`%self` is never redefined by the recursion).

The op produces a `Function<T>` value—a first-class callable that can be stored, passed, and
called like any other value.

```
// %self is defined once at the block boundary; the recursive call
// produces a fresh %r. No use-def cycle.
%factorial : Function<Index> = func.recursive() (%self: Function<Index>, %n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond) ():
        %base : Index = 1
    else (%n: Index, %self: Function<Index>):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %r : Index = call(%self, [%n1])    // %self is a block arg — leaf in the DAG
        %res : Index = llvm.mul(%n, %r)
```

The `else` block declares `%n` and `%self` as its own block arguments under the closed-block
invariant: they are defined in the outer block (`func.recursive`'s body) and must be explicitly
threaded. The `if` op is responsible for passing the captured values to each branch; see Section
3.4 for the required update to `if`. The constant `%one` is an ambient node and is redeclared
in the `else` block rather than threaded.

The type of `%self` is `Function<Index>`, resolved at definition time. This is **nominal
isorecursion**: the declared type is the fixpoint, and the self-reference is broken by the
declaration boundary rather than by an explicit roll/unroll.

Calling `%factorial` from the outside:

```
%fn_ref : Function<Index> = func_ref<%factorial>()    // ambient — wraps the static reference
%five : Index = 5
%result : Index = call(%fn_ref, [%five])
```

The `func_ref` op is an ambient node that wraps a module-level function into a `Function<T>`
runtime value. See Section 3.6 for the three options for how `call` and self-reference interact.

#### Mutual Recursion via Nesting

Mutual recursion does not require a separate `func.letrec` primitive. One function may be defined
inside the other, capturing the outer function's `%self` as an explicit argument:

```
%even : Function<Index> = func.recursive() (%self: Function<Index>, %n: Index):
    // %odd is defined inside %even; it captures %self via its own argument list
    %odd : Function<Index> = function() (%even_ref: Function<Index>, %m: Index):
        %one : Index = 1
        %m1 : Index = subtract_index(%m, %one)
        %res : Index = call(%even_ref, [%m1])
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"eq">(%n, %zero)
    %result : Index = if(%cond) ():
        %t : Index = 1
    else (%self: Function<Index>, %n: Index):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        // %odd is ambient (no block-arg deps) — no threading needed
        %res : Index = call(%odd, [%self, %n1])
```

`%odd` is a `function` op with no external operands. Its transitive dependencies include only its
own block arguments (`%even_ref`, `%m`), not anything from `%even`'s outer scope. Therefore `%odd`
is an **ambient node** and may be referenced from the `else` block without threading. `%self` and
`%n`, however, are block arguments of `%even`'s body and must be declared in the `else` block's
argument list.

`%odd` is not self-recursive, so it does not need `func.recursive`. The mutual recursion cycle is
broken by nesting: `%even` owns `%odd`, and `%odd` calls back through the captured reference
`%even_ref`. The use-def graph is a DAG throughout.

The trade-off is asymmetry: one function must be nominated as the owner. For larger mutual
recursion groups (A → B → C → A), all inner functions are nested under the root with captured
references threaded explicitly. A `func.letrec` group op can be added later as syntactic sugar
for the symmetric case if warranted.

### 3.2 Label `%self`: Eliminating Use-Def Cycles in Loops

The loop back-edge problem is symmetric to the recursion problem, and the solution is symmetric.
Every `llvm.label` block receives `%self : llvm.Label` as its first argument. A loop is expressed
as a block that branches to `%self` with updated arguments:

```
// %main function containing a count-down loop
%main : Function<Nil> = function() ():
    %loop : llvm.Label = llvm.label() (%self: llvm.Label, %i: Index):
        %zero : Index = 0
        %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
        %exit : llvm.Label = llvm.label() ():
            %done : Index = 0
        %one : Index = 1
        %i_next : Index = subtract_index(%i, %one)
        // pass %self and %i_next as args to %self (the back-edge)
        %_ : Nil = llvm.cond_br(%cond, %self, %exit, [%self, %i_next], [])
    %ten : Index = 10
    %_ : Nil = llvm.br(%loop, [%loop, %ten])    // initial call: pass %loop as its own %self
```

Comparing this to the back-edge version from Section 1.2: the use-def cycle
`%i → %i_next → llvm.br → %i` is eliminated. `%self` is fixed at the block boundary, and
`%i_next` is an argument at the call site, not a redefinition of `%i`.

The initial call `llvm.br(%loop, [%loop, %ten])` passes `%loop` as the first argument, binding it
to `%self` inside the body. This is valid because `%loop` is defined in the enclosing block
(the function body) and is an ambient node from the inner blocks' perspective—it has no
block-argument transitive dependencies. Subsequent recursive iterations call `llvm.br(%self, ...)`
rather than `llvm.br(%loop, ...)`, using the block argument `%self` rather than the outer-scope
`%loop`. Under the closed-block invariant, referring to `%loop` from inside the body would be an
implicit capture violation; `%self` is the correctly-threaded reference.

#### Sibling Blocks and Mutual Block Recursion

When two sibling blocks mutually recurse (as in a loop split across blocks), the capture
discipline requires the label to be threaded explicitly. The pattern is identical to the
`even`/`odd` nesting:

```
%main : Function<Nil> = function() ():
    %loop : llvm.Label = llvm.label() (%self: llvm.Label, %i: Index):
        %zero : Index = 0
        %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
        %exit : llvm.Label = llvm.label() ():
            %done : Index = 0
        // pass %self to %body so it can call back to the loop header
        %_ : Nil = llvm.cond_br(%cond, %body, %exit, [%self, %i], [])
    %body : llvm.Label = llvm.label() (%loop_ref: llvm.Label, %i: Index):
        %one : Index = 1
        %i_next : Index = subtract_index(%i, %one)
        %_ : Nil = llvm.br(%loop_ref, [%loop_ref, %i_next])
    %ten : Index = 10
    %_ : Nil = llvm.br(%loop, [%loop, %ten])
```

`%loop` passes `%self` to `%body` as an explicit argument `%loop_ref`. `%body` calls back through
`%loop_ref`. The entire graph is a DAG. Neither `%body` nor any op inside it holds a direct
reference to `%loop`—only to the block argument that carries `%loop`'s value.

#### Blocks and Functions: A Unified View

With `%self : llvm.Label` on every label block, the distinction between a `func.recursive` and a
looping block largely collapses: both are callable things that receive `%self` and can recurse
through it. The remaining distinction is that a `function` has a return type (its body's result
propagates out), while a label is a continuation—control transfers via `llvm.br` and there is no
return value. This is explored further in Section 4.

### 3.3 Explicit Capture

The explicit capture discipline is: **a block's free variables must be declared as its
arguments**. More precisely: if op V is reachable from block B.result, and V transitively depends
on the arguments of some block B′ other than B itself, then V must be threaded through the block
argument lists on all paths between B′ and B.

This makes each block a **closed term**: its meaning is determined entirely by its arguments.
Benefits:

- Block-level analysis, inlining, and partial evaluation are sound in isolation.
- The full data dependencies of each block are manifest in its argument list.
- No implicit environment or dynamic scope to track.
- Aligns with scheduling requirements: the scheduler can determine a block's inputs by inspection.

### 3.4 Migrating Existing Ops with Implicit Capture

The current IR has two ops that produce blocks with implicit capture: `builtin.if` and
`affine.for`. Both allow their body blocks to reference values from the enclosing scope without
threading them through block arguments.

**`builtin.if`**: The `then_body` and `else_body` blocks currently take no arguments. Any
outer-scope value used inside a branch is implicitly captured. Under explicit capture, the `if` op
must be updated to accept per-branch argument lists (analogous to `llvm.cond_br`'s `true_args` and
`false_args`), and outer values must be declared as block arguments:

```
// Current (implicit capture):
%alloc : llvm.Ptr = llvm.alloca<6>()
%result : F64 = if(%cond) ():
    %loaded : F64 = llvm.load(%alloc)    // %alloc captured implicitly
else ():
    %c : F64 = 0.0

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
- not an ambient node (i.e., it has no block-argument transitive dependencies).

Running this verifier as a post-pass assertion makes capture regressions visible immediately
rather than silently producing unsound IR.

### 3.5 Ambient Nodes

The explicit capture discipline has a natural exemption: ops that have no block-argument
dependencies at all. We call these **ambient nodes**. An ambient node cannot transitively depend
on any block's arguments—there is nothing in its transitive input set that could cause a
cross-block capture violation. Therefore, ambient nodes may be referenced from any block without
declaration.

The canonical ambient nodes are:

- Constants and integer/float literals (`%c : Index = 0`, `%f : F64 = 3.14`).
- Global references (immutable global values known at compile time).
- Function and label ops with no external operands (`function() (...)`, `llvm.label() (...)`).
- Type metadata and other purely static values.

More generally, an op's **scope** is determined by the deepest block whose arguments it
transitively depends on. Ambient nodes have no such block—they float above the entire block
structure. This is not a special case; it is the natural consequence of the capture discipline
applied to nodes with an empty dependency set.

Ambient node sharing is the foundation for **hash-consing**: two ambient subgraphs with identical
structure and inputs are semantically equivalent and can be unified to a single node. This supports
implicit common subexpression elimination without a separate CSE pass—a desirable property for
dgen's meta-compilation workloads.

### 3.6 Tension with `call` and the Callee Parameter

The `func.recursive` and `%self` designs require calling a value passed as a block argument. This
creates tension with the current `call` design, where the callee is a compile-time parameter:

```
op call<callee: Function>(args: List) -> Type
```

ASM: `%result : T = call<%fn>([%arg1, %arg2])`

Currently `call`'s callee must resolve to a `FunctionOp` at compile time. This works for static
calls between module-level functions. It does not directly work for:

1. **Recursive calls**: `%self` is a block argument, not a module-level op—but in practice this
   already works in the parser, since block arguments can appear in parameter position.
2. **Higher-order functions**: a function value passed as a block argument of type `Function<T>`
   can appear as a parameter.
3. **walk_ops**: when `call<%fn>` has a `FunctionOp` as its parameter, `walk_ops` descends into
   that function's body. This requires the `isinstance(param, FunctionOp)` guard.

Three design options address this tension. Each is shown with a **recursive function** example and
a **first-class (higher-order) function** example.

---

#### Option A: Move callee to operand

Change `call` so that the callee is a runtime operand: `call(callee, args)`. For static calls to
module-level functions, a `func_ref` ambient op wraps the reference:

```
op call(callee: Function<T>, args: List<T>) -> T
op func_ref<fn: Function>() -> Function<T>    // ambient — no block-arg dependencies
```

**Recursive function:**
```
%factorial : Function<Index> = func.recursive() (%self: Function<Index>, %n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond) ():
        %base : Index = 1
    else (%n: Index, %self: Function<Index>):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %r : Index = call(%self, [%n1])     // %self is a runtime operand
        %res : Index = llvm.mul(%n, %r)

// Calling from outside:
%fn_ref : Function<Index> = func_ref<%factorial>()
%five : Index = 5
%result : Index = call(%fn_ref, [%five])
```

**First-class function:**
```
%apply : Function<Index> = function() (%f: Function<Index>, %n: Index):
    %result : Index = call(%f, [%n])    // uniform — identical syntax to the recursive case
```

`walk_ops` on `call(%self, [%n1])` follows the operand edge to `%self`, which is a
`BlockArgument` — a leaf. `walk_ops` on `call(%fn_ref, [%five])` follows the operand edge to
`func_ref<%factorial>()`, which is an ambient node with no block-argument dependencies — also a
leaf. The `FunctionOp` guard in `walk_ops` disappears entirely; there is no path from inside a
block's op graph back into another function's body.

**Trade-offs:** Requires changing the `call` op signature and adding `func_ref`. All call sites
(static and dynamic) become uniform. The `func_ref` op provides a clean place to express "I am
taking a reference to a module-level function as a value."

---

#### Option B: Keep callee as parameter; use the op's own identity for self-reference

Leave `call<callee: Function>(args)` unchanged. For `func.recursive`, the body refers to the
op being defined directly by its SSA name — the recursive function is ambient (no external
operand dependencies), so this reference is valid under the closed-block invariant:

**Recursive function:**
```
%factorial : Function<Index> = func.recursive() (%n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond) ():
        %base : Index = 1
    else (%n: Index):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        %r : Index = call<%factorial>([%n1])    // %factorial used as its own parameter
        %res : Index = llvm.mul(%n, %r)

// Calling from outside — unchanged:
%five : Index = 5
%result : Index = call<%factorial>([%five])
```

Note: `%self` is not a block argument under this option — it is not threaded at all. The function
refers to itself directly.

**First-class function:**
```
%apply : Function<Index> = function() (%f: Function<Index>, %n: Index):
    %result : Index = call<%f>([%n])    // %f is a block arg used as parameter — works today
```

**Trade-offs:** No change to `call`, no `func_ref` op needed. However, `call<%factorial>` inside
`%factorial`'s body creates a **parameter-level cycle**: `%factorial` contains an op whose
parameter IS `%factorial`. This violates the DAG property from Section 2.1. `walk_ops` handles
it via the `visited` set (no infinite loop), but the IR is technically not a DAG for recursive
functions. The FunctionOp guard is replaced by implicit cycle tolerance, not removed. For first-
class function values that are not self-recursive, the design is unchanged and clean; the DAG
violation is confined to the recursive case.

---

#### Option C: Block parameters — a new binding form

Introduce **block parameters**: a new syntactic position in a block declaration that binds a
value at the block boundary without requiring the caller to pass it. Block parameters are
automatically provided by the runtime/compiler when the block is invoked. The first block of an
op is named explicitly (currently the first block name is elided, a "weird carveout" in the
formatter):

```
op_name() block_name<%param: Type>(arg1: T1, arg2: T2):
    ...
```

`%param` is a block parameter: it is in scope throughout the block, available to sub-blocks via
threading, but is never passed by callers. For `func.recursive` and `llvm.label`, the compiler
automatically binds the block parameter to the op's own callable identity.

**Recursive function:**
```
%factorial : Function<Index> = func.recursive() body<%self: Function<Index>>(%n: Index):
    %one : Index = 1
    %cond : llvm.Int<1> = llvm.icmp<"sle">(%n, %one)
    %result : Index = if(%cond) ():
        %base : Index = 1
    else (%n: Index, %self: Function<Index>):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        // %self threaded to else block as an ordinary block arg; called as parameter
        %r : Index = call<%self>([%n1])
        %res : Index = llvm.mul(%n, %r)

// Calling from outside — caller does not pass %self:
%five : Index = 5
%result : Index = call<%factorial>([%five])
```

**Loop with block parameter:**
```
%main : Function<Nil> = function() ():
    %loop : llvm.Label = llvm.label() body<%self: llvm.Label>(%i: Index):
        %zero : Index = 0
        %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
        %exit : llvm.Label = llvm.label() ():
            %done : Index = 0
        %one : Index = 1
        %i_next : Index = subtract_index(%i, %one)
        // caller does not need to pass %self — it is auto-bound
        %_ : Nil = llvm.cond_br(%cond, %self, %exit, [%i_next], [])
    %ten : Index = 10
    %_ : Nil = llvm.br(%loop, [%ten])    // initial call — no %self arg needed
```

**First-class function:**
```
%apply : Function<Index> = function() (%f: Function<Index>, %n: Index):
    %result : Index = call<%f>([%n])    // unchanged — non-recursive, no block param
```

**Trade-offs:** No change to `call`. Callers are simpler (no need to pass the label/function as
its own first argument). The IR remains a strict DAG: `%self` as a block parameter is a leaf,
never a back-edge. However, this option requires introducing a new concept into the IR (block
parameters), updating the formatter and parser to handle the named-block syntax, and specifying
how the compiler auto-binds block parameters for each op type. Additionally, `%self` when
threaded to sub-blocks becomes an ordinary block argument — the distinction between block
parameter and block argument complicates reasoning about what is and is not automatically provided.

---

**Recommendation.** Option A is the most uniform and best satisfies the DAG invariant from
Section 2.1. The `func_ref` ambient op provides a clean mechanism for taking a reference to a
static function, and the single `call(callee, args)` form handles all cases without exception.
Option C is architecturally appealing but introduces new complexity. Option B is the cheapest to
implement but violates the DAG property and does not eliminate the structural special case in
`walk_ops`.

### 3.7 Scheduling Shared Pure Subgraphs

After explicit capture is enforced, some ops are still "free-floating": their scope is not pinned
to a specific block because all of their dependencies are ambient. These ops have a legal
scheduling window:

- **Earliest**: the block that dominates all of the op's dependencies. For ambient ops, this is
  the function entry.
- **Latest**: the block that post-dominates all of the op's uses. You cannot defer past the point
  where any consumer needs the value.

Any placement within `[earliest, latest]` is **sound**. Soundness requires only the use-def graph
and the dominator relation; no loop analysis is needed.

#### Heuristics

Within the legal window, placement heuristics optimize for performance:

- **Rematerialization** (for trivially cheap ops): constants and literals are emitted at each use
  site. Their live range is zero; no register allocation pressure.
- **Loop hoisting** (for non-trivial pure ops): an op whose uses are all inside a loop but whose
  dependencies are all outside it should be placed at the loop's pre-header—computed once, not on
  every iteration.
- **Placement at LCA** (default): for ops that cannot be further hoisted, the LCA of all use
  sites in the dominator tree is the natural placement.

#### Loop Detection

Loop detection is required only for the hoisting heuristic, not for soundness. With the
`%self : llvm.Label` design, loops are structurally manifest: a block is a loop if and only if its
body contains a `llvm.br(%self, ...)` (Option A / B) or `llvm.cond_br(..., %self, ...)`. Loop
nesting depth is read directly from block nesting. No dominator analysis is needed to identify
loop structure.

#### Division of Labor with LLVM

dgen currently targets LLVM IR as its exit dialect. The recommended strategy for dgen's scheduler
is:

- Produce **sound** LLVM IR: legal placement, correct use-def threading, nothing more required.
- Defer loop-invariant code motion, register-pressure-aware scheduling, and other optimizations
  to LLVM.
- Revisit dgen-side scheduling when dgen's higher-level semantic knowledge (purity by
  construction, specialization decisions) can demonstrably improve on what LLVM can infer.

### 3.8 `walk_ops`: End State

Under the closed-block and DAG invariants, `walk_ops` reduces to a straightforward recursive DAG
walk:

```python
def visit(value):
    if isinstance(value, list):
        for item in value: visit(item)
        return
    if isinstance(value, Type):
        for _, p in value.__params__: visit(p)
        return
    if not isinstance(value, Value) or id(value) in visited:
        return
    visited.add(id(value))
    if not isinstance(value, Op):
        return
    for _, operand in value.operands: visit(operand)
    for _, param in value.parameters: visit(param)
    visit(value.type)
    for _, block in value.blocks:
        for arg in block.args: visit(arg.type)
    order.append(value)
```

No `isinstance(param, FunctionOp)` guard. No circular import. The `FunctionOp` guard is
unnecessary because under Option A, `FunctionOp` values never appear as `call` parameters—they
are only referenced through `func_ref` ambient ops, which are leaves. Under Option C they are
similarly never reached via parameter edges inside a recursive body. Under Option B the visited
set prevents infinite loops but the DAG violation remains, as discussed in Section 3.6.

The block argument type traversal (`for arg in block.args: visit(arg.type)`) remains correct
and necessary when a block argument's type is itself an SSA value (a dependent type). Under
explicit capture, such type dependencies are either ambient or already explicit block arguments;
no special cross-scope handling is required.

The end state is a simple recursive DAG walk with no `isinstance` guards and no special cases.
The structure is guaranteed by the capture discipline rather than enforced by ad hoc checks.

---

## 4. Relationship to CPS

The design described above is closely related to Continuation-Passing Style (CPS). The
correspondences are direct:

| dgen IR concept                        | CPS equivalent                              |
|----------------------------------------|---------------------------------------------|
| `%self : llvm.Label`                   | Continuation argument                       |
| `llvm.br(%self, [%i_next])`            | Tail call to a continuation                 |
| Block with explicit capture            | Closed lambda abstraction                   |
| `function<R>` with return type `R`     | Function taking explicit return continuation|
| `llvm.br` to another label             | Tail call to another continuation           |
| Block argument list                    | Lambda parameter list                       |

### 4.1 Is CPS Always Flat?

"Flat" in CPS refers to the *dynamic* structure: there is no call stack, because every call is a
tail call and continuations are just jumps. It does not imply that the *lexical* structure is flat.
CPS-based IRs (MLton, SML/NJ's FLINT, Thorin/MimIR) have lexical nesting for scope and capture;
the flatness property means that dynamically, control never returns—it always transfers forward.

dgen's IR is flat in the CPS sense (`llvm.br` is a jump, not a stack-pushing call) but lexically
nested (blocks can be nested inside ops). This combination is valid and useful: dynamic flatness
gives efficient implementation, and lexical nesting makes scope manifest.

### 4.2 The `func` / `block` Distinction

In fully continuation-passing style (as in Thorin/MimIR), even the return from a function is
modeled as calling a return continuation argument `%return : llvm.Label`. This collapses the
`func`/`block` distinction entirely: everything is a continuation. The only distinction between a
"function" and a "basic block" is calling convention and ABI.

dgen does not go this far today: `function` retains a return type rather than an explicit return
continuation. The design is compatible with a future collapse to full CPS if desired.

### 4.3 Thorin/MimIR

Thorin (CGO 2015) and its successor MimIR (arXiv 2411.07443, POPL 2025) take the full CPS approach
described above. In Thorin, a function is a continuation; there are no back-edges in the IR; loops
are recursive continuations; and the IR is a DAG by construction. dgen's design is converging on
the same structural properties through the `%self` mechanism, while retaining a more familiar
block/function surface syntax.

---

## 5. Alternatives Considered

### 5.1 Symbol Tables (Rejected)

Using a symbol table for function and label references is the conventional approach (LLVM IR, early
MLIR). It is rejected here because it makes references non-values, prevents use-def graph
completeness, and requires a separate name resolution mechanism. See Section 1.4.

### 5.2 `func.letrec` Group Op

A `func.letrec` op that introduces a set of mutually recursive functions simultaneously, with all
signatures in scope for all bodies, is the symmetric alternative to the nesting approach for mutual
recursion. It avoids nominating a root function. It is not adopted as a primitive today because
nesting + `func.recursive` covers the same ground and is simpler. `func.letrec` can be added as
syntactic sugar later.

### 5.3 Equirecursive Types for `%self`

The type of `%self` could in principle be an equirecursive type: `T = (T, ...) → R`. This makes
`T` equal to its own unfolding and requires coinductive type equality, which is expensive and
unusual. It is rejected in favor of nominal isorecursion (see Section 3.1), where the declared
type is the fixpoint and self-reference is broken by the declaration.

### 5.4 Implicit Capture (Dominance-Based)

MLIR's `func.func` regions use dominance-based implicit capture: a value defined in an outer
region is automatically visible in all nested regions without declaration. This is convenient for
sequential code but makes each block's dependency set non-local, complicating independent
analysis. Explicit capture is preferred for dgen's analysis and meta-compilation goals.

### 5.5 Sea of Nodes

Sea of Nodes (Click & Paleczny, PLDI 1995) delays all scheduling to the backend by giving pure
nodes no control edges, letting the scheduler place them anywhere in the legal window. dgen's
ambient node design achieves the same effect for ops with no block-argument dependencies, while
using explicit capture to pin non-ambient ops. The scheduling heuristics in Section 3.7 are
derived directly from the SoN literature.

---

## 6. References

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
