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
2. Its transitive block-argument dependencies are entirely within `B.parameters ∪ B.args ∪
   B.captures`.

A block has three kinds of declared inputs:

- **`parameters`**: compile-time values bound once by the op that owns the block (e.g. `%self`
  for loop-header labels). Callers never pass them. Syntax: `block_name<%param: Type>`.
- **`args`**: runtime values passed by callers at every branch site, generating phi nodes in
  the lowered CFG. Syntax: `block_name(%arg: Type)`.
- **`captures`**: outer-scope values referenced directly. They are leaves in `walk_ops` (the
  walk stops at capture boundaries) and do not generate phi nodes. Syntax:
  `block_name(...) captures(%val)`.

Captures must **chain**: if a child block captures a value, every block between the value's
definition and the child must also capture it. This ensures `replace_uses` on any block
correctly propagates through its nested blocks' captures.

Cross-scope references that are not ambient must appear in one of these three lists.

**DAG property.** The use-def graph is always a directed acyclic graph. Back-edges from
recursion and loops are broken at block-argument boundaries: a recursive call passes `%self` (a
block parameter, a leaf in the graph) as the callee, not a direct edge back to the defining op.

### 2.2 walk_ops: The Clean Invariant

With the closed-block and DAG invariants enforced, `walk_ops` on any well-formed IR value has a
simple, exception-free specification:

> Given value V, `walk_ops(V)` visits ops by following **operands**, **parameter values**, the
> **result type**, and **block argument types**. Captured values are treated as leaves (the walk
> stops at capture boundaries). No special cases. No `isinstance` guards. The traversal
> terminates because the use-def graph is a DAG.

**The locality property.** `walk_ops(B.result)` produces exactly:
- Ops whose transitive dependencies are entirely within `B.parameters ∪ B.args ∪ B.captures`, and
- Ambient ops (no block-argument dependencies) reachable via the use-def graph.

It will never surface ops whose scope belongs to a different block. This is what makes block-level
analysis, inlining, and scheduling sound in isolation: a block's complete dependency set is
readable directly from its parameters, arguments, and captures.

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

**Under the closed-block invariant (with captures):**

```
%src : llvm.Ptr = llvm.alloca<6>()
%5 : Nil = affine.for<0, 4>() (%i: Index) captures(%src):
    %val : F64 = affine.load(%src, [%i])
```

`walk_ops` on this block's result treats `%src` as a leaf (it's in the captures stop set) and
returns only the block-local ops. The block's complete dependency set is manifest in its argument
and capture lists. Any tool that operates on this block—a scheduler, a verifier, an inliner—reads
its inputs directly by inspection.

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

The `if` op is responsible for forwarding captured values to each branch as block arguments; here
`%n` and `%self` are forwarded to the `else` block. The updated `if` semantics are described in
Section 3.4. The type of `%self` is `Function<Index>`, resolved at definition time. This is
**nominal isorecursion**: the declared type is the fixpoint, and the self-reference is broken by
the declaration boundary rather than by an explicit roll/unroll.

Calling `%factorial` from the outside (Option A syntax — see Section 3.6):

```
%five : Index = 5
%result : Index = call(%factorial, [%five])
```

`%factorial` is an ambient op (no external block-argument dependencies) and appears as an operand.
See Section 3.6 for the three options for how `call` and self-reference interact.

#### Mutual Recursion via Nesting

Mutual recursion does not require a separate `func.letrec` primitive. One function may be defined
inside the other, capturing the outer function's `%self` as an explicit argument:

```
%even : Function<Index> = func.recursive() (%self: Function<Index>, %n: Index):
    // %odd is defined inside %even; it receives %even as an explicit argument
    %odd : Function<Index> = function() (%even': Function<Index>, %m: Index):
        %one : Index = 1
        %m1 : Index = subtract_index(%m, %one)
        %res : Index = call(%even', [%m1])
    %zero : Index = 0
    %cond : llvm.Int<1> = llvm.icmp<"eq">(%n, %zero)
    %result : Index = if(%cond) ():
        %t : Index = 1
    else (%self: Function<Index>, %n: Index):
        %one : Index = 1
        %n1 : Index = subtract_index(%n, %one)
        // %odd is ambient (no block-arg deps) — referenced without threading
        %res : Index = call(%odd, [%self, %n1])
```

`%odd` is a `function` op with no external operands. Its transitive dependencies include only its
own block arguments (`%even'`, `%m`), not anything from `%even`'s outer scope. Therefore `%odd`
is an **ambient node** and may be referenced from the `else` block without threading. `%self` and
`%n`, however, are block arguments of `%even`'s body and must be declared in the `else` block's
argument list; they are forwarded there by the `if` op.

`%odd` is not self-recursive, so it does not need `func.recursive`. The mutual recursion cycle is
broken by explicit argument passing: `%odd` is an ambient function that receives `%even` as the
argument `%even'` and calls back through it. `%odd` has no inherent placement—the ASM above shows
it defined inside `%even`'s block, but this is a scheduling choice; as an ambient node it could
legally appear anywhere. The use-def graph is a DAG throughout.

The trade-off is asymmetry: one function must pass its reference to the other as an explicit
argument. For larger mutual recursion groups (A → B → C → A), all ambient inner functions receive
the root's reference threaded explicitly. A `func.letrec` group op can be added later as syntactic
sugar for the symmetric case if warranted.

### 3.2 Label `%self`: Eliminating Use-Def Cycles in Loops

The loop back-edge problem is symmetric to the recursion problem, and the solution is symmetric.
Every `llvm.label` header block receives `%self : llvm.Label` as a **block parameter**—a
compile-time value bound to the label itself. The body captures `%self` and uses it as the
back-branch target:

```
// %main function containing a count-down loop
%main : Function<Nil> = function() body():
    %loop : llvm.Label = llvm.label() body<%self: llvm.Label>(%i: Index):
        %zero : Index = 0
        %cond : llvm.Int<1> = llvm.icmp<"sgt">(%i, %zero)
        %body : llvm.Label = llvm.label() body(%j: Index) captures(%self):
            %one : Index = 1
            %i_next : Index = subtract_index(%j, %one)
            %_ : Nil = llvm.br<%self>([%i_next])
        %exit : llvm.Label = llvm.label() body():
            %done : Index = 0
        %_ : Nil = llvm.cond_br<%body, %exit>(%cond, [%i], [])
    %ten : Index = 10
    %_ : Nil = llvm.br<%loop>([%ten])
```

Comparing this to the back-edge version from Section 1.2: the use-def cycle is eliminated.
`%self` is a block parameter (a leaf in the DAG), and the back-branch `llvm.br<%self>` targets
it as a compile-time parameter, not a use-def edge. `%i_next` is passed as a branch argument
(the only value that varies per iteration); `%self` is captured (constant across iterations).

Branch targets are parameters (`<>` syntax), not operands—they are compile-time label references.
The codegen reads the parameter value directly to determine which label a branch targets.

#### Nested Loops

For nested loops, inner blocks capture enclosing headers' `%self` values through the capture
chain. No explicit threading through block arguments is needed:

```
%outer : llvm.Label = llvm.label() body<%self: llvm.Label>(%i: Index):
    %inner : llvm.Label = llvm.label() body<%inner_self: llvm.Label>(%j: Index) captures(%self):
        // %self captured from outer header; %inner_self is this header's own parameter
        ...
        %_ : Nil = llvm.br<%inner_self>([%next_j])
    %exit : llvm.Label = llvm.label() body() captures(%self):
        %_ : Nil = llvm.br<%self>([%next_i])    // outer back-branch uses captured %self
```

#### Blocks and Functions: A Unified View

With `%self : llvm.Label` on every label block, the distinction between a `func.recursive` and a
looping block largely collapses: both are callable things that receive `%self` and can recurse
through it. The remaining distinction is that a `function` has a return type (its body's result
propagates out), while a label is a continuation—control transfers via `llvm.br` and there is no
return value. This is explored further in Section 4.

### 3.3 Explicit Capture

#### The problem with MLIR's implicit capture

In MLIR, an op inside a region may reference any SSA value that dominates it from an enclosing
region. This is convenient but has costs:

- **Inlining requires capture analysis.** MLIR's `InlinerInterface` must discover which values
  are captured from enclosing scopes, remap them, and handle type conversions — hundreds of lines
  of code. Any pass that moves ops across region boundaries must do similar analysis.
- **Dominance analysis is required.** MLIR needs a dominance tree to verify that uses are valid
  across region boundaries. This is a global analysis that must be recomputed after structural
  changes.
- **Dependencies are non-local.** A region's complete dependency set cannot be read from its
  signature — you must walk all ops inside it and check which values come from outside.
- **Fusion and outlining require free-variable analysis.** To fuse two operations or extract a
  subgraph into a separate function, you must first compute the set of free variables — an
  analysis that closed blocks make unnecessary.

#### Explicit capture: the design

The explicit capture discipline is: **a block's free variables must be declared in its interface**.
A block has three input lists:

- **`parameters`**: compile-time values bound by the owning op (e.g. `%self`).
- **`args`**: runtime values that vary per entry (loop induction variables). Passed by branches,
  generate phi nodes.
- **`captures`**: outer-scope values referenced directly. Listed on the block, not passed by
  branches, no phi nodes.

```
%loop : llvm.Label = llvm.label() body<%self: Label>(%iv: Index) captures(%alloc):
    %val = affine.load(%alloc, [%iv])    // %alloc is a capture — no phi, no threading
    ...
    llvm.br<%self>([%next_iv])           // only %iv is passed; %alloc and %self are not
```

This makes each block a **closed term**: its meaning is determined entirely by its declared
inputs. Benefits:

- **Inlining is mechanical.** `inline_block(block, args)` substitutes args; captures are
  already valid in the caller's scope. No capture analysis needed.
- **Fusion and outlining read the interface.** The complete dependency set is
  `parameters ∪ args ∪ captures`, readable by inspection.
- **No dominance analysis.** Validity is structural — if a value is in the interface, it's in
  scope. The verifier is a local check per block.
- **Independent analysis.** Any pass working on block B needs only its declared inputs. It never
  needs to reconstruct implicit dependencies from the ops inside.

#### Rejected alternative: threading through block arguments only

An earlier design threaded all outer-scope references through block arguments — every value
crossing a block boundary became an additional block arg, passed through branches and generating
phi nodes. This is the purest form of "defunctionalization": no closures, every input is an
explicit argument.

This approach was rejected because it creates severe complexity in practice:

- **O(n * d) block arguments.** For n outer values and d nesting levels, each block at each
  level carries copies of all outer values. A 3-deep nested loop with 5 outer values has 15
  extra block args across the three header/body/exit blocks, each generating phi nodes.
- **Lowering complexity.** The lowering pass must maintain parallel lists of block args at every
  scope level (`outer_header_args`, `outer_body_args`, `outer_exit_args`), an `exit_remap` dict
  to substitute values when exiting a loop, and an `_enclosing_loops` stack to track which outer
  `%self` values to thread into nested loops. The resulting code is fragile and hard to read.
- **Codegen complexity.** The code generator must trace `%self` block args back through
  predecessor branches to discover which label they refer to, requiring a fixed-point propagation
  loop — all to recover information that was known at IR construction time.
- **Every op reinvents threading.** Each op with blocks must define its own convention for how
  outer values reach inner blocks: `ForOp` has `init_args`, `CondBrOp` has `true_args`/
  `false_args`, `PipelineOp` threads through its body arg. There is no shared mechanism.

The captures design eliminates all of this. Outer-scope values are listed once on each block
that references them. `walk_ops` stops at capture boundaries, maintaining the locality property.
`replace_uses` maintains captures with a single list substitution per block. The lowering lists
captures at construction time (typically derived from the input IR's own block args), and no
further tracking or remapping is needed.

#### Capture chaining

Captures must **chain**: if block B contains a nested block B' that captures value V, then V
must also be in B's scope (in `B.parameters ∪ B.args ∪ B.captures ∪ B.ops`). The verifier
enforces this. The chaining invariant ensures that `replace_uses` on B can propagate the
substitution through B' — without it, a replacement of V in B would leave B' holding a stale
reference.

#### Verifier

`verify_closed_blocks` runs as a pre/post-condition of every pass. For each block, it checks:

1. Every operand and parameter value of every op in the block is in the block's valid set
   (`parameters ∪ args ∪ captures ∪ ops`).
2. Every capture of every child block is in the parent block's valid set (chaining).
3. The block's result is in the valid set.

Ambient ops (no block-argument dependencies) are always in scope and do not need capturing.

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

Change `call` so that the callee is a runtime operand: `call(callee, args)`. Static calls pass
the ambient function op directly as the callee operand—no wrapper needed.

```
op call(callee: Function<T>, args: List<T>) -> T
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
%five : Index = 5
%result : Index = call(%factorial, [%five])    // %factorial is an ambient op
```

**First-class function:**
```
%apply : Function<Index> = function() (%f: Function<Index>, %n: Index):
    %result : Index = call(%f, [%n])    // uniform — identical syntax to the recursive case
```

`walk_ops` on `call(%self, [%n1])` follows the operand edge to `%self`, which is a
`BlockArgument` — a leaf. `walk_ops` on `call(%factorial, [%five])` follows the operand edge to
`%factorial`, an ambient op. `walk_ops` visits `%factorial` itself but does not descend into its
body (nested blocks are not traversed). The `FunctionOp` guard disappears entirely.

**Trade-offs:** Requires changing the `call` op signature. All call sites (static and dynamic)
become uniform. The cost is one op-signature change; the gain is a uniform, guard-free `walk_ops`.

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

Introduce **block parameters**: a new syntactic position in a block declaration, distinct from
block arguments. Block parameters appear before the argument list in a named-block form. The first
block of an op is named explicitly (currently the first block name is elided, a "weird carveout"
in the formatter):

```
op_name() block_name<%param: Type>(arg1: T1, arg2: T2):
    ...
```

`%param` is a block parameter: it is in scope throughout the block and may be threaded to
sub-blocks as an ordinary block argument. The meaning of block parameters is defined by each op's
lowering pass — the IR has no implicit semantics for them. For `func.recursive` and `llvm.label`,
the convention is that the block parameter holds the op's own callable identity, and lowering
passes are responsible for populating it.

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

**Trade-offs:** No change to `call`. The IR remains a strict DAG: `%self` as a block parameter
is a leaf, never a back-edge. However, this option requires introducing a new concept into the IR
(block parameters), updating the formatter and parser to handle the named-block syntax, and
establishing a convention (enforced by lowering passes) for how each op populates its block
parameters. When `%self` is threaded to sub-blocks it becomes an ordinary block argument, which
is coherent but means the same value appears in two different binding positions depending on which
block is in scope.

---

**Recommendation.** Option C best preserves the current `call` semantics and aligns naturally
with how `%self` works for labels (auto-supplied by lowering, not threaded by callers). The
named-block syntax makes the self-reference explicit and structurally distinct from ordinary block
arguments. Option A is clean and uniform but requires changing the `call` op signature.
Option B is the cheapest to implement but violates the DAG property and does not eliminate the
structural special case in `walk_ops`.

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
    if value in visited:
        return
    visited.add(value)
    for _, operand in value.operands: visit(operand)
    for _, param in value.parameters: visit(param)
    visit(value.type)
    for _, block in value.blocks:
        for arg in block.args: visit(arg.type)
    order.append(value)
```

No `isinstance` guards of any kind. Every value has operands, parameters, a type, and blocks —
the same three loops apply uniformly. The `FunctionOp` guard disappears under Option C (block
parameters are leaves) and under Option A (`%factorial` is an ambient op visited but not
descended into). Under Option B the visited set prevents infinite loops but the DAG violation
remains, as discussed in Section 3.6.

The block argument type traversal remains correct and necessary when a block argument's type is
itself a value (a dependent type). Under explicit capture, such type dependencies are either
ambient or already explicit block arguments; no special cross-scope handling is required.

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
