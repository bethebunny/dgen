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
  `transitive_dependencies`.

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

**All ops in a block must be reachable from the block's `result` via `block.ops`.** An op
not reachable from the result is dead and will not execute. The block's `result` is the
use-def root; `block.ops` on `result` gives the canonical op list.

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

### What `transitive_dependencies` Follows

`transitive_dependencies(root, stop)` returns all values that are transitive dependencies of the root,
in the same block. The dependency edges are: operands, parameters, types, block
captures, and block argument types. These are exactly the edges that connect a value
to the values it depends on within a single block scope.

It does NOT descend into nested block bodies. Each block is its own walk scope.
Captures are boundaries — the walk visits them as dependencies but doesn't traverse
past them into their own dependency subgraphs (they're in the stop set).

---

## Control Flow Dialects

### `goto` dialect — Unstructured control flow

Labels, branches, conditional branches. Used for loops (back-edges) and for if/try
merges (forward-converging branches).

**Label-as-expression model**: A `goto.label` is not a jump target — it's an expression
block that runs when control reaches it in use-def order. No explicit entry branch is
needed. The label's `initial_arguments` provide first-iteration values for its block args.

#### Block parameter conventions

Every region body declares parameters `[%self, %exit]`:

- `%self` is a back-edge target — `branch<%self>` re-enters the body block.
  Loops use this to iterate. If/try-merge regions don't use it (the slot is
  vestigial there).
- `%exit` is the post-region label. `branch<%exit>(value)` leaves the region with a
  value. The region's value (the type declared on `goto.region`) is the phi
  populated at `%exit` from the args carried by every `branch<%exit>`. Regions
  whose type is `Nil` produce no LLVM-level value and the exit phi is skipped.

Labels do not declare their own `[%self, %exit]`. When a label needs to refer
to the enclosing region's `%self` (back-edge) or `%exit` (exit-with-value), it
lists them in its `captures` — the parameters belong to the region, the label
just borrows them.

#### Loop example

```
%header : Nil = goto.region([0]) body<%self: Label, %exit: Label>(%iv: Index):
    %cmp = algebra.less_than(%iv, %limit)
    %body = goto.label([]) body(%jv: Index) captures(%self):
        %next = algebra.add(%jv, 1)
        goto.branch<%self>([%next])      # back-edge: re-enter body with new IV
    goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])
```

Body's `%iv` is loop-carried — phi'd at body entry from `initial_arguments` (first
iteration) and the back-edge to `%self` (subsequent iterations). The region produces
no value (type `Nil`); the exit is reached when the conditional branches there.

#### If-merge example

```
%if : Index = goto.region([]) body<%self: Label, %exit: Label>():
    %then = goto.label([]) body() captures(%exit):
        %ten = 10
        goto.branch<%exit>([%ten])       # exit-with-value
    %else = goto.label([]) body() captures(%exit):
        %twenty = 20
        goto.branch<%exit>([%twenty])
    goto.conditional_branch<%then, %else>(%cond, [], [])
```

No body args (no loop carrier). Region's type is `Index`; the value is the phi at
`%exit` populated from each branch's args. `%self` is unused.

#### Codegen rule

Codegen has no "is this a loop?" heuristic. For each label (either the body block or
`%exit`), codegen emits a phi when the predecessors recorded against that label
provide phi-eligible values. Loops fill predecessors at the body block (initial
args + back-edges to `%self`); if-merge regions fill predecessors at `%exit`
(branches carrying values). The same emission rule serves both.

### `control_flow` dialect — Structured control flow

Higher-level ops that all lower to `goto` via `ControlFlowToGoto`.

- **`control_flow.for`** / **`control_flow.while`** — lower to a `goto.region` whose
  body has loop-carried args (phi at body entry) and back-edges to `%self`. Region
  type is `Nil`.
- **`control_flow.if`** — lowers to a `goto.region` with a conditional branch
  dispatching to then/else `goto.label`s; each label terminates with
  `branch<%exit>(value)`. Region type is the if's result type; the merge phi
  emerges at `%exit`.

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

## Generic Divergence Detection

`Value.totality` is a binary classification — **TOTAL** or **PARTIAL** — that
records whether a value carries any evidence of the `Diverge` effect. The check
is purely type-evidence-based: it inspects the value's existing operands,
parameters, and owned-block captures without any IR rewrites. It deliberately
does *not* fold the result type into the classification; whether divergence is
unconditional on a particular evaluation is observable directly via
`isinstance(v.type, Never)` at the (one) callsite that cares.

The two-state form mirrors Idris's `Total` / `Partial` and F\*'s `Tot` / `Div`.

### The `Diverge` effect

```dgen
type Diverge:
    layout Void
    has trait Effect
```

`Diverge` is the umbrella effect for "control transfer that may not return". Every
other diverging effect (`Raise<E>`, branch via `Label`, future actor-failure ops)
identifies itself as `Diverge` so a single `Handler<Diverge>` query suffices to
detect potentially-divergent ops without per-effect special cases.

### Evidence: `Handler<Diverge>`

Per the effect framework (`docs/effects.md`), a value that *can* trigger an effect `E`
carries a `Handler<E>` — typically a runtime operand or a value captured from an outer
scope. Today's `Handler<Diverge>`-bearing types:

| Type                     | Declares                                                |
| ------------------------ | ------------------------------------------------------- |
| `error.RaiseHandler<E>`  | `Handler<Raise<E>>` *and* `Handler<Diverge>` (see TODO) |
| `goto.Label`             | `Handler<Diverge>`                                      |

The dual declaration on `RaiseHandler` is a forward-compat shim: dgen does not yet
support type subtyping, so until `Raise<E> <: Diverge` is expressible the handler
declares both traits explicitly. Tracked under "Type system / effects" in `TODO.md`.

### Classification rule

For a value `v`:

```
partial(v)  ≡  ∃ operand   of v.            type(operand)   has trait Handler<Diverge>
            ∨  ∃ parameter of v.            type(parameter) has trait Handler<Diverge>
            ∨  ∃ block ∈ v's owned blocks.
               ∃ capture  of block.         type(capture)   has trait Handler<Diverge>

totality(v) =  PARTIAL  if partial(v)
            =  TOTAL    otherwise
```

### Why parameters count too

Compile-time parameters are weaker evidence than operands or captures, but they're
still the route by which `goto.branch<target: Label>` carries its handler. Including
parameters in the check keeps the predicate uniform: every dialect-level "this op
takes a `Handler<Diverge>`" surface — operand, parameter, or capture — counts the
same.

### API

```python
from dgen.type import Totality

class Totality(enum.Enum):
    TOTAL = "total"
    PARTIAL = "partial"

# On any Value:
v.totality  # -> Totality
```

The property is defined on `Value` (in `dgen/type.py`). The trait/type imports
inside the body are deferred to function scope because `Handler` and `Diverge`
live in `dgen.dialects.builtin`, which sits downstream of `dgen.type` — the
cycle is broken by lazy lookup at call time.

### Worked examples

```
%c : Index = 7                                  # constant       → TOTAL
%t : Index = error.try<Index>() body<%h: ...>:  # try            → TOTAL
    %r : Never = error.raise<Index>(%h, %c)     # raise          → PARTIAL  (type=Never)
except(%e: Index):
    %z : Index = 0
```

The `try` is TOTAL: the handler `%h` is bound as a body *parameter* of the body block,
not consumed by the try op itself. The `raise` is PARTIAL: `%h` is an operand whose
type is `RaiseHandler<Index>` (carrying `Handler<Diverge>`). It also has result type
`Never` — the standard combination "a divergence handler is in scope and divergence is
unconditional", but the totality property doesn't compress these two facts into a
single label.

```
%outer : Index = error.try<Index>() body<%ho: ...>:
    %inner : Index = error.try<Index>() body<%hi: ...>:    # inner try → PARTIAL
        %ok : Index = 5
    except(%err: Index) captures(%ho):                     # captures outer handler
        %re : Never = error.raise<Index>(%ho, %err)
except(%err: Index):
    %z : Index = 0
```

The inner try is PARTIAL: its except block captures `%ho`, whose type is
`RaiseHandler<Index>`. Its own result type is still `Index`, so a normal return is
possible — but the divergence handler is in scope.

```
goto.branch<%exit>([%val])      # branch          → PARTIAL
```

`branch` carries its target as a parameter, so it's PARTIAL. Its declared result type
is `Nil` today (a placeholder for "this is a terminator"); a future cleanup might
change it to `Never` to better reflect that no value flows past it, but that's
orthogonal to the totality classification.

### Implementation notes

- The rule examines only the value itself — no transitive walk. A `ChainOp` whose
  `rhs` is a `raise` is **not** itself PARTIAL; divergence-evidence is a property of
  the value that holds the handler, not of every op downstream of it. Consumers that
  want a "may-this-block-diverge" predicate should walk the block's ops with this
  property.
- `has_trait` does structural equality on type ASM, so the
  `Handler(effect_type=Diverge())` literal in the property body is a cheap lookup
  key, not an object-identity comparison.
- Types and constants always return `TOTAL` (no operands, parameters, blocks, or
  captures); the interesting cases are `Op` values.

---

## Function References

Function references (`function.FunctionOp`) are currently referenced by value across
block boundaries. Blocks should explicitly capture function references they depend on,
just like any other outer-scope value.
