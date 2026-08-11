# End State: Memory Effects and Origins

## Status

Proposal. Describes the intended *end state* of the memory effect system and
origins, reconciling the landed `State`/`Reference` design with the Origins
sketch in `docs/effects.md`. Supersedes the "Origins" section of
`docs/effects.md`; the effect framework and raise/try sections there remain
authoritative.

## Where we are (landed)

- `Effect`, `Handler<E>`, `Linear`, `Affine` traits (`dgen/dialects/builtin.dgen`).
- Raise/try with CPS lowering through goto (`dgen/dialects/error.dgen`,
  `dgen/passes/raise_catch_to_goto.py`).
- The `State` effect with `Reference<T>` as its linear handler: `load`/`store`
  consume the input `Reference` and produce a fresh one; `deallocate`
  discharges the thread (`dgen/dialects/memory.dgen`).
- The linearity verifier (`verify_linearity`, `dgen/ir/verification.py`),
  wired into every pass's pre/post hooks. Block-holding ops are treated
  conservatively (`MaybeAvailable`) — no per-op contract framework yet.
- `Buffer<T>`: unrestricted pointer with the legacy `mem`-token operand for
  ordering. Used by the ndbuffer/record/existential lowerings.
- `memory.deallocate` lowers to a no-op (leaks). No destructors. No aliasing
  model beyond "same token thread".

## The tension to resolve

The landed `Reference<T>` fuses two roles into one value:

1. **Data**: the pointer.
2. **Evidence**: the linear `Handler<State>` that orders accesses and carries
   the deallocation obligation.

Fusing them works for a single cell but cannot scale:

- A buffer has one allocation but many addressable elements — evidence
  per-element would make disjointness and deallocation incoherent.
- Two loads of the same cell serialize (each consumes the reference), even
  though reads commute.
- A pointer that must be aliased freely (`Some`/`Any`, shared immutable data)
  cannot be linear at all — which is why `Buffer` fell back to untyped mem
  tokens.

The end state splits the roles.

## The model

Five commitments:

1. **References are data.** `Ref<T>` is an unrestricted pointer value. It can
   be copied, stored, packed into records, passed anywhere. Producing an
   address is pure.
2. **Origins are evidence.** An `Origin` is a linear, zero-layout SSA value:
   simultaneously the `Handler<State>` for a region of memory, the carrier of
   its destruction obligation, and its provenance for alias analysis. Origins
   are erased at codegen.
3. **Addressing is pure; access requires evidence.** Computing
   `element_ref(ref, i)` or a field address needs no origin. `load`/`store`
   take the governing origin; stores consume it and produce a fresh one.
4. **Ordering is origin threading.** Two accesses are ordered iff they are
   connected through an origin's use-def thread. Accesses through unrelated
   origins commute — and that commutation is *sound*, because unrelated
   origins govern disjoint memory (see "Alias forest").
5. **Alias analysis is the IR.** Provenance is not a separate analysis
   lattice; it is literally the origin value's def chain. Two accesses may
   alias iff their origins are related by ancestry through split/join ops.

### Types

```
type State:
    data: Nil
    has trait Effect

# Linear evidence for a disjoint region of memory. Zero layout; erased.
type Origin:
    layout Void
    has trait Linear
    has trait Handler<State>

# Plain pointer data. Unrestricted.
type Ref<element_type: Type>:
    data: Pointer<Nil>
```

`Reference<T>` (linear, fused) and `Buffer<T>` (unrestricted, mem-token) both
dissolve into `Ref<T>` + `Origin`. Indexed storage is `Ref<Array<T, n>>` /
`Ref<Span<T>>` rather than a distinct buffer type, matching the existing TODO
to move `toy.Tensor` to `Pointer<Array<...>>`.

### Core ops

```
# Allocation returns the address and the evidence. The origin's base
# destructor is the matching deallocation (free / stack lifetime end).
op heap_allocate<T: Type>()  -> Tuple<Ref<T>, Origin>
op stack_allocate<T: Type>() -> Tuple<Ref<T>, Origin>

# Addressing: pure, no evidence involved.
op element_ref(ref: Ref<Array<T, n>>, index: Index) -> Ref<T>
op field_ref<index: Index>(ref: Ref<R>) -> Ref<F>

# Access: evidence in, evidence out.
op load(o: Origin, ref: Ref<T>) -> Tuple<T, Origin>
op store(o: Origin, ref: Ref<T>, value: T) -> Origin

# Discharge: runs the origin's destructor stack, then its base deallocation.
op destroy(o: Origin) -> Nil
```

Ops that produce `Linear` values are never CSE'd, duplicated, or deleted by
passes — each execution mints a distinct obligation. This is a pass-framework
invariant the verifier backstops (a deleted allocation shows up as a consumed
origin with no producer; a duplicated one as a double obligation).

### Destructors

The base destructor comes from the allocation op. Additional cleanup wraps
the origin:

```
op attach(o: Origin) -> Origin:
    block destruct    # signature: (%o: Origin) -> Origin
```

**Destruct-block contract.** The `destruct` block receives, as a block
parameter, the origin that `attach` consumed — evidence for the region for
the duration of teardown. Cleanup loads/stores thread through it, and the
block's result *is* the threaded origin: yield-as-consume, the same rule loop
carries use (`docs/linear_types.md`). The framing: `attach` does not end the
origin's thread — it defers its continuation into the block. `destroy`
resumes it: the most recently attached block runs first (LIFO), each block's
yielded origin feeds the next block inward, and the base deallocation is the
final consumer.

Three properties fall out with no new machinery:

- Evidence stays total inside cleanup code — no raw-access escape hatch.
- Resurrection is impossible: the only way to satisfy linearity is to yield
  the origin onward.
- Verification is the ordinary local Γ walk: origin `Available` at block
  entry, the yield is its consumption.

**Destruct blocks must be total**: their signature may not include a
`Handler<Diverge>` capability (checkable from the signature alone, per the
totality rules in `docs/linear_types.md`). A raising destructor would leak
the remainder of the destructor stack; totality is also what makes running
destructors on divergent paths sound (see "Interaction with raise" below).

Destructor code is ordinary explicit IR — no hidden runtime, no registration
machinery; codegen inlines the blocks at the destroy site.

This fixes the current leak: `deallocate`-as-no-op disappears; heap origins'
base destructor is a real `free`, and stack origins' base destructor lowers to
an LLVM lifetime-end marker (a concrete payoff of tracking lifetimes: alloca
reuse and better register allocation for free).

### Sub-origins: split and join

Disjoint parallel mutation needs disjoint evidence. Structural decomposition
ops consume a parent origin and produce child origins that govern provably
disjoint sub-regions, plus a linear `Join` ticket that reconstitutes the
parent:

```
type Join:
    layout Void
    has trait Linear

# Examples of the schema — the concrete set is enumerated per structure.
op split_at(o: Origin, index: Index) -> Tuple<Origin, Origin, Join>  # [0,i) / [i,n)
op split_field<index: Index>(o: Origin) -> Tuple<Origin, Origin, Join>  # field / rest
op join(j: Join, a: Origin, b: Origin) -> Origin
```

Disjointness is by construction per op (distinct fields; ranges split at an
index), never by arbitrary pointer arithmetic. Linearity already enforces the
whole forest protocol with zero new machinery: the parent is consumed by the
split, so it cannot be accessed or destroyed while children are live; children
must each be consumed exactly once, and `join` is the only op that gets the
parent back. "Sub-origins must be destroyed before their parents" is not a
special rule — it is ordinary single-consume.

### Alias forest

- Fresh roots come from allocation ops.
- Children come from split ops; `join` returns to the parent.
- Two accesses may alias iff their origins are connected through the origin
  def chain without diverging at a split (ancestry).
- Origins whose def chains diverge at a split, or that come from different
  allocations, are disjoint — accesses through them commute and may be
  reordered or parallelized.
- `world` (below) is top: may alias anything.

A pass asking "may these two stores alias?" walks two origin def chains. No
side tables, no invalidation problem — replacement cascades keep the def
chains correct the same way they keep every other operand correct.

### The world origin

Opaque external effects (extern calls, `print_memref`, I/O) thread a
distinguished top origin:

```
op world() -> Origin   # one per function entry; may-alias-everything
```

Extern calls that touch memory take and produce the world origin. This
replaces ad-hoc chaining for externals, gives them a principled ordering
story, and marks exactly where alias analysis must give up. Passing `world`
into a function is the v1 story for "this function may perform I/O".

### Reads: observe vs consume

The end state distinguishes two operand modes in per-op linearity contracts:

- **Consume**: the op ends the value's thread (store, destroy, split).
- **Observe**: the op requires the value live but does not end it (load).

Scheduling rule: every observer of a linear value executes before its
consumer — the consume is a fence for its observers. This makes read/read
commute (two loads observe the same origin, no order between them) while
read/write and write/write stay ordered through the thread.

This is staged *after* the per-op contract framework lands (existing TODO in
`TODO.md`): observe is precisely a contract annotation, and the verifier's Γ
transitions extend naturally (observing uses don't transition state; codegen
gains observer→consumer anti-dependence edges). Until then, `load` consumes
and reproduces, exactly as today — correct, merely over-sequential.

### Interaction with raise and partial ops

An op is *partial* iff `Value.totality` is `PARTIAL`: it has a direct
dependency — operand, parameter, or owned-block capture — on a value whose
type is `Handler<Diverge>` (`dgen/type.py`). Partiality propagates outward
through captures: an `if` whose branch captures a raise handler is itself
partial from the enclosing block's view.

A raise that unwinds past an open origin must not leak its obligation. An
earlier draft of this section resolved that with *declared semantics* — "a
partial op discharges, on its divergent edges, every origin thread open
across it" — materialized by `raise_catch_to_goto` inserting destroys. That
design has a disqualifying flaw: it gives the partial op an **implicit
dependency** on every open origin. Which origins are open across an op is a
fact about the op's *uses* (the downstream store is what makes an origin
open across the `if`), so the op's divergent behavior would be defined by
its context — invisible to `dependencies`, `transitive_dependencies`
(staging extraction would miss the origin on the divergent path),
`replace_uses_of`, and dead-code elimination (deleting the store would
silently change what the `if` does on divergence). dgen's core contract is
that dataflow is explicit; discharge must be too.

The end state instead makes divergence consume explicitly, via two rules:

1. **Threading rule (per-block verification).** For every partial op `P`
   and linear value `v` in a block, `v`'s thread must relate to `P` in
   exactly one of three ways: it *completes before* `P` (its consumer is a
   transitive dependency of `P`), *starts after* `P` (its creation depends
   on `P`'s result), or *threads through* `P` (passed in via `P`'s
   arguments/captures and yielded back in `P`'s result on every
   non-divergent path — the loop-carry pattern applied to alternatives).
   A thread that *bypasses* `P` is rejected: on a divergent execution the
   bypassed obligation leaks, and whether its consumer had run at
   divergence time is schedule-dependent. The check needs only
   `transitive_dependencies`.

2. **Divergent-block drain rule (local, signature-level).** A block whose
   result type is `Never` must consume every linear input (argument,
   parameter, capture). Checkable from the block alone. The discharge is
   ordinary IR the frontend writes — `destroy` the origins and chain them
   into the raise's operands, ordering cleanup before the transfer. `raise`
   needs no signature change, and `raise_catch_to_goto` inserts nothing: it
   lowers exactly what is written.

Destruction *order* on the unwind path is whatever the frontend writes; the
one hard requirement — children before parents — is already enforced by
linearity (a parent cannot be destroyed while a child's thread is open).

The division of labor this preserves: *semantics* stay local (every consume
is an operand edge), while *verification* may be non-local (the threading
rule inspects the whole block, exactly as `verify_linearity` already does).
Non-local checks are fine; non-local meaning is not.

**Frontend ergonomics.** The liveness a frontend needs is trivial by
construction: linearity forbids conditional consumption (an alternative
that consumes a capture must diverge or yield a replacement), so "is this
origin still live here" is path-unique and syntactic — no drop flags exist
in this IR. A frontend with lexically scoped resources reads the live set
out of its own symbol table. Frontends that want automatic RAII insertion
can use an optional, dialect-independent **explication pass** that computes
open threads and rewrites them into threaded form with divergent-path
destroys — normalization sugar that produces the canonical explicit IR
*before* verification gates. Semantics are defined only on the explicit
form; the pass is convenience, never meaning.

Deliberate leaks (process exit, arena teardown, C frontends) should be an
explicit `forget(o)`-style op that forfeits the obligation visibly, not a
verifier exemption — see open questions.

#### Worked example

`safe_div(a, b)`: allocate a cell, compute `div_checked(a, b)` — the
canonical composite from `docs/effects.md`, not a primitive but an expansion
containing an explicit conditional and internal raise — store the quotient,
read it back, destroy, with a fallback on the exceptional path.
Pre-lowering, fully explicit: the origin *threads through* the partial `if`
— in via both argument spans, out via the result tuple on the surviving
path, destroyed before the raise on the divergent path. (A real frontend
would define a `DivByZero` error type; `index.Index` stands in here.)

```
%f : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
    %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%a, %b):
        %alloc : Tuple<[memory.Ref<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
        %r : index.Index = unpack(%alloc) body(%ref: memory.Ref<index.Index>, %o: memory.Origin) captures(%a, %b, %h):
            # -- div_checked(%h, %a, %b), expanded at this site --
            %zero : index.Index = 0
            %iszero : number.Boolean = algebra.equal(%b, %zero)
            %qo : Tuple<[index.Index, memory.Origin]> = control_flow.if(%iszero, [%o], [%o]) then_body(%o_t: memory.Origin) captures(%h):
                %code : index.Index = 1
                %d0 : Nil = memory.destroy(%o_t)          # explicit discharge
                %err0 : index.Index = chain(%code, %d0)   # ordered before the raise
                %raised : Never = error.raise<index.Index>(%h, %err0)
            else_body(%o_e: memory.Origin) captures(%a, %b):
                %quot : index.Index = algebra.divide(%a, %b)
                %pair : Tuple<[index.Index, memory.Origin]> = pack([%quot, %o_e])
            # -- end div_checked --
            %res : index.Index = unpack(%qo) body(%q: index.Index, %o1: memory.Origin) captures(%ref):
                %o2 : memory.Origin = memory.store(%o1, %ref, %q)
                %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o2, %ref)
                %out : index.Index = unpack(%loaded) body(%v: index.Index, %o3: memory.Origin):
                    %d : Nil = memory.destroy(%o3)
                    %done : index.Index = chain(%v, %d)
    except(%err: index.Index):
        %z : index.Index = 0
        %fallback : index.Index = algebra.add(%err, %z)
```

The expansion is where the caller's linear context gets woven in. The
composite's *definition* (`handler, a, b → quotient`) says nothing about
origins; the *expansion site* threads whatever is live there — a different
call site with two open origins would thread both. This is why
composites-as-expansions compose with the threading rule while
composites-as-functions are harder: an outlined `div_checked` called via
`function.call` would make the call op partial (handler operand), and a
caller holding an origin across it would need the origin *in the callee's
signature* (`(h, a, b, o: Origin) -> Tuple<Index, Origin>`) — linear
context surfacing in signatures. v1 already forbids handlers crossing
function boundaries, so composites are expansions for now; the outlined
form is exactly the deferred function-boundary effect design.

#### The un-expanded form

Expansion need not happen at frontend elaboration time. A dialect can keep
`div_checked` un-expanded in the IR — a leaf op expanded later by a lowering
pass — provided the threading is part of its signature. The pattern: a
`Span` operand carrying the linear values to thread, returned positionally
in the result tuple on the surviving path:

```
# checked.dgen (illustrative)
op div_checked(handler: RaiseHandler, dividend, divisor, thread: Span) -> Tuple
```

The call site, replacing the expansion above:

```
            %qo : Tuple<[index.Index, memory.Origin]> = checked.div_checked(%h, %a, %b, [%o])
            %res : index.Index = unpack(%qo) body(%q: index.Index, %o1: memory.Origin) captures(%ref):
                %o2 : memory.Origin = memory.store(%o1, %ref, %q)
                ...as before...
```

- **The discharge dependency stays explicit dataflow**: `%o` is an operand.
  The op's linearity contract (migration step 1) declares that elements of
  `thread` are consumed and re-produced positionally in the result on the
  surviving path, destroyed on divergence. The flaw that killed implicit
  discharge cannot return: the op discharges exactly what it names, never
  what its context happens to leave open.
- **The result type depends on the `thread` operand's types**
  (`Tuple<[Index, Origin]>` here). Result types are SSA values in dgen, so
  operand-dependent result types are the ordinary dependent-type machinery,
  resolved by staging.
- **`control_flow.if` already has this shape** — its argument spans are its
  threading surface. Leaf composites and block-holding ops thread the same
  way; a call site with nothing open passes `[]`.
- **The expansion pass must produce IR satisfying the contract** — the
  expanded example above is exactly that output, and the post-pass verifier
  checks it with the ordinary rules. Contract on the un-expanded op,
  explicit ops after expansion: the same relationship `try` has to
  `raise_catch_to_goto`.
- An outlined function version would have the same signature shape; the
  thread span is what the deferred function-boundary design generalizes
  (manual threading first, polymorphism later).

Verifier's view, per block (locality as in `docs/linear_types.md`):

- In the outer unpack body, the partial op is the `if` (its `then_body`
  captures `%h`, a `Handler<Diverge>`). `%o` *threads through* it: consumed
  as an argument (appearing in both spans, deduplicated per branch
  composition), reborn from the result tuple on the surviving path.
  `%o1`/`%o2`/`%o3` all *start after* the `if` — the other legal relation.
  No thread bypasses the partial op.
- `then_body` has result type `Never`, so the divergent-block drain rule
  applies: its linear input `%o_t` must be consumed — it is, by the destroy,
  chained into the raise so cleanup is ordered before the transfer. `Never`
  is result-compatible with the `if`'s tuple type, as usual.
- The destroyed origin is the thread state *as of the divergence*: the store
  has not run on this path, so the destructor stack is whatever was attached
  before the `if`; the base destructor frees the cell regardless of its
  contents.
- In the try body, the partial op is the *unpack* (partiality propagates
  outward through its `%h` capture). `%alloc` — a tuple containing a linear
  component, hence linear — is consumed *by* the partial op: the third legal
  relation.

`raise_catch_to_goto` then has nothing to insert. It rewrites the raise to a
`goto.branch` targeting the except label via the existing handler→label
capture cascade; the destroy is already in the IR, ordered before the branch
by the same chain:

```
            %qo : Tuple<[index.Index, memory.Origin]> = control_flow.if(%iszero, [%o], [%o]) then_body(%o_t: memory.Origin) captures(%except):
                %code : index.Index = 1
                %d0 : Nil = memory.destroy(%o_t)
                %err0 : index.Index = chain(%code, %d0)
                %1 : Nil = goto.branch<%except>([%err0])
            else_body(%o_e: memory.Origin) captures(%a, %b):
                ...unchanged...
```

The same rules verify the lowered form with no special contract. One
refinement the precise contract framework needs either way: **a capture or
argument consumed only inside an alternative whose every exit diverges does
not charge the parent's Γ** — the divergence-aware version of the
branch-composition rule, sound because that alternative never returns
control to the parent thread.

### Functions

Origins are ordinary values, so function boundaries need no new features:

- **Ownership transfer out**: return an `Origin` (alone or paired with its
  `Ref`). Returning consumes it locally; the caller receives the obligation.
- **Ownership transfer in**: take an `Origin` parameter. The callee must
  consume it (destroy, return, or thread into a returned structure).
- **Borrowing**: take an origin and return it — `(o: Origin, ...) ->
  Tuple<..., Origin>`. The caller's thread continues from the returned value.

Since origins have `layout Void`, none of this has ABI cost — signatures
carry the evidence at compile time and erase at codegen. Effect polymorphism
and richer borrow inference remain out of scope, as in `docs/effects.md`.

### Loops

A loop body that stores must thread its origin as a loop carry: the body
consumes the carry and yields a fresh origin of the same type
(yield-as-consume, `docs/linear_types.md`). This makes the verifier's
carry-pair modeling (existing TODO) a *prerequisite* for migrating the
ndbuffer lowering — its generated loops are exactly stores-in-a-body.

### Shared immutable data

`Some`/`Any` and other freely-aliased pointer-shaped values get:

```
op freeze(o: Origin) -> FrozenOrigin   # FrozenOrigin: unrestricted, read-only evidence
```

Freezing consumes the linear origin and forfeits the destruction obligation
(today's `Some`/`Any` already leak; this makes the leak a visible, typed
decision rather than an accident of `Buffer`). Loads accept either origin
kind; stores require `Origin`. Reclaiming frozen data (arenas, refcounts) is
future work layered on `attach`.

### Staging

Origins are compile-time values in exactly the sense types are: SSA values
that participate in dataflow, constrain scheduling, and erase. They are stage
artifacts, not runtime state — `layout Void`, no `Memory` representation, no
constant form. An op is never runtime-dependent *because of* its origin
operands.

### Lowering

Origins lower by erasure in `memory_to_llvm`:

- `heap_allocate` → malloc call; the origin result vanishes, its consumers
  rewire to pure ordering (the use-def thread is preserved through lowering,
  so codegen's use-def scheduling keeps access order without tokens).
- `store`/`load` → LLVM store/load; the threaded origin becomes the chain
  that already orders them today.
- `destroy` → inlined destructor blocks, then free / lifetime-end.
- `split`/`join`/`freeze`/`world` → nothing.

## Migration plan

Ordered so each step keeps the tree green:

1. **Per-op linearity contracts** (existing TODO): the framework for declaring
   consume/observe/borrow semantics per op. Prerequisite for everything below;
   also immediately improves precision for block-holding ops.
2. **Loop-carry linearity** (existing TODO): carry-pair modeling in
   `verify_linearity`. Prerequisite for origin-threaded loops.
3. **Introduce `Origin` + split ops** in `memory.dgen` alongside the current
   API: `Ref<T>`, two-result allocation, `load(o, ref)`/`store(o, ref, v)`,
   `destroy`, `attach`. Mark `Origin` as `Linear` — the verifier picks it up
   with no plumbing (existing TODO).
4. **Migrate single-cell users** off fused `Reference<T>`; delete
   `Reference`'s `Linear`/`Handler` traits by folding it into `Ref<T>`.
5. **Migrate buffer users** (ndbuffer, record, existential lowerings,
   `passes/support/memory.py`) off mem tokens; delete `Buffer<T>` and the
   `mem` operands. Real `free` in destroy; stack lifetime markers.
6. **split/join + alias-aware reordering**: land the decomposition ops, then
   let passes use origin-forest disjointness (first client: loop
   parallelization / vectorization legality in the structured lowering).
7. **Observe contracts for loads** (read/read commutation), the
   threading + divergent-drain verifier rules for partial ops, and the
   optional discharge-explication normalization pass.
8. **`world` origin** for externs; retire ad-hoc chaining of effectful calls.

## Open questions

Each question is annotated with its *forcing point* — the migration step (or
external event) by which it must be resolved. Questions with no forcing point
are clean extensions: resolving them later strengthens the verifier or adds
ops, without revisiting decisions made here. (The two questions that had hard
forcing points — the destruct-block contract and the partial-op drain rule —
are resolved in the "Destructors" and "Interaction with raise" sections
above.)

- **Handlers inside sum types**: `Value.totality` classifies partiality from
  the *direct* types of dependencies. There is no art yet for a union or
  existential value that *may* contain a `Handler<Diverge>` (or a linear
  component). Intended conservative rules when such types land: a sum that
  may contain a diverging handler counts as one for totality; a sum with a
  linear alternative is itself linear. *Forcing point:* when sums/
  existentials start carrying handlers or linear values.
- **Deliberate leaks**: a `forget(o)` op that consumes an origin and
  visibly forfeits its obligation (process exit, arena teardown, C
  frontends whose semantics permit leaks). Purely additive. *Forcing
  point:* first frontend that needs it.
- **Runtime-index split soundness**: `split_at(o, i)` is disjoint by
  construction, but proving *which* side a given `element_ref(ref, j)` falls
  in requires relating `j` to `i`. v1: splits are introduced only by compiler
  passes that construct both the split and the accesses (trust the producer).
  The future refinement needs child origins to carry their range —
  parameterizing `Origin` later is verifier-facing only (origins have no
  runtime representation), so it is additive. *Forcing point:* none for v1;
  revisit when a frontend wants to write splits by hand.
- **Escape analysis**: a `Ref` outliving its origin is dangling. Linearity
  prevents the origin disappearing while *threaded* uses remain, but a stored
  `Ref` reloaded after `destroy` is not caught. Candidate: origins
  parameterize `Ref` types (`Ref<T, o>`) so staleness is a type error. This
  is a monotone strengthening — it rejects more programs and changes no
  semantics — and addressing ops just propagate the parameter, so
  addressing-is-pure survives. Churn is broad but mechanical. *Forcing
  point:* none; deferred until dependent types mature.
- **Concurrency, atomics, volatile/MMIO**: linear origins rule out
  concurrent access by construction — that is the feature, but it means
  shared mutable state has no story here. Candidates: a sibling effect to
  `State` with unrestricted handlers (atomic cells as their own type, not
  origin-governed), `world`-threading as a stopgap, fractional permissions
  for cross-thread read sharing. Cross-thread ordering is not use-def
  expressible, so this needs its own design regardless; the effect framework
  supports sibling effects, and `world`/top covers mixed access
  conservatively in the meantime. *Forcing point:* none until dgen targets
  concurrent code (actor dialect sharing memory).
- **Erasure-boundary enforcement**: after `memory_to_llvm`, origins are
  erased and ordering survives only as opaque chains — so alias-dependent
  transformations are sound only *above* the boundary, and raw accesses may
  only be produced by lowering, never written above it. The invariant is
  settled; how to enforce it mechanically (a verifier rejecting raw llvm
  accesses in pre-erasure IR, dialect-legality checks, or pass-ordering
  constraints in `Compiler`) is open. *Forcing point:* none — pass ordering
  already provides the soundness; enforcement is hygiene.
- **Frozen reclamation**: arenas or refcounting via `attach`-style wrapping.
  Purely additive (new ops beside `freeze`). One deliberate irreversibility:
  data frozen via v1 `freeze` is unreclaimable — matching today's
  `Some`/`Any` behavior. *Forcing point:* none.
