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

1. **References are data.** `Reference<T>` is an unrestricted pointer value. It can
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
type Reference<element_type: Type>:
    data: Pointer<Nil>
```

`Reference<T>` (linear, fused) and `Buffer<T>` (unrestricted, mem-token) both
dissolve into `Reference<T>` + `Origin`. Indexed storage is `Reference<Array<T, n>>` /
`Reference<Span<T>>` rather than a distinct buffer type, matching the existing TODO
to move `toy.Tensor` to `Pointer<Array<...>>`.

### Core ops

```
# Allocation returns the address and the evidence. The origin's base
# destructor is the matching deallocation (free / stack lifetime end).
op heap_allocate<T: Type>()  -> Tuple<Reference<T>, Origin>
op stack_allocate<T: Type>() -> Tuple<Reference<T>, Origin>

# Addressing: pure, no evidence involved.
op element_ref(ref: Reference<Array<T, n>>, index: Index) -> Reference<T>
op field_ref<index: Index>(ref: Reference<R>) -> Reference<F>

# Access: evidence in, evidence out.
op load(o: Origin, ref: Reference<T>) -> Tuple<T, Origin>
op store(o: Origin, ref: Reference<T>, value: T) -> Origin

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

The end state instead makes divergence consume explicitly. The organizing
idea: **the try is the cleanup scope**. A frontend holding linear values
across may-raise code wraps that code in its own `try` whose `except`
destroys the values it owns and re-raises outward. Each scope cleans up its
own values; re-raising composes scopes. This is the landing-pad structure
every production exception implementation uses, expressed as ordinary IR —
and after CPS lowering it is *literally* that structure: a cascade of goto
labels, each destroying its scope's origins and branching to the next
handler out. Zero cost on the normal path, no unwinder.

The verifier rules:

1. **Block drain (already landed behavior).** Every block must consume its
   linear inputs (arguments, parameters, captures) by its root — this is
   ordinary linearity (`LinearLeakError` today). For a divergent block
   (`Never` result), consumption is chained before the divergence. An
   `except` that cleans up and re-raises is verified by exactly this rule,
   with nothing added.

2. **Coverage rule (per-scope verification).** For each partial op `P`
   diverging via handler `h`, and each linear value `v` *open across* `P`
   (created independently of `P`, consumed by an op depending on `P`): `v`
   must be captured and consumed by the `except` block of the `try` that
   introduced `h`. Two corollaries:
   - **Uniformity**: a linear value consumed by an `except` must be open
     across *every* `h`-diverging op in the body — a value live at only
     some raise sites means the try is mis-scoped; the fix is nesting
     another try at the scope boundary where the value's lifetime begins.
   - **Ordering**: a consumer unordered with respect to `P` (neither a
     dependency of `P` nor dependent on it) is rejected — whether the value
     was still open at divergence would be schedule-dependent, and cleanup
     would risk double-destroy.

   Values may still *complete before* `P`, *start after* `P`, or thread
   through pure control flow (`if` argument spans, loop carries) as
   ordinary dataflow; the coverage rule governs only threads that stay open
   across a divergence.

3. **Recursion for free.** An `except` that re-raises captures the outer
   handler, which makes the whole inner `try` partial in the enclosing
   block (`Value.totality` propagates through owned-block captures) — so
   the enclosing scope's coverage obligation arises automatically, with no
   extra machinery. Scopes compose because partiality composes.

Destruction *order* within a scope's cleanup is whatever the frontend
writes; the one hard requirement — children before parents — is already
enforced by linearity (a parent cannot be destroyed while a child's thread
is open).

The division of labor this preserves: *semantics* stay local (every consume
is an operand edge in an `except` the frontend wrote), while *verification*
may be non-local (the coverage rule inspects the scope, exactly as
`verify_linearity` already does). Non-local checks are fine; non-local
meaning is not.

**Frontend ergonomics.** The try-per-scope structure mirrors what a
frontend has anyway: each source scope owning resources compiles to a try
whose except destroys them and re-raises. Liveness is trivial by
construction — linearity forbids conditional consumption, so "is this
origin still live here" is path-unique and syntactic; no drop flags exist
in this IR. A cleanup-only scope (except that always re-raises) is the
IR encoding of `finally`-on-unwind / RAII drop scopes; sugar (a `defer` or
`scope` op lowering to it) can come later without touching semantics.

Deliberate leaks (process exit, arena teardown, C frontends) should be an
explicit `forget(o)`-style op that forfeits the obligation visibly, not a
verifier exemption — see open questions.

#### Worked example

`safe_div(a, b)`: allocate a cell, compute `div_checked(a, b)` — the
canonical composite from `docs/effects.md`, kept **un-expanded** with its
natural signature — store the quotient, read it back, destroy, with a
fallback on the exceptional path:

```
# checked.dgen (illustrative) — no origins in the signature, ever
op div_checked(handler: RaiseHandler, dividend, divisor)
```

The caller holds an origin across the may-raise call, so it wraps the call
in its own **cleanup scope**: a `try` whose `except` destroys the origin
and re-raises outward. (A real frontend would define a `DivByZero` error
type; `index.Index` stands in here.)

```
%f : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
    %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%a, %b):
        %alloc : Tuple<[memory.Reference<index.Index>, memory.Origin]> = memory.heap_allocate<index.Index>()
        %r : index.Index = unpack(%alloc) body(%ref: memory.Reference<index.Index>, %o: memory.Origin) captures(%a, %b, %h):
            # cleanup scope for %o
            %qo : Tuple<[index.Index, memory.Origin]> = error.try<index.Index>() body<%h2: error.RaiseHandler<index.Index>>() captures(%a, %b, %o):
                %q : index.Index = checked.div_checked(%h2, %a, %b)
                %pair : Tuple<[index.Index, memory.Origin]> = pack([%q, %o])
            except(%err: index.Index) captures(%o, %h):
                %d0 : Nil = memory.destroy(%o)            # this scope's cleanup
                %err2 : index.Index = chain(%err, %d0)    # ordered before the re-raise
                %raised : Never = error.raise<index.Index>(%h, %err2)
            %res : index.Index = unpack(%qo) body(%q2: index.Index, %o1: memory.Origin) captures(%ref):
                %o2 : memory.Origin = memory.store(%o1, %ref, %q2)
                %loaded : Tuple<[index.Index, memory.Origin]> = memory.load(%o2, %ref)
                %out : index.Index = unpack(%loaded) body(%v: index.Index, %o3: memory.Origin):
                    %d : Nil = memory.destroy(%o3)
                    %done : index.Index = chain(%v, %d)
    except(%err: index.Index):
        %z : index.Index = 0
        %fallback : index.Index = algebra.add(%err, %z)
```

**Why this composes.** The composite's signature carries only its own
effect (the handler), never its callers' resources. A site with two open
origins destroys both in its one except; the callee is untouched. The
outlined-function form works the same way: a future `div_checked`
*function* is `(h, a, b) -> Index` — the caller's cleanup scope wraps the
call, exactly as real exception systems keep callee signatures free of
caller frame information. The deferred function-boundary design only needs
to answer how *handlers* cross boundaries; linear cleanup never crosses at
all.

Verifier's view, per block (locality as in `docs/linear_types.md`):

- **Inner try body**: the partial op is `div_checked` (operand `%h2`).
  `%o` is open across it — created outside, consumed by the pack, which
  depends on `%q`. Coverage: `%h2`'s except captures and consumes `%o` ✓;
  it is the only `%h2`-diverging op, so uniformity is trivial.
- **Except block**: linear capture `%o` consumed by the destroy, chained
  into the re-raise; affine capture `%h` consumed by the raise; result
  `Never`. This is plain block drain — the rule the landed verifier
  already enforces (`LinearLeakError` on an unconsumed linear capture).
- **Unpack body (parent of the inner try)**: `%o` is captured by the try's
  body *and* its except — consumption alternatives, deduplicated exactly as
  branch composition prescribes; one of them runs to completion per
  execution. The parent sees the try consume `%o` and rebinds `%o1` from
  the result tuple. Nothing is open across the inner try from outside: by
  the time control reaches the *outer* except, the origin is already
  destroyed by the inner scope, so the outer except needs no cleanup
  captures. **Each scope cleans its own — that is the composition
  invariant.**
- **Recursion via totality**: the inner try captures `%h` (through its
  except), making the try itself partial in the unpack body — the
  enclosing scope's coverage obligation arises automatically and is
  satisfied vacuously.
- The destroyed origin is the thread state *as of the divergence*: the
  store has not run on the exceptional path, so the destructor stack is
  whatever was attached before the call; the base destructor frees the
  cell regardless of its contents.

If `div_checked` is expanded — by the frontend or a later lowering pass —
the expansion drops into the inner try body unchanged and context-free.
Its internal conditional raises against `%h2`; `%o`, open across that
`if`, is covered by the same except. The expansion never names the
caller's origins:

```
                # -- div_checked(%h2, %a, %b), expanded --
                %zero : index.Index = 0
                %iszero : number.Boolean = algebra.equal(%b, %zero)
                %q : index.Index = control_flow.if(%iszero, [], []) then_body() captures(%h2):
                    %code : index.Index = 1
                    %raised2 : Never = error.raise<index.Index>(%h2, %code)
                else_body() captures(%a, %b):
                    %quot : index.Index = algebra.divide(%a, %b)
                %pair : Tuple<[index.Index, memory.Origin]> = pack([%q, %o])
```

Compare with threading `%o` through the `if`'s argument spans: no origin
arguments, no destroy at the raise site, no tuple result from the `if`.
Both styles remain legal — an at-site destroy-before-raise satisfies
coverage vacuously, since that thread completes before the divergence —
but the scope style is the composable default.

After `raise_catch_to_goto`, the scopes become the landing-pad cascade:
each except is a `goto.label` that destroys its scope's origins and
branches to the next handler out. The re-raise is a branch; the cascade is
straight-line cleanup code on the exceptional path and free on the normal
path:

```
            %qo : Tuple<[index.Index, memory.Origin]> = goto.region([]) body<%self: goto.Label, %exit2: goto.Label>() captures(%a, %b, %o, %except):
                %except2 : goto.Label = goto.label([]) body(%err: index.Index) captures(%o, %except):
                    %d0 : Nil = memory.destroy(%o)
                    %err2 : index.Index = chain(%err, %d0)
                    %1 : Nil = goto.branch<%except>([%err2])   # re-raise = branch to outer handler
                %q : index.Index = checked.div_checked(%except2, %a, %b)
                %pair : Tuple<[index.Index, memory.Origin]> = pack([%q, %o])
```

(The handler→label substitution flows into the un-expanded op's operand,
as it flows into capture lists today; the op's own expansion then emits
the branch.) The same rules verify the lowered form with no special
contract. One refinement the precise contract framework needs either way:
**a capture consumed only inside an alternative whose every exit diverges
does not charge the parent's Γ** — the divergence-aware version of the
branch-composition rule, sound because that alternative never returns
control to the parent thread. Try body/except are exactly such
alternatives.

### Functions

Origins are ordinary values, so function boundaries need no new features:

- **Ownership transfer out**: return an `Origin` (alone or paired with its
  `Reference`). Returning consumes it locally; the caller receives the obligation.
- **Ownership transfer in**: take an `Origin` parameter. The callee must
  consume it (destroy, return, or thread into a returned structure).
- **Borrowing**: take an origin and return it — `(o: Origin, ...) ->
  Tuple<..., Origin>`. The caller's thread continues from the returned value.

Since origins have `layout Void`, none of this has ABI cost — signatures
carry the evidence at compile time and erase at codegen.

**Effectful functions.** The cleanup-scope design leaves surprisingly
little for functions to add. A may-raise function's signature carries only
its own effect — `div_checked : (RaiseHandler<E>, Index, Index) -> Index`
— never its callers' resources; callers hold origins across calls using
the same local try/cleanup/re-raise scopes as anywhere else. What extends,
and what is genuinely deferred:

- **Call-site partiality** falls out of `Value.totality` once aggregate
  types taint: `function.call` arguments arrive packed, so a `Tuple` with
  a `Handler<Diverge>`-bearing component must itself count for totality —
  the products half of the "handlers inside aggregates" open question,
  with effectful calls as its forcing consumer. The coverage rule then
  applies at call sites unchanged.
- **Callee-side coverage terminus**: inside the callee the handler is a
  parameter — no local try introduced it. The rule's extension: a raise
  targeting a handler *parameter* must have an empty coverage set (no
  linear value open across it); a callee that holds origins across risky
  code wraps its own local try and re-raises to the parameter. Always
  achievable, no new ops.
- **Handlers are second-class**: passable downward as arguments, never
  returned, stored to memory, or captured into escaping function values.
  They are `layout Void`, so storage is meaningless anyway; a verifier
  rule makes the restriction explicit. This is what keeps handler
  resolution meaningful under inlining and lowering.
- **The only real gap is lowering.** `raise_catch_to_goto` resolves each
  raise to its try by handler identity, and `goto.Label`s are
  intra-function — that is the actual content of `docs/effects.md`'s "v1:
  no effects across functions". Outlined may-raise functions need a
  calling convention: the standard two-path result (tagged return; the
  call site branches on the tag, continuing or branching to its local
  except label), or inlining — which a JIT wants for composites anyway,
  and under which identity resolution just works. Either is a lowering
  strategy; none of the semantics above change with the choice.

Effect polymorphism and richer borrow inference remain out of scope, as in
`docs/effects.md`.

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
   API: `Reference<T>`, two-result allocation, `load(o, ref)`/`store(o, ref, v)`,
   `destroy`, `attach`. Mark `Origin` as `Linear` — the verifier picks it up
   with no plumbing (existing TODO).
4. **Migrate single-cell users** off fused `Reference<T>`; delete
   `Reference`'s `Linear`/`Handler` traits by folding it into `Reference<T>`.
5. **Migrate buffer users** (ndbuffer, record, existential lowerings,
   `passes/support/memory.py`) off mem tokens; delete `Buffer<T>` and the
   `mem` operands. Real `free` in destroy; stack lifetime markers.
6. **split/join + alias-aware reordering**: land the decomposition ops, then
   let passes use origin-forest disjointness (first client: loop
   parallelization / vectorization legality in the structured lowering).
7. **Observe contracts for loads** (read/read commutation) and the
   coverage rule for cleanup scopes (block drain is already landed
   behavior).
8. **`world` origin** for externs; retire ad-hoc chaining of effectful calls.

## Open questions

Each question is annotated with its *forcing point* — the migration step (or
external event) by which it must be resolved. Questions with no forcing point
are clean extensions: resolving them later strengthens the verifier or adds
ops, without revisiting decisions made here. (The two questions that had hard
forcing points — the destruct-block contract and the partial-op drain rule —
are resolved in the "Destructors" and "Interaction with raise" sections
above.)

- **Handlers inside aggregate types**: `Value.totality` classifies
  partiality from the *direct* types of dependencies. There is no art yet
  for a product, union, or existential that contains (or may contain) a
  `Handler<Diverge>` or a linear component. Intended rules: a `Tuple` with
  a handler component bears `Handler<Diverge>` for totality (taint); a sum
  that *may* contain one counts as one; linear components make the
  aggregate linear. *Forcing point:* effectful function calls —
  `function.call` arguments arrive packed, so without aggregate taint a
  handler argument hides inside a `Tuple` operand and the call is
  misclassified `TOTAL`. Sums/existentials force later, when they start
  carrying handlers or linear values.
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
- **Escape analysis**: a `Reference` outliving its origin is dangling. Linearity
  prevents the origin disappearing while *threaded* uses remain, but a stored
  `Reference` reloaded after `destroy` is not caught. Candidate: origins
  parameterize `Reference` types (`Reference<T, o>`) so staleness is a type error. This
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
