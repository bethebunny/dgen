# Design: Effects and Origins

## Status

Proposal. Replaces mem tokens. Introduces a minimal effect system. Targets CPS lowering through the goto dialect.

## Goals

- A formally solid effect system with a small surface area.
- Raise/catch for error handling, lowered to CPS within functions.
- Origins as an instance of the effect framework, replacing mem tokens and tracking destruction.
- No effect querying. Effects are declared at op boundaries; type-checker enforces locality.

## Non-goals

- Effects crossing function boundaries. All effects and handlers are local to a single function body for v1.
- Effect polymorphism. Row polymorphism. Effect inference.
- Automatic error propagation or Result wrapping.

## Framework

### Effect and handler traits

```
trait Effect
trait Handler<effect_type: Effect>
```

An effect type carries no data at the type level beyond its parameters; it is a tag that names the effect.

A handler is a runtime value whose type implements `Handler<E>`. Handler values are passed as parameters to effect operations and are consumed (or scoped) by handler-introducing ops.

### Effect ops

Every observable effect has:

1. A family of **primitive ops** that take a handler as parameter and unconditionally perform the effect.
2. One or more **handler-introducing ops** that scope a handler over a block, discharging the effect.

Primitive ops take `Some<Handler<E>>` as a parameter. Handler-introducing ops accept a body block where the handler is in scope.

### Raise and catch

```
type Raise<error_type: Type>:
    layout Void
    has trait Effect

type RaiseHandler<error_type: Type>:
    layout Void
    has trait Handler<Raise<error_type>>

op raise<error_type: Type, handler: RaiseHandler>(error: error_type) -> Never

op catch<error_type: Type>() -> RaiseHandler<error_type>:
    block on_raise
```

Semantics:

- The `catch` op evaluates to a fresh `RaiseHandler<error_type>` value. The
  handler flows through the enclosing dataflow like any other SSA value; its
  live range is the region in which the catch is active.
- `raise<handler>(error)` unconditionally transfers control to the `on_raise`
  block of the `catch` that produced `handler`. The `on_raise` block receives
  `error` as its block argument.
- `on_raise`'s result type must be `Never` — the block must escape (re-raise
  to an outer handler, diverge, or, once function-scope return exists,
  terminate the function). v1 does not provide a built-in merge that spliced
  `on_raise`'s result back into the body's dataflow; users who need recovery
  arrange it explicitly via the enclosing control flow.
- Handlers are values. They can be captured by nested blocks. If `on_raise`
  captures a handler from an outer catch, it may invoke raise through that
  handler; nested handling falls out of normal scoping.
- A raise whose handler is not dominated by any live catch is undefined
  behavior in v1 (see "Functions" below).

### Composite "may-raise" ops

Composite ops like `div_checked` are not primitives. They are compositions containing an explicit conditional and an internal raise:

```
// conceptual:
def div_checked<handler: Some<Handler<Raise<DivByZero>>>>(a, b):
    if b == 0:
        raise<handler>(DivByZero)
    else:
        div(a, b)
```

No "may-raise" annotation on the primitive IR. The raise is local and explicit.

### Lowering

Raise/catch lowers to CPS via the goto dialect:

- The `catch` op is erased. Its result (the `RaiseHandler` value) carries no
  runtime representation — `RaiseHandler` has `layout Void`.
- Its `on_raise` block becomes a `goto.label` placed in the enclosing scope.
- Each `raise<handler>(error)` becomes a `goto.branch` targeting the
  `goto.label` derived from the catch that produced `handler`, passing
  `error` as the block argument.

Handler resolution is compile-time: `raise<handler>` walks `handler` back to
its originating `catch` (by following the use-def graph through
`ChainOp`/identity ops). The `raise` op becomes a direct branch. No runtime
dispatch. No evidence value survives codegen.

## Functions

Out of scope for v1. All effect ops and handlers must be local to a single function body. How effects interact with function boundaries — propagation, transformation, polymorphism, or handoff — is deferred to a future design.

Concretely, for v1:

- A function body containing an unhandled primitive effect op is not a defined case. Initial implementation may assume all effects are handled within the function; behavior otherwise is undefined.
- Effect ops that would require a handler from outside the function have no mechanism to obtain one.

This limits the scope of what can be expressed (no cross-function error propagation, no effect-polymorphic functions) but keeps the design minimal and correct for its v1 goals.

## Origins

### Motivation

Origins replace mem tokens. They track provenance for alias analysis and carry destruction obligations. Each origin is a linear value: using it produces a new origin; the old one is consumed. This preserves the dataflow ordering the mem pattern provided and adds static guarantees.

An origin is simultaneously:

- Evidence for memory operations (loads/stores take an origin, produce a new origin).
- A handler for the `NeedsDestructor` effect, which discharges when the origin is destroyed.

### Types and ops

```
type NeedsDestructor:
    data: Nil
    has trait Effect

type Origin:
    parent: memory.Reference<Origin>
    has trait Handler<NeedsDestructor>

op origin(ref: memory.Reference<T>) -> Origin:
    body destruct(ref: memory.Reference<T>)

op destroy(o: Origin) -> Nil
```

Semantics:

- `origin(ref) destruct(ref): ...` creates a fresh origin associated with `ref`. The `destruct` block defines what runs when the origin is destroyed.
- `destroy(o)` consumes the origin, running its destruct block. After destroy, `o` is no longer usable.
- Memory ops (`memory.load`, `memory.store`) take an origin and produce a new origin. They thread linearly through dataflow, matching the current mem pattern.

### Example

```
%ref : memory.Reference<Float64> = memory.heap_allocate<Float64>()
%o : Origin = origin(%ref) destruct(%ref : memory.Reference<Float64>):
    memory.dealloc(%ref)
%o' : Origin = memory.store(%o, %ref, 3.14)
%_ = destroy(%o')
```

`%o` and `%o'` are distinct origins. `%o` is consumed by the store; `%o'` is consumed by destroy. The destruct block runs during destroy, deallocating the reference.

### Linearity

Origins are linear. Every origin value must be consumed exactly once by either:

- A memory op (consumes the input origin, produces a new one).
- A destroy op (consumes, discharges the destructor).
- A function return (consumes, transfers ownership to the caller).

Linearity is enforced by a pass verifier — see "Linearity verifier" below.

### Alias forest

Origins form a forest. Fresh roots come from allocation ops (`heap_allocate`, `stack_allocate`, etc.). Sub-origins come from structural decomposition ops (GEP, field projection) and are children of the source origin in the forest.

Aliasing rule:

- Two origins alias iff one is an ancestor of the other in the forest.
- Siblings (sub-origins with a common parent from structural decomposition) are disjoint.
- Origins in different trees are disjoint.
- Unknown origins (rare, from escaped pointers) are top: they may alias anything.

Structural decomposition is limited to ops the type system recognizes as producing disjoint sub-origins. Arbitrary pointer arithmetic is not supported; that would break disjointness guarantees.

### Destruction ordering

Destruction fires when `destroy` is called on an origin. The destruct block has access to the origin's captured context (e.g., the reference it was created with).

Sub-origins must be destroyed before their parents. The type system enforces this: destroying a parent while a sub-origin is still live is a type error — the sub-origin's linear thread has not reached destroy.

This also means a destructor cannot run while sub-origins of its target are live, which is the invariant required for safe cleanup.

### Returning references and ownership transfer

A function returning an origin transfers ownership to the caller. The function produces an origin value (linear), returning consumes it in the function's scope, and the caller receives a new linear origin. The destructor obligation transfers with the value.

Example:

```
def make_thing() -> Origin:
    %ref = memory.heap_allocate<T>()
    %o = origin(%ref) destruct(%ref):
        memory.dealloc(%ref)
    return %o   // ownership transfers to caller
```

The caller now owns `%o`. They must consume it exactly once.

References to values inside an origin work similarly: returning a `memory.Reference<T>` paired with (or derived from) an origin transfers access. Returning a reference without its owning origin would be a dangling reference; linearity prevents this because the origin must go somewhere, and it cannot be both returned to the caller and destroyed locally.

For v1:

- Functions may return origins. Returning transfers ownership.
- Functions may return references paired with their owning origin. The pair is linear in the origin.
- Functions may not return references that escape their owning origin. Enforced by linearity.

No new ops or types are required beyond linearity enforcement.

### Lowering

Origins lower by erasure. The origin value itself has no runtime representation beyond what its `memory.Reference` field already carries. At codegen:

- `origin(ref) destruct(...)` registers `destruct` as the cleanup action associated with this dataflow position. No runtime value is emitted.
- Memory ops lower to normal loads/stores; the origin threading is pure compile-time.
- `destroy(o)` lowers to the inlined destruct block.

The linear thread of origin values enforces ordering; codegen emits the destruct block at the destroy point. No runtime bookkeeping.

### Linearity verifier

A pass verifier enforces origin linearity:

1. Every origin-typed value has exactly one consuming use.
2. Consuming uses are: memory ops (consume and produce), destroy ops (consume and discharge), or function returns (consume and transfer).
3. An origin-typed value with zero uses at a block terminator is an error (leak).
4. An origin-typed value with multiple uses is an error (aliasing of linear value).
5. Sub-origins must be consumed before their parent origin.

The pass runs on every function after origin-introducing transformations. Failures are reported with the offending origin value and its use sites.

## Interaction with actors

Effects for actor support will reuse the same framework. Expected effect types (to be designed in a separate doc):

- `Send<M>` — evidence required to send a message of type M to some actor.
- `Spawn` — evidence required to create a child actor.
- `Supervise<F>` — evidence that failures of type F can be caught by a supervisor.

These will follow the same pattern: primitive ops take a handler parameter; handler-introducing ops scope the handler over a block. Lowering strategies will vary — some may lower to runtime calls rather than CPS — but the type-level machinery is shared.

## Implementation plan

1. Add `Effect` and `Handler<E>` traits.
2. Implement Raise/catch in the IR. Verify CPS lowering through goto.
3. Introduce origin ops alongside existing mem tokens. Migrate memory ops to take origins.
4. Remove mem tokens once all memory ops use origins.
5. Implement the linearity verifier pass.
6. Implement forest-based alias analysis using origin provenance.
7. Formalize the destructor effect and destroy lowering.
8. Actor effects (separate doc and implementation).

## Open questions

- Structural sub-origin ops: which specific ops produce sub-origins, and how is their disjointness proved? Needs enumeration per concrete op (struct field GEP, array index, slice).
- Result-type compatibility in `catch`: when body returns `T` and on_raise returns `T'`, is equality required, or is a least-upper-bound computed? Initial: require equality, with `Never` universally compatible.
- Escape analysis for references: how are references that *could* escape their origin detected and rejected? Linearity handles the common case; pathological cases (closures capturing references) may need additional analysis.
