# Linear Types

## Status

Implemented (v1). Type-level traits + a verifier; no IR rewrites.

## Goal

A generic resource-discipline check that's symmetric with
`Value.totality`. Two marker traits (`Linear`, `Affine`) classify a
type, and a single verifier — invoked from every pass's
`verify_preconditions` and `verify_postconditions` — enforces:

- **Linear**: every value of a linear type is consumed *exactly once*.
- **Affine**: every value of an affine type is consumed *at most once*.
- A potentially-divergent op (`Totality.PARTIAL`) must drain every
  in-scope linear value before it runs, otherwise the divergent path
  would leak the unmet obligation.

The mechanism is a structural over-approximation: we count direct uses
in the use-def graph. It's tight enough for the resources we have today
(`RaiseHandler`, future `Origin`), and explicitly leaves out
control-flow-sensitive cases (see "Reentrant continuations" below).

## Non-goals

- Quantitative linearity (drop / copy / exact-N multiplicities).
- Control-flow-sensitive linearity. The verifier counts every static use
  site as a consume, regardless of whether the op is on a `then` vs
  `else` branch or inside a loop.
- Cross-function linear plumbing. Effect ops are local to a function
  body in v1 (per `docs/effects.md`).

## Traits

```dgen
trait Linear
trait Affine
```

Both are zero-parameter marker traits — declared in
`dgen/dialects/builtin.dgen` next to `Effect` / `Handler`. A type
declares membership with `has trait Linear` or `has trait Affine`.

### What's marked today

| Type                     | Trait    | Why                                      |
| ------------------------ | -------- | ---------------------------------------- |
| `error.RaiseHandler<E>`  | `Affine` | A raise handler may be invoked zero or one times along any one execution path; `raise<handler>(err)` consumes it. |
| `goto.Label`             | (none)   | Reentrant — see below.                   |
| `Origin` (future)        | `Linear` | Per `docs/effects.md`, an origin is consumed exactly once by a memory op, `destroy`, or function return. |

### Reentrant continuations

`goto.Label` is *not* marked Affine even though it is a divergence
handler. A label is a reentrant continuation: a region's `%self` is
branched to once per loop iteration; many branch sites can target the
same `%exit`. The simple "consumed at most once" Affine rule would
flag every realistic goto-form lowering as a `DoubleConsumeError`.
Tracked in `TODO.md` under "Type system / effects".

## API

```python
from dgen.type import Linearity

class Linearity(enum.Enum):
    UNRESTRICTED = "unrestricted"
    AFFINE = "affine"
    LINEAR = "linear"

# On any Value:
v.linearity              # -> Linearity
v.is_linear              # -> bool
v.is_affine_or_linear    # -> bool   # the verifier's main predicate
```

Read off the value's *type* (not the value's own declared traits): a
value of type `T` is `LINEAR` iff `T has trait Linear`. `Type`
instances themselves always return `UNRESTRICTED` — types-as-values
are universe-1 metadata, not resources subject to linearity.

`Value.linearity` defers `Linear`/`Affine` imports to function scope to
break the `dgen.type` ↔ `dgen.dialects.builtin` cycle (same trick
`totality` uses for `Diverge` / `Handler`).

## Verifier semantics

Three core relations:

- **`consumed_by(v)`** — affine-or-linear values `v` directly references
  one hop out: `{d for d in v.dependencies if d.is_affine_or_linear}`.
  Type instances are filtered.
- **`introduced_in(block)`** — affine-or-linear values whose definition
  lives in this block, plus any captures (which the inner block is on
  the hook for).
- **`live_at(op_index, block)`** — running set during the topo walk of
  `block.values`: linear values introduced earlier and not yet consumed
  by an earlier op.

The verifier walks each block once in `block.values` order (already
post-order topological from `transitive_dependencies`):

### Rule 1 — at most one direct consumer (Affine *and* Linear)

Bump a per-value refcount on each direct consume. `refcount[v] > 1`
raises **`DoubleConsumeError`**.

### Rule 2 — no leak of linear values at block exit

After the walk, every introduced `Linear` value must satisfy
`refcount[v] >= 1` *or* `v is block.result`. Otherwise raise
**`LinearLeakError`**. Affine values are exempt — un-consumed is fine.

### Rule 3 — PARTIAL ops drain linear values

When the walk reaches an op whose `totality is PARTIAL`, the
`live_linear` set (after accounting for this op's own consumes) must be
empty. If anything remains, the divergent path would leak it — raise
**`LinearLeakAtPartialError`**. Affine exempt: an affine value being in
scope at a divergence point is the normal case (e.g. the divergence
handler itself).

## Block boundaries

Captures count once at the parent op. `Value.dependencies` walks
`block.dependencies` for owned blocks, which yields the block's
captures — so capturing an affine-or-linear value into an inner block
shows up as a single direct consume by the parent op.

Each owned block then runs its own independent verification walk. For
that walk, captures are treated as `introduced` for the inner block —
this lets the inner block's double-use and PARTIAL-drain checks fire
on the captured value as if it were a fresh resource leaf for the
inner block's lifetime.

The top-level entry wraps the input value in a synthetic
`Block(result=value)` so the same algorithm covers both block bodies
and bare values, mirroring `Pass.run`.

## Edge cases

- **`Type` values.** Filtered out of `consumed_by`; `Value.linearity`
  short-circuits to `UNRESTRICTED` for `Type` instances.
- **`Constant` and `BlockArgument`.** Subject to the same rules as
  ops — if their type is linear/affine, they're tracked.
- **State across passes.** The verifier holds no module-level cache;
  state lives in plain locals per call. Ops mutate via
  `replace_uses_of` between passes, so any persistent cache would have
  to be invalidated; the per-call sweep is cheap enough that no
  memoization is needed.

## Failure modes

- **`DoubleConsumeError`** — an affine-or-linear value is consumed by
  more than one op.
- **`LinearLeakError`** — a linear value is introduced and never
  consumed and is not the block result.
- **`LinearLeakAtPartialError`** — a `PARTIAL` op is reached while
  linear values remain live in scope.

All three subclass `LinearityError`, which subclasses
`VerificationError`.

## Pipeline integration

`verify_linearity` is called from both `Pass.verify_preconditions` and
`Pass.verify_postconditions` (`dgen/passes/pass_.py`). The opt-in
context var `verify_passes` (`dgen/passes/compiler.py`) gates whether
the hooks fire — same shape as the existing structural verifiers.

## Relationship to effects

Divergence (`docs/control-flow.md`) and linearity meet at
`Totality.PARTIAL`. A PARTIAL op signals "control may not return". The
linearity rule says: before it runs, every linear obligation in scope
must be discharged. This makes resource-leaks on the divergent path a
verifier-time error rather than a runtime bug.

`Origin` (per `docs/effects.md`) is the canonical linear value; once
the origin work lands it just appends `has trait Linear` to its type
definition and the verifier picks it up.

## Open questions

- **Reentrant / control-flow-sensitive Affine for `Label`.** A second
  trait — say `JoinPoint` — could carry "branched to many times,
  evaluated at most once per execution path" semantics. Not in v1.
- **Quantitative / parametric linearity.** Drop, copy, exact-N
  multiplicities are a separate design.
- **Cross-function linearity.** Once functions stop being effect-local
  (post-v1), the verifier needs a way to talk about linear values
  flowing across call boundaries.
