# Plan: Actor Framework with SDF Scheduling

## Context

DGEN's staging system automatically resolves compile-time-known parameters eagerly and
generates JIT callback thunks for runtime-dependent parameters. This is a natural fit for
synchronous dataflow (SDF) actor scheduling, where fusion and buffer allocation decisions
depend on actor rates that may be statically known or determined at runtime.

This plan implements a single-port linear pipeline model inspired by StreamIt. Actors have
one input and one output, each with a declared rate. The compiler computes a repetitions
vector from the rates, decides fusion (matching rates → shared loop) vs separation
(mismatched rates → separate loops with intermediate buffer), and lowers to the affine
dialect.

The same pipeline definition works with static rates (compile-time fusion) or runtime rates
(JIT callback for dynamic fusion). This demonstrates DGEN's core value proposition: one
definition, two execution strategies, zero user-visible difference.

### Prior art

- StreamIt (MIT, 2002): single-port filters with `push`/`pop`/`peek` rates, `pipeline`
  and `splitjoin` composition. Schedule computed automatically from balance equations.
- Lee & Messerschmitt (1987): SDF balance equation `T * q = 0` for repetitions vector.
- FAUST: algebraic composition with implicit scheduling and full fusion.

### Relationship to existing code

The `actor/` directory contains a prototype with a `PipelineOp` that hardcodes a
multiply-then-add computation. This plan replaces that with general-purpose actors whose
bodies are defined in IR.

## Design

### Dialect definition (`actor/dialects/actor.dgen`)

```dgen
from builtin import Index, Nil, HasSingleBlock

op actor<consume_rate: Index, produce_rate: Index>(input):
  block body
  has trait HasSingleBlock

op produce(value)

op pipeline():
  block body
  has trait HasSingleBlock
```

**`actor`**: A dataflow actor. Consumes `consume_rate` tokens from `input` per firing,
produces `produce_rate` tokens via the `produce` op in its body. The body block receives
the consumed value as a block argument.

**`produce`**: Designates the output value of an actor firing. Must appear exactly once
in an actor's body (enforced by verification, not syntax).

**`pipeline`**: A composition scope containing a linear chain of actors. Actors chain via
use-def: if actor B takes actor A as its `input` operand, A's output feeds B's input.

### Wiring via use-def

Actors in a pipeline form a chain through their `input` operands:

```
%pipe = actor.pipeline() ():
  %a = actor.actor<1, 1>(%pipe_input) ():
    %x = ...  // body sees consumed value as block arg
    actor.produce(%result_a)
  %b = actor.actor<1, 1>(%a) ():
    %y = ...
    actor.produce(%result_b)
  return(%b)
```

`%b`'s input is `%a` — that's the edge. No explicit `connect` op needed. The pipeline's
input is threaded as the first actor's input. The pipeline's output is the last actor's
result.

### Scheduling

The lowering pass (`ActorToAffine`) walks the pipeline body, extracts the actor chain,
and computes a schedule:

1. **Extract chain**: Follow use-def from the pipeline's return value back through actor
   `input` operands to build the ordered actor list.
2. **Compute repetitions**: For each adjacent pair (A, B), check if A's `produce_rate`
   equals B's `consume_rate`. If equal, they can fuse (fire together in one loop). If not,
   compute the repetitions vector from the balance equation.
3. **Emit loops**: For each fusible group, emit a single `affine.for` loop containing the
   inlined bodies of all actors in the group. For non-fusible boundaries, emit separate
   loops with `affine.alloc` intermediate buffers.

### Lowering detail: fused group

For actors A and B with matching rates R:

```
%output = affine.alloc([R])
affine.for(0, R) (%i):
  %val = affine.load(%input, [%i])
  // inline A's body, replacing block arg with %val, produce with %mid
  // inline B's body, replacing block arg with %mid, produce with %result
  affine.store(%result, %output, [%i])
```

The `produce` op in A's body becomes a direct value (`%mid`) consumed by B's body.
The `produce` op in B's body becomes the `affine.store`.

### Lowering detail: unfused boundary

For actors A (produce_rate=P) and B (consume_rate=C) where P != C:

```
%buffer = affine.alloc([rep_A * P])
affine.for(0, rep_A) (%i):
  // A fires rep_A times, each producing P tokens
  affine.for(0, P) (%j):
    %idx = add_index(%i * P, %j)
    %val = affine.load(%input, [%idx])
    // inline A's body
    affine.store(%result, %buffer, [%idx])

%output = affine.alloc([rep_B * C])
affine.for(0, rep_B) (%k):
  affine.for(0, C) (%l):
    %idx = add_index(%k * C, %l)
    %val = affine.load(%buffer, [%idx])
    // inline B's body
    affine.store(%result, %output, [%idx])
```

Where `rep_A` and `rep_B` satisfy the balance equation: `rep_A * P = rep_B * C`.

### Staging integration

Rates are `__params__` (angle-bracket syntax in `.dgen`). The staging system handles
two scenarios automatically:

- **Static rates**: `Index().constant(4)` — stage-0, resolved at compile time. The
  lowering pass sees concrete integers and emits specialized loops.
- **Dynamic rates**: Block arguments of type `Index` — stage-1, resolved at runtime.
  The staging system builds a callback thunk that JIT-compiles the lowering when rates
  become known. Same lowering pass, same code, different timing.

## Implementation steps

### Step 1: Dialect definition and generation

- Rewrite `actor/dialects/actor.dgen` with the new ops (actor, produce, pipeline)
- Regenerate `actor/dialects/actor.pyi`
- Delete the old `PipelineOp`-based code

### Step 2: Lowering pass (`actor/passes/actor_to_affine.py`)

- Rewrite `ActorToAffine` to:
  - Walk pipeline body and extract actor chain via use-def
  - Identify fusible groups (adjacent actors with matching rates)
  - Inline actor bodies into affine loops
  - Replace `produce` ops with stores (unfused) or direct values (fused)
  - Emit intermediate buffers for unfused boundaries

### Step 3: Tests (`actor/test/test_actor.py`)

- Test fused pipeline: two actors with matching rates → one loop
- Test unfused pipeline: two actors with mismatched rates → two loops + buffer
- Test three-actor chain with mixed fusion (A-B fuse, B-C don't)
- IR snapshot tests for lowered output
- End-to-end JIT execution tests

### Step 4: Demo script (`actor/demo.py`)

- Build a pipeline with concrete actors (e.g., gain + downsample)
- Run with static rates (compile-time fusion)
- Run with dynamic rates (JIT callback)
- Print IR at each stage to show the difference

### Step 5: Verification

- Add `verify_preconditions` / `verify_postconditions` to `ActorToAffine`
- Check: each actor body has exactly one `produce` op
- Check: pipeline body contains only actor ops (no stray ops)
- Check: actor chain is linear (each actor consumed by at most one other)

## Future work: multi-port actors

The single-port model extends to multi-port via:

```dgen
op actor():
  block body
  has trait HasSingleBlock

op port_in<rate: Index, channel: Index>()
op port_out<rate: Index, channel: Index>(value)

op splitjoin(input):
  block body
  has trait HasSingleBlock
```

Where `channel` identifies which port (0, 1, 2...) and the `splitjoin` op handles
fan-out/fan-in patterns. Rates move from the actor to individual ports. The lowering
pass would need to analyze the port connectivity graph rather than a linear chain.

This is deferred because single-port linear pipelines already demonstrate the core
staging story and keep the implementation focused.
