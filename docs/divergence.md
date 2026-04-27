# Design: Generic Divergence Detection

## Status

Implemented (v1). Type-level only — no IR rewrites or new ops. The
classification rules below are pure functions over an op's existing
operands, owned-block captures, and result type.

## Goal

Provide a single generic predicate ``Value.totality`` that classifies any
op as **Total**, **Partial**, or **Divergent** with respect to non-local
control transfer ("divergence"). The classification has to compose across
dialects: ``error.raise``, ``goto.branch``, future ``actor.send`` failures
and any other "may not return" effect should all light up under the same
query without per-effect special cases in the consumer.

## Non-goals

- Inferring divergence from operation *semantics* (e.g. an infinite loop
  built out of ``algebra`` ops). The classification is type-evidence-based.
- Effect polymorphism or a row-effect calculus.
- Lowering or codegen behaviour. Knowing that an op is Divergent is useful
  for verifiers, optimizers, and pretty-printers; it doesn't change how the
  op is emitted.

## Framework

### The ``Diverge`` effect

```dgen
type Diverge:
    layout Void
    has trait Effect
```

``Diverge`` is the umbrella effect for "control transfer that may not
return". It is the *sole* effect that the generic detection rule observes.
Other diverging effects identify themselves as ``Diverge`` so a single
``Handler<Diverge>`` query suffices.

### Evidence: ``Handler<Diverge>``

Per the existing effect framework (``docs/effects.md``), a value that *can*
trigger an effect ``E`` carries a ``Handler<E>`` — typically a runtime
operand or a value captured from an outer scope. For divergence we look at
``Handler<Diverge>``:

| Type                       | Declares                                  |
|----------------------------|-------------------------------------------|
| ``error.RaiseHandler<E>``  | ``Handler<Raise<E>>`` *and* ``Handler<Diverge>`` |
| ``goto.Label``             | ``Handler<Diverge>``                      |

The dual declaration on ``RaiseHandler`` is a TODO: dgen does not yet
support type subtyping, so until ``Raise<E> <: Diverge`` is expressible the
handler redundantly declares both traits. See the TODO comment in
``dgen/dialects/error.dgen``.

### Classification rule

For a value ``v``:

```
partial(v)  ≡  ∃ operand of v.            type(operand) has trait Handler<Diverge>
            ∨  ∃ block ∈ v's owned blocks.
               ∃ capture of block.        type(capture) has trait Handler<Diverge>

totality(v) =  TOTAL      if not partial(v)
            =  DIVERGENT  if  partial(v) ∧ v.type = Never
            =  PARTIAL    otherwise
```

Equivalent invariants:

- A **Total** op never produces ``Never``.
- A **Divergent** op always produces ``Never``.
- A **Partial** op produces a normal result on the non-divergent paths and
  ``Never`` on the divergent ones, but its result type ``T`` reflects the
  former.

### Why operands and captures, not parameters

Compile-time parameters (``op branch<target: Label>(...)``) don't survive
into the runtime use-def graph the same way operands and captures do.
Detecting divergence via parameters would force consumers to special-case
"is this label argument really used?" Restricting the rule to operands and
captures keeps the predicate uniform — divergence is gated on a runtime
value being in scope.

This is a deliberate v1 choice. It means today's ``goto.branch``, which
takes its target as a parameter, is *not* flagged as Divergent by the
generic predicate — only ops that consume a label as an operand or capture
are. If a future revision wants ``branch`` to be detected, the right move
is to refactor ``branch`` to take its label as an operand, not to bend the
classification rule.

## API

```python
from dgen.builtins import Totality

class Totality(enum.Enum):
    TOTAL = "total"
    PARTIAL = "partial"
    DIVERGENT = "divergent"

# On any Value:
v.totality  # -> Totality
```

The property is defined on ``Value`` (in ``dgen/type.py``) so every value
in the IR — ops, types, constants, block args/params — answers the query
uniformly. Types and constants always return ``TOTAL`` (no operands, no
blocks, no captures); the interesting cases are ``Op`` values.

## Worked examples

```
%c : Index = 7                                  # constant       → TOTAL
%t : Index = error.try<Index>() body<%h: ...>:  # try            → TOTAL
    %r : Never = error.raise<Index>(%h, %c)     # raise          → DIVERGENT
except(%e: Index):
    %z : Index = 0
```

The ``try`` is TOTAL: the handler ``%h`` is bound as a body *parameter*,
not consumed by the try op itself. The ``raise`` is DIVERGENT: ``%h`` is an
operand, ``%h.type`` is ``RaiseHandler<Index>`` which carries
``Handler<Diverge>``, and the result type is ``Never``.

```
%outer : Index = error.try<Index>() body<%ho: ...>:
    %inner : Index = error.try<Index>() body<%hi: ...>:    # inner try → PARTIAL
        %ok : Index = 5
    except(%err: Index) captures(%ho):                     # captures outer handler
        %re : Never = error.raise<Index>(%ho, %err)
except(%err: Index):
    %z : Index = 0
```

The inner try is PARTIAL: its except block captures ``%ho``, whose type is
``RaiseHandler<Index>`` (Handler<Diverge>). Its own result type is still
``Index``, so it is not Divergent — control reaches a normal result via the
inner body's success path.

## Implementation notes

- The property lives on ``Value`` in ``dgen/type.py``. Imports of
  ``Totality``, ``Handler``, ``Diverge``, and ``Never`` are deferred to
  function scope to break the ``dgen.type`` ↔ ``dgen.builtins`` ↔
  ``dgen.dialects.builtin`` import cycle. ``TYPE_CHECKING`` carries the
  ``Totality`` annotation for type checkers.
- ``has_trait`` does structural equality on type ASM, so the
  ``Handler(effect_type=Diverge())`` literal in the property body is a
  cheap lookup key, not an object-identity comparison.
- The rule examines only the value itself — no transitive walk. A
  ``ChainOp`` whose ``rhs`` is a ``raise`` is **not** itself Divergent;
  divergence is a property of the divergent op, not of every op
  downstream of it. Consumers that want a "may-this-block-diverge"
  predicate should walk the block's ops with this property.

## Open questions

- Should ``BranchOp`` (and friends) become operand-taking so the generic
  rule catches them? Or do branches deserve a distinct ``Terminator``
  classification orthogonal to ``Diverge``?
- When dgen gains type subtyping, ``RaiseHandler<E>`` should drop the
  redundant ``Handler<Diverge>`` trait and inherit it via ``Raise<E> <:
  Diverge``. The dual declaration today is a forward-compat shim.
- ``Totality`` is the working name for the enum. ``Termination`` or
  ``Convergence`` are reasonable alternatives if the project later
  prefers one of them.
