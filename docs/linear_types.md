# Linearity Verification

## Purpose

Verify that values of linear and affine types are used correctly across an IR
where ops own their semantics and may not all declare a full dataflow contract.

The verifier handles unknown ops by conservatively consuming any linear
operands; this may reject programs that would be sound under a more precise
contract, but never accepts programs that leak. Filling in an op's contract
makes the verifier more precise.

## Substructural types

Each value carries a multiplicity:

- `Linear`: must be consumed exactly once.
- `Affine`: must be consumed at most once.
- `Unrestricted`: not tracked.

## Block totality

Determined entirely by the block's signature (parameters, captures, operands,
result type):

- If the block has a `Handler<Diverges>` capability in its inputs and result
  type `Never`: `divergent`.
- If the block has a `Handler<Diverges>` capability in its inputs and result
  type non-`Never`: `partial`.
- If the block has no `Handler<Diverges>` capability: `total`. Result type
  must be non-`Never` (a `total` block claiming `Never` is ill-formed).

A block's totality determines what its body is permitted to do. The body's
correctness is verified separately by the per-block algorithm.

## Context

`Γ : Var → State` where State ∈ {Available, MaybeAvailable, Consumed}.

Lookup failure means "not in scope."

The third state is for ops the verifier doesn't yet have a precise contract
for; see "Block-holding ops" below.

## Per-block verification

Input: Γ_in.

For an entry block, Γ_in contains the function's substructural parameters and
captures, all `Available`.

For a child block, Γ_in contains the block's substructural parameters and
captures, all `Available`.

Algorithm:

1. Walk the body in dataflow order (topological order of the DAG rooted at
   the block's result).

2. For each op:
   - For each substructural operand consumed by the op: require Γ shows it
     `Available` or `MaybeAvailable`; transition to `Consumed`. Reject on
     double-consume.
   - For each substructural result produced by the op: bind as `Available`.
   - If the op holds child blocks: verify each child block independently
     (recursively), receiving pass/fail. The parent's Γ is updated using
     only the op's own signature, not the children's internal state.

3. At the root op, apply the exit check:
   - For each `Linear` value in Γ as `Available`: reject (leak).
   - For each `Linear` value in Γ as `MaybeAvailable`: permissive — ok.
   - For each `Affine` value in Γ: ok in any state.
   - For each value in Γ as `Consumed`: ok.

4. Pass.

## Block-holding ops

The op's signature declares how its operands and captures flow into its child
blocks and how its results are produced.

For each child block:
- Captures from the parent's Γ are accounted at op invocation according to
  the op's signature (see below).
- The child block is verified independently with Γ_in containing its
  parameters and captures as `Available`.

The op's effect on the parent's Γ is determined by the op's signature:
- Operands: transition to `Consumed`.
- Results: bound as `Available`.

The verifier does not consult the children's internal Γ. Each child block is
locally responsible for its own correctness, including consuming its captured
linear values by its root.

### Unknown block-holding ops

An op with owned blocks whose block-execution contract isn't known to the
verifier is handled conservatively. Today every op with blocks falls in this
category — there is no per-op contract framework yet, so the predicate
`_has_known_block_semantics(op)` returns `False` unconditionally (see
`dgen/ir/verification.py`). When that lands, a future fix.

For each capture into an unknown op's child block:

- `Available → MaybeAvailable`
- `MaybeAvailable → MaybeAvailable`
- `Consumed → reject` (cannot capture an already-consumed value)

`MaybeAvailable` says "the inner block might or might not have actually
consumed this." Multiple sibling unknown ops capturing the same value all
park at `MaybeAvailable` rather than each charging a `Consumed` transition.
This is what lets the goto-style if/else lower without tripping double-consume:
`%then` and `%else` are sibling `goto.label` ops both capturing `%exit`, and
neither op individually can be said to "definitely consume" `%exit` — only
the runtime path picks one.

Direct operand consumes still transition `MaybeAvailable → Consumed`: if a
known-semantics op afterwards uses the value as a plain operand, the verifier
trusts that explicit consume. The model is permissive on the unknown side
and precise on the known side; tightening happens by giving more ops their
contracts.

## Branch composition (at-most-once-alternative ops)

For an op like `if` whose alternatives are mutually exclusive:

- Captures into multiple alternative children of the same op are deduplicated
  at the parent — they contribute one transition (`Available → MaybeAvailable`
  while the op is unknown, or `Available → Consumed` when its contract is
  known and says so), not one per alternative.
- Each alternative child is verified independently with the captured value
  as `Available` in its Γ_in.
- Each alternative child is independently responsible for consuming its
  captured linear values by its root. If one alternative consumes and another
  doesn't, the non-consuming alternative fails its own verification (linear
  leak at root). The parent does not need to reconcile.

This means "branch disagreement" is detected through each alternative's local
verification, not through a parent-side join.

## Loops (zero-or-more-with-carry)

The body's substructural parameters include the loop carries, all `Available`
in the body's Γ_in.

The body's root must produce fresh values for each carry (matching types),
which become the carry inputs for the next iteration or the op's results
after the final iteration.

Each carry must be `Consumed` at body exit; it is "consumed" by being
re-yielded as a fresh carry value (the yield is the consumption). Non-carry
linear values introduced inside the body must be `Consumed` at body exit
(they cannot escape iterations).

The op's result rebinds the carry values as `Available` in the parent's Γ.

## Function-level

The function's signature declares its parameters, captures, and totality. The
body is a single block whose totality must match the function's declared
totality.

## Failure modes

- Double consume (consume of `Consumed`, or capture-into-unknown of
  `Consumed`).
- Linear leak (Linear value `Available` at block root and not the
  block result).
- Loop carry violation (non-carry value escapes body, or carry doesn't
  roundtrip with fresh values at body root).

`MaybeAvailable` at block exit is never a failure on its own — it's
permissive by design.

## Locality

Each block's verification consults only:
- The block's own ops and their signatures.
- The block's Γ_in.

Child block verification is recursive but each level is local — the parent
updates its Γ from its op's signature alone, and the child verifies
independently. No information flows back from child to parent except pass/fail.
