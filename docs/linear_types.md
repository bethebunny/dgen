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

`Γ : Var → State` where State ∈ {Available, Consumed}.

Lookup failure means "not in scope."

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
     `Available`; transition to `Consumed`. Reject on double-consume.
   - For each substructural result produced by the op: bind as `Available`.
   - If the op holds child blocks: verify each child block independently
     (recursively), receiving pass/fail. The parent's Γ is updated using
     only the op's own signature, not the children's internal state.

3. At the root op, apply the exit check:
   - For each `Linear` value in Γ as `Available`: reject (leak).
   - For each `Affine` value in Γ as `Available`: ok.
   - For each value in Γ as `Consumed`: ok.

4. Pass.

## Block-holding ops

The op's signature declares how its operands and captures flow into its child
blocks and how its results are produced.

For each child block:
- Captures from the parent's Γ are consumed at op invocation (transition to
  `Consumed` in parent).
- The child block is verified independently with Γ_in containing its
  parameters and captures as `Available`.

The op's effect on the parent's Γ is determined by the op's signature:
- Operands: transition to `Consumed`.
- Results: bound as `Available`.

The verifier does not consult the children's internal Γ. Each child block is
locally responsible for its own correctness, including consuming its captured
linear values by its root.

## Unknown ops

An op without a declared contract is handled conservatively:

- Captures into its children are still known (captures are syntactic). Each
  captured-by-value linear value transitions to `Consumed` in the parent.
- Each child block is verified normally with its captures as `Available`. The
  child's local correctness is checked.
- For each substructural operand of the op: in the absence of a declaration,
  treat as consumed (transition to `Consumed` in parent).
- For each substructural result the op produces: bind as `Available` in
  parent.
- The op's totality, if undeclared, defaults to `partial`.

This is sound (never accepts a leaking program) but may reject programs that
would be accepted under a more precise contract. The remedy is to declare the
op's contract, not to make the verifier guess.

## Branch composition (at-most-once-alternative ops)

For an op like `if` whose alternatives are mutually exclusive:

- Captures into multiple alternative children consume the source value once
  from the parent's Γ at op invocation, not once per alternative.
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

- Double consume.
- Linear leak (Linear value `Available` at block root).
- Loop carry violation (non-carry value escapes body, or carry doesn't
  roundtrip with fresh values at body root).

## Locality

Each block's verification consults only:
- The block's own ops and their signatures.
- The block's Γ_in.

Child block verification is recursive but each level is local — the parent
updates its Γ from its op's signature alone, and the child verifies
independently. No information flows back from child to parent except pass/fail.
