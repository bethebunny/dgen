# Review: the June-21 PR cluster (#185, #186, #187, #188, #189, #190)

*Reviewed 2026-07-08 against `main` (`5df0acb`). All diffs read in full; the
suite was run on `main` (10 failed / 824 passed), on #186 alone (835 passed),
and on a trial merge of the full cluster (#186 → #188 → #190 → #185 → #189),
which is green (**837 passed**) after a single one-line reconciliation
(`ChainOp(lhs=…, rhs=…)` → `(result=…, effect=…)` in
`examples/dcc/passes/thread_loop_memory.py:125` — the exact semantic conflict
the PR descriptions predicted; git merges it silently, so it must be fixed
deliberately).*

---

## What the cluster is actually about

The IR's core axiom is *ordering = use-def edges, nothing implicit*. The
Buffer/mem-token substrate honors that in straight-line code (each load/store
consumes the previous token), but **loops broke the axiom**: back-edges carried
no tokens, so iteration N+1's loads read the *pre-loop* token and
cross-iteration ordering existed only implicitly, via the alloca. Commit
`08a7a51`'s tuple-body assertion made `ControlFlowToGoto` reject dcc's
zero-carry `Nil`-bodied loops, turning the latent modeling gap into 10 red
tests.

The cluster's answer, rather than relaxing the assertion:

- **#186** — a dcc pass (`ThreadLoopMemory`) that threads a `Nil` effect token
  through every loop carry, plus a written **loop concurrency contract**
  (token-threading carry ⇒ sequential iterations; no carry ⇒ reorderable).
- **#188** — the same threading for toy's one true reduction
  (`CountNonzero`), enabled by generalizing `lower_for` to real carries.
- **#190** — carry arity/type consistency preconditions in `ControlFlowToGoto`
  (stacked on #188).
- **#185** — renames `chain(lhs, rhs)` → `chain(result, effect)` and corrects
  the docs' false "establishes ordering" claim.
- **#189** — `assert_valid_llvm` next to IR-text snapshot asserts.
- **#187** — the roadmap / ongoing-work docs (reviewed separately in
  `2026-07-08-design-review.md`).

### My hypothesis of the direction

This is groundwork for a **future loop parallelizer** (the actor/affine-fusion
north star keys off absence of cross-iteration dependencies), built on the
*legacy* Buffer substrate rather than waiting for origins. That is consistent
with the roadmap's own decision that dcc rides `Buffer` permanently and that
origins (H2.2) come later for the safe dialects. The "right" long-term
mechanism for this is #184's linear `Reference`/`State` — a typed, *verifier
enforced* version of exactly this threading — but that path is blocked on
loop-carry linearity support (TODO.md notes the verifier can't model carry
pairs yet). So the cluster is a deliberate bridge: correct axiomatics, shim
substrate.

## Design assessment (treating the choices as suspect)

**The contract's polarity is the real risk.** "No token carry ⇒ concurrent"
makes *parallelizable* the default, opted out of by remembering to thread. It
is enforced purely by construction discipline: #190 verifies carry
*consistency*, but nothing verifies carry *presence* — a frontend bug, or any
future pass that introduces/moves buffer ops in a loop, silently produces a
"concurrent" racy loop, and the failure will surface only when a parallelizer
exists, far from the cause. Within dgen's philosophy the contract is the
*consistent* reading (absence of edges = absence of ordering; frontends own
edge creation), so I accept the direction — but it needs a checker: reject (or
flag) any loop containing a buffer op whose `mem` is loop-external and not the
carried token. That check is cheap, mirrors #190, and turns the contract from
convention into invariant. **It is the missing sixth PR.**

**The `Nil` ghost token is weak evidence.** Any `Nil` satisfies #190's type
check — a frontend that accidentally feeds a *fresh* `Nil` constant as the
next carry value passes verification while the ordering silently vanishes.
Typed linear references (origins) fix this class by construction; until then
the presence-checker above is the mitigation.

**Per-variable edges are incomplete in nested loops, but the loop-level marker
holds for dcc.** `ThreadLoopMemory` threads each nesting level independently;
a variable touched *only* in an inner loop gets no cross-outer-iteration edge.
In practice every dcc variable is memory-backed, so any terminating outer loop
touches some buffer in its direct body/condition and its token carry stays
live — a parallelizer keying off "has live token carry" stays safe. Toy's
`_nested_for_carry` explicitly threads every level for exactly this reason.
Worth documenting as a known limitation of per-level threading.

**`lower_for` carries are a half-feature.** The loop result stays `Nil` and no
exit phis are materialized, so carries only work for effect tokens /
memory-backed state — a real SSA reduction can't exit the loop. The
multi-carry `unpack` branch is untested (the PR itself flags this). The old
`lower_for` silently *dropped* carries (the `NESTED_FOR` test's `[%i]` carry
was dead weight main accepted) — so #188+#190 strictly improve honesty here —
but I'd trim the machinery to the single-carry case until a real multi-carry
user exists.

**Smaller findings**

- #186's docs (both the `control_flow_to_goto.py` docstring and
  `docs/control-flow.md`) say "dcc's `CLvalueToMemory` threads the token", but
  the final implementation moved threading into the separate
  `ThreadLoopMemory` pass — stale wording from an earlier revision; fix on
  merge.
- #185's semantic analysis is correct: `chain` orders neither operand relative
  to the other; consumers depend on both. The rename encodes the roles, ASM is
  positional so snapshots are unaffected. The `ForOp` comment fix is right.
- #189 closes a real false-confidence hole (snapshot string comparison never
  parsed the IR; regenerated snapshots would bless garbage). Applied to only
  five call sites — a follow-up should fold validation into the snapshot
  fixture so it's automatic.
- #190's `types_equivalent` (wire-format equality) stand-in is acknowledged in
  the PR and structured for replacement; fine.
- The pre-existing `ruff` error on the merged tree is also on `main` — not
  introduced by the cluster.

## Verdict

**Merge the cluster, in its stated order, with small changes.** It fixes a red
`main`, replaces implicit ordering with explicit edges (the IR's own axiom),
and every verified claim in the PR descriptions checked out empirically —
including the predicted cross-PR conflict.

| PR | Verdict | Conditions |
|----|---------|------------|
| **#186** | Merge with changes | Fix `CLvalueToMemory` → `ThreadLoopMemory` doc references; reconcile `ChainOp` kwargs if #185 lands first. **This is the PR that turns `main` green — land it first.** |
| **#188** | Merge with changes | Trim (or test) the untested multi-carry `unpack` branch; keep single-carry. Land after #186 (overlaps `control_flow_to_goto.py`; merges clean in this order). |
| **#190** | Merge as-is | After #188 (stacked). |
| **#185** | Merge as-is | Whichever of #185/#186 lands second updates the one `ChainOp(lhs=…)` line — verified one-line, and git will NOT flag it. |
| **#189** | Merge as-is | Independent; land anytime. Follow-up: fold into the snapshot fixture. |
| **#187** | Merge with a freshness pass | Roadmap content is sound; its PR-triage table predates this very cluster. |

**Follow-ups the cluster should generate** (none blocking):

1. A token-**presence** verifier for loops over buffer ops (the missing
   enforcement of the concurrency contract).
2. A stated invariant that passes moving buffer ops must preserve threading.
3. Mark toy's `Nil`-token threading as a placeholder pending origins, so the
   shim doesn't quietly become the design.

## Resolution (2026-07-08)

This branch integrates the five code PRs in the order above and lands the
review findings on top:

- `ChainOp` kwarg reconciliation and the `ThreadLoopMemory` doc-reference fix.
- Follow-up 1: `ThreadLoopMemory.verify_postconditions` rejects any in-loop
  buffer op reading a loop-external mem, with a negative test.
- The multi-carry `lower_for` path is now tested (LLVM-validated two-carry
  snapshot) and hardened (clear `TypeError` for undecomposable body results).
  Writing that test surfaced and fixed a codegen bug: aggregate `Constant`s
  over 16 bytes were referenced as host pointers where codegen types
  aggregates as `{ ... }` structs (invalid IR in loop-entry phis).
- Comment/docstring/naming cleanups across the cluster's code.

Suite after integration: 840 passed (main before: 824 passed / 10 failed).
