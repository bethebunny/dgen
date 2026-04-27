# Diagnosis: `%self` and `%exit` conventions in `goto.region`

Read-only analysis — no code changes proposed yet. Written off `main` so
the user can sign off on direction before any refactor.

## What the design doc says

`docs/control-flow.md`:

> `%self` parameter enables back-edges (breaks use-def cycles)
> `%exit` parameter: codegen emits a fall-through label after the header
> block

Concrete loop example from the doc:

```
%header = goto.label([0]) body<%self: Label, %exit: Label>(%iv: Index):
    %cmp = algebra.less_than(%iv, %limit)
    %body = goto.label([]) body(%jv: Index) captures(%self):
        %next = algebra.add(%jv, 1)
        goto.branch<%self>([%next])     ← back-edge: re-enter %header
    goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])
```

`branch<%self>` re-enters the region; `branch<%exit>` leaves it.

The doc also says, line 113-115:

> `control_flow.if` — NOT lowered to goto. Codegen expands it inline
> during linearization into `cond_br → then → else → merge(phi)`. This
> avoids the fundamental difficulty of representing merge labels and
> phi values in a label-as-expression model.

So per the doc, IfOp shouldn't be in the `goto.region` shape at all.
That doesn't match what the code does today (see below).

## What the code actually does

### `emit_region_op` (`dgen/llvm/codegen.py:414-452`)

Three modes based on the body block:

| Mode                       | `initial_args` | `body.args` | What `%region.name` emits as              | What `%self` resolves to |
|----------------------------|----------------|-------------|--------------------------------------------|---------------------------|
| Loop (with init values)    | yes            | yes         | single block, phi at entry                 | the entry/body block      |
| If-merge (no init values)  | no             | yes         | two blocks: `{name}_entry` (body), `{name}` (phi merge) | the **phi merge block** |
| Plain region               | no             | no          | single block, no phi                       | the body block            |

`value_reference(self_param)` (line 564-570) returns `%{op.name}` — the
region's name — in every case. The label that name resolves to at the
LLVM level changes based on whether the region has merge args.

That is the root of the confusion: **`%self` is overloaded.**

- For loops and plain regions: `%self` ≡ the body block. `branch<%self>`
  is a back-edge. Matches the design doc.
- For if-merge regions: `%self` ≡ the phi merge block, *not* the body
  block. `branch<%self>` is "exit with this value via the phi." Doesn't
  match the design doc.

The "if-merge mode" was added so `IfOp` (and the recently-added
`TryOp`) could lower to a `goto.region` and have a phi at the merge. It
shoehorns merge-with-value into the same `%self` parameter the loops
use for back-edges.

### `emit_label_op` (`dgen/llvm/codegen.py:478-490`)

```python
exit_name = f"{op.name}_exit"
yield f"  br label %{exit_name}"     # skip the label by default
yield f"{op.name}:"                   # label entry (jump target)
yield from _emit_phi_nodes(op)
yield from emit_linearized(op.body)   # label body
yield f"{exit_name}:"                 # exit fall-through
```

A label emits as `{name}_exit:` (skip-redirect entry, fall-through),
then `{name}:` (the actual label). This is what generates the
`X_exit → X` redirect chains we see in compiled output. The redirect
is needed so a label embedded in a parent block's emission doesn't
accidentally execute on fall-through.

## Audit of consumers

| Consumer                                                          | Uses `%self` for           | Aligns with doc? |
|-------------------------------------------------------------------|----------------------------|------------------|
| `lower_for` (`control_flow_to_goto.py:198`)                       | back-edge (correct)        | ✓                |
| `lower_while` (`control_flow_to_goto.py:252`)                     | back-edge (correct)        | ✓                |
| `_resolve_jump_markers` continue (`control_flow_to_goto.py:99`)   | back-edge (correct)        | ✓                |
| `_resolve_jump_markers` break (`control_flow_to_goto.py:94`)      | uses `%exit` not `%self`   | ✓                |
| `lower_if` (`control_flow_to_goto.py:167`)                        | "exit with merge value"    | ✗                |
| `RaiseCatchToGoto.lower_try` (effects PR)                         | "exit with merge value"    | ✗                |

The mismatch is concentrated at the if-merge use case (and its copy in
the effects PR). Loops are clean. Break/continue are clean.

## Concrete LLVM artefact

For the trivial try-with-raise on the effects PR:

```
entry:
  br label %try0_entry
try0_entry:                         ← region's body (initial dispatch)
  br label %except0_exit            ← rewritten raise's branch to %except0
except0:                            ← except label
  %err = phi i64 [ 42, %except0_exit ]
  %_2 = add i64 %err, 1
  br label %try0                    ← except's branch<%self> ≡ branch to phi merge
except0_exit:                       ← skip-redirect for the except label
  br label %except0                 ← redirect lands at except0
try0:                               ← phi merge block (this is what %self points at)
  %try_result0 = phi i64 [ %_2, %except0 ]
  br label %try_exit0
try_exit0:
  ret i64 %try_result0
```

Two unwanted artefacts:

1. **`%self` lands at the phi merge** — readers of the IR expect a
   back-edge per the design doc. They have to know about the if-merge
   special case in `emit_region_op` to interpret it correctly. We fell
   into this exact misreading mid-review.

2. **`X_exit → X` redirect chain.** `try0_entry → except0_exit →
   except0` is two trivial branches for one logical edge. Same redirect
   appears in every label use in the compiled output. The pattern in
   `emit_label_op` is "always skip the label by default, then place
   it" — but a branch op that targets the label could just point at it
   directly without the redirect.

## Why this got here

Looking at the timeline:

- The label/region model was designed for loops
  (`docs/control-flow.md`'s example is a loop).
- The doc said IfOp wouldn't lower to goto; codegen would expand it
  inline.
- Later, someone added `lower_if` to lower `IfOp` → `goto.region` with
  merge args. To make the merge work, they added the
  `has_merge_args` mode in `emit_region_op` that reinterprets `%self`
  as the phi block.
- The doc was never updated. The convention diverged.

The effects PR copied `lower_if`'s pattern, so it inherited the same
issue.

## Two ways forward

### Option A — Codegen and consumers align with the doc

- `goto.region` no longer has the if-merge mode. `%self` is always
  back-edge (or the body block in plain regions).
- Introduce a clean way to express "exit with value via phi at merge."
  Two sub-options:
  - A1: a new `%merge: Label` parameter alongside `%self` and `%exit`,
    so `branch<%merge>([value])` is the exit-with-value primitive.
    Codegen emits the phi at the `%merge` label.
  - A2: keep two parameters; the `%exit` parameter carries the phi
    when the region has merge args. `branch<%exit>([value])` is
    exit-with-value. Code matches the doc's "exit fall-through" bullet
    more naturally — `%exit` is where you go to leave with a value.
- Update `lower_if` (and the effects PR's `lower_try`) to branch to
  the new label.

Diff size: medium. Touches `codegen.py` (region/label emission),
`control_flow_to_goto.py`, snapshot files. The IR text shape changes
(`<%self, %exit>` stays, but the branch targets change).

### Option B — Codegen keeps the current mode; doc/IR clarify it

- Keep the `has_merge_args` mode in `emit_region_op`.
- Rename the parameter when merge args are present so the IR text
  reflects what it actually is. e.g.
  `body<%self: Label, %merge: Label, %exit: Label>(%result: type)`.
  `%self` stays for back-edges. `%merge` is the phi block (the new
  name for what's currently `%self`-as-merge). `%exit` is the
  fall-through.
- Update the doc to spell out the merge mode.
- Update `lower_if` and `lower_try` to use the new name.

Diff size: smaller for codegen; same touch surface in consumers.
Renames everywhere. Cements an unintuitive split (regions have
different parameter layouts based on whether they merge).

### Recommendation

**Option A2** — make `%exit` carry the merge phi when block args are
present. The doc already calls `%exit` "a fall-through label after the
header block"; for a region without merge args, that's just the
fall-through. For a region *with* merge args, `%exit` is the natural
place for the phi (you're leaving the region carrying a value). One
parameter does the right job in both cases. `%self` stays for
back-edges only.

Concretely the artefact above becomes:

```
entry:
  br label %try0
try0:                               ← body block (was %try0_entry)
  br label %except0                ← direct branch (no redirect)
except0:
  %err = phi i64 [ 42, %try0 ]
  %_2 = add i64 %err, 1
  br label %try_exit0              ← branch<%exit> with value
try_exit0:
  %try_result0 = phi i64 [ %_2, %except0 ]
  ret i64 %try_result0
```

Five blocks instead of seven; phi at the natural exit point; `%self`
unused (and would've been a back-edge if the region had a loop body).

While we're here, kill the `X_exit → X` redirect in `emit_label_op` —
when a label has at least one explicit branch targeting it, the
redirect adds nothing. Keep it only if the label is positioned where
fall-through could otherwise reach it (rare in practice; would need
audit).

## Out of scope for this branch

This branch only contains the diagnosis. Any code changes are gated on
the user picking a direction.

## Files involved (when implementation begins)

- `dgen/llvm/codegen.py::emit_region_op`, `emit_label_op`,
  `_emit_phi_nodes`, `_emit_exit_phi_nodes`, `value_reference`
- `dgen/passes/control_flow_to_goto.py::lower_if`,
  `_make_branch_label`
- `dgen/passes/raise_catch_to_goto.py::lower_try` (after the effects
  PR resumes)
- `docs/control-flow.md`, `docs/codegen.md`
- snapshot files under `test/__snapshots__/test_if_explicit_capture/`,
  `test/__snapshots__/test_break_continue/`,
  `test/__snapshots__/test_raise_catch/` (regenerate)

## Verification at end of fix

1. Loop snapshots unchanged (loops weren't affected).
2. If/try snapshots: shorter, no `_entry`/`_exit` redirect chains,
   `%self` no longer used in if/try lowerings.
3. Hand-inspect emitted LLVM for one if and one while; confirm fewer
   blocks and no spurious branches.
4. Effects PR rebases cleanly on the upstream change; resumed PR's
   tests pass without modification.
