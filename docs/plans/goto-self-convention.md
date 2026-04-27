# Plan: Align `goto` codegen with the `%self` / `%exit` design

Diagnosis of the current state, then an implementation plan to fix it
(Option A2 below). Off `main`, isolated from the parked effects PR
(#170) so the convention fix lands first.

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

## Direction: Option A2 — `%exit` carries the merge phi

`%exit` already means "a fall-through label after the region" per the
doc. For a region without block args, that's just the fall-through. For
a region *with* block args, `%exit` is the natural place for the merge
phi: you're leaving the region carrying a value. One parameter does the
right job in both cases. `%self` stays for back-edges only.

After the change, the trivial try-with-raise emits as:

```
entry:
  br label %try0
try0:                               ← body block (was %try0_entry)
  br label %except0                 ← direct branch (no _exit redirect)
except0:
  %err = phi i64 [ 42, %try0 ]
  %_2 = add i64 %err, 1
  br label %try_exit0               ← branch<%exit> with value
try_exit0:
  %try_result0 = phi i64 [ %_2, %except0 ]
  ret i64 %try_result0
```

Five blocks instead of seven. Phi at the natural exit point. `%self`
unused (the region has no loop body, so there's no back-edge target).

The label-redirect cleanup (`X_exit:` chains in `emit_label_op`) is
included in this plan as a separate, smaller change — without it,
`%try0 → %except0_exit → %except0` would still appear even after the
A2 region rewrite.

## Implementation plan

Five sequential changes. Each leaves the suite green so the work can
land incrementally.

### Step 1 — codegen: `_resolve_target` and `prepare_function`

Currently every `BlockParameter` of a region/label maps to its owner
in `param_to_owner`, so `_resolve_target` returns the owner for both
`self` and `exit`. That's why a `branch<%exit>` records its
predecessor on the *region* instead of on `exit_param`, and the
phi-at-exit machinery in `_emit_exit_phi_nodes` never fires for
explicit `branch<%exit>` (it's only reachable via fall-through today).

Change so `self_param → owner` (back-edge resolves to the region's
body block) but `exit_param` is left out of `param_to_owner`. Then:

- `branch<%self>` resolves to the owning region — predecessors recorded
  on the region. Loops emit phi at the region's name as today.
- `branch<%exit>` resolves to `exit_param` itself — predecessors
  recorded on `exit_param`. Phi-at-exit emits the merge phi.

Files: `dgen/llvm/codegen.py::prepare_function` (the
`for param in op.body.parameters` loop) and `_resolve_target` (no
change to `_resolve_target` itself — drop exit_param from the mapping
upstream).

### Step 2 — codegen: `emit_region_op` block layout

Collapse the three modes to two, with `%exit` carrying the phi when
present:

| Region shape                      | Emits as                               |
|-----------------------------------|----------------------------------------|
| `has_initial_args`  (loops)       | `{name}:` body w/ phi at entry. `{exit_param.name}:` fall-through after. (Unchanged.) |
| Otherwise (if/try-merge or plain) | `{name}:` body. `{exit_param.name}:` w/ phi for body.args. |

Concretely, in the non-loop branch of `emit_region_op`:

```python
yield f"  br label %{op.name}"
yield f"{op.name}:"
yield from emit_linearized(op.body)
# body emits its own terminators (cond_br, branch<%exit>, etc.)
yield f"{exit_param.name}:"
yield from _emit_exit_phi_nodes(exit_param)  # was _emit_phi_nodes(op) at {op.name}:
```

`_emit_exit_phi_nodes` reads `ctx.predecessors[exit_param]`, which step
1 just made non-empty for explicit `branch<%exit>(args)`. The phi name
matches body.args[0].name (`merge_result.name = "if_result0"` etc.) so
`value_reference(region) → body.args[0]` keeps working unchanged.

Drop the `{name}_entry:` block — it's no longer needed; the body lives
at `{name}:`. Update `prepare_function` to record the body's LLVM block
name as `op.name` (not `{op.name}_entry`) for the non-loop merge case.

`_emit_exit_phi_nodes` already exists; it generates phi names as
`{param.name}_phi{idx}`, which doesn't match what the consumer expects.
Replace its body to emit phis named per `body.args` (mirrors
`_emit_phi_nodes`), reading from `ctx.predecessors[exit_param]` instead
of `ctx.predecessors[region]`.

After this step, **regions still work correctly** but `lower_if` is
still branching to `%self`. Branches to `%self` would now record on the
region but we no longer emit a phi there. So this step deliberately
doesn't ship in isolation — combine with step 3 in one commit, or land
both together.

### Step 3 — `lower_if`: branch to `%exit`, not `%self`

In `dgen/passes/control_flow_to_goto.py::_make_branch_label` and
`lower_if`:

```python
# Before:
result = goto.BranchOp(target=merge_self, arguments=pack([body.result]))
captures = [merge_self, *body.captures]

# After:
result = goto.BranchOp(target=merge_exit, arguments=pack([body.result]))
captures = [merge_exit, *body.captures]
```

`merge_self` is no longer captured by then/else labels (it would only
matter for back-edges). The region's body still has `[merge_self,
merge_exit]` parameters since the IR shape is unchanged.

Loops are untouched: `lower_for` and `lower_while` already use
`header_self` for the back-edge correctly.

### Step 4 — `emit_label_op` redirect cleanup

Today every label use emits a skip-redirect even when nothing falls
through to it:

```
br label %{name}_exit
{name}:
  ...body...
{name}_exit:
```

Replace with: only emit the skip-redirect when the previous emission
*didn't* terminate. Track a `terminated` flag through `emit_linearized`
(it already does — the `terminated = yield from emit_linearized(...)`
return value). When `emit_label_op` runs at a position where the
previous op already terminated, drop the leading `br label %{name}_exit`
and just emit `{name}:` directly. The trailing `{name}_exit:` stays —
it's where post-label ops continue. (If we want to cleanup that too,
audit later; not required for the immediate goal.)

Most labels in real lowerings *are* preceded by a terminator (the prior
op is a branch or another label whose exit-edge runs out at the
position). Keeping the redirect logic but firing it only when actually
needed eliminates the redundant branches without breaking the cases
that genuinely needed them.

### Step 5 — docs and IR-shape refresh

- `docs/control-flow.md`: clarify that `%exit` carries the merge phi
  when the region has block args. Add a small example beside the loop
  example showing the if-merge shape:
  ```
  %if = goto.region([]) body<%self: Label, %exit: Label>(%result: T):
      %then_lbl = goto.label([]) body() captures(%exit):
          ...
          goto.branch<%exit>([then_value])
      ...
  ```
- `dgen/passes/control_flow_to_goto.py` docstring (the long block
  comment near the top): update the `IfOp` lowering schematic to use
  `%exit` instead of `%self`.
- `docs/codegen.md`: refresh `emit_region_op` description.

## Files

| File                                             | Change          |
|--------------------------------------------------|-----------------|
| `dgen/llvm/codegen.py`                           | steps 1, 2, 4   |
| `dgen/passes/control_flow_to_goto.py`            | step 3          |
| `docs/control-flow.md`                           | step 5          |
| `docs/codegen.md`                                | step 5          |
| `test/__snapshots__/test_if_explicit_capture/*`  | regenerate      |
| `test/__snapshots__/test_break_continue/*`       | regenerate (label cleanup may simplify these) |

`dgen/passes/raise_catch_to_goto.py` (effects PR) follows the same
pattern as `lower_if` and only needs the same one-line change. That
happens after the effects PR is rebased onto this work — not in this
branch.

## Verification

After step 3 (the smallest landable unit):
1. `pytest . -q --ignore=examples/dcc` clean.
2. Snapshot diffs for `test_if_explicit_capture/*` show the phi moves
   from `%if0` to `%if_exit0`. Block count drops by one (no
   `%if0_entry` block).
3. `test_break_continue/*` snapshots either unchanged or strictly
   simpler.
4. Hand-inspect emitted LLVM for `test_if_value_lowering`; confirm
   `branch<%self>` doesn't appear and the phi sits at the exit block.

After step 4 (label redirect cleanup):
1. `pytest . -q` still clean.
2. Snapshot diffs show fewer `%X_exit:` redirect chains.
3. Hand-inspect emitted LLVM for one loop and one if; confirm no
   spurious `br label %X_exit; X_exit: br label %X` patterns.

After all five steps:
1. The example artefact in this doc (5 LLVM blocks for try-with-raise)
   matches what's actually emitted when the effects PR is rebased.
2. Effects PR diff stays small under rebase (its `lower_try` swaps
   `merge_self` → `merge_exit` in two places; everything else carries
   over unchanged).
