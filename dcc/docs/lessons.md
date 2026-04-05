# Lessons from working on the C frontend

A running collection of things that would have been useful to know before
digging into the C frontend. Append your own as you learn them.

## How the lowered IR actually looks

### Unreachable ops are silently dropped

The C frontend builds a function body by collecting ops from `_compound`
and setting `block.result = ops[-1]`. **Only values reachable from
`block.result` via `transitive_dependencies` end up in the block.** If
you yield an op that isn't chained to the result, it disappears. It
doesn't appear in `block.ops`, `block.values`, ASM formatting, or
`all_values(func)` — not anywhere.

This trips up structural assertions in tests. Example:

```c
if ((*p = 5)) return 1;
return 0;
```

The lowered IR contains only `return 0`. The `if`, the dereference, the
store — all orphaned, all dropped. Not a bug in your assertion; the
frontend genuinely isn't chaining statements with side effects to
anything that reaches the function's return.

**Workaround for tests**: write C where the thing you want to inspect
*is* the final expression, or threads into it. Void functions whose
last statement is the op under test work cleanly.

**Real fix (TODO)**: `_compound` should chain side-effecting statements
with `ChainOp` so they're reachable from the last expression.

### Named callees don't create use-def edges

`c.CallOp` stores the callee as `String`, not as a `Value`. So
`f() { g(); }` doesn't put `g` in `f`'s use-def graph. On its own, the
entry function returned from `lower()` can't reach any other function.

`dcc.parser.lowering._resolve_callee_captures` fixes this after lowering:
it scans every function body for `CallOp`s, maps each string name to
the matching `FunctionOp`, and adds it as a capture on the caller's
body. Only then is the whole program reachable from a single entry.

Forget this and JIT'd calls will fail to parse (`use of undefined value
'@g'`) because `emit_llvm_ir` walks from the entry and never finds `g`.

### Struct field access is stubbed out

`lower_pointer_member_access` in `c_to_memory.py` emits
`GepOp(base, index=0)` then `LoadOp` — that is, it ignores the field
name entirely and returns the base pointer as the field type.
`FieldAddressOp` does the same. This means struct-field *typing* works
but struct-field *offsets* are wrong for anything past the first field.

Emitted IR looks valid and often parses/verifies, so the sqlite3
`verified` count is a substantial overestimate of semantic correctness.
Treat "verified" as "LLVM accepted the shape" not "will produce the
right runtime value".

## Debugging sqlite3 failures

### Bucket the error messages

`test_codegen_sqlite3` normalizes llvmlite parse errors (strips `%names`,
`@names`, `iN`, numeric IDs) and buckets them. One `{error string ->
count}` dict tells you the top 5 things to fix and roughly how much
lowering they'll unblock. Much better than staring at one failing
function.

```python
norm = re.sub(r"[%@][\w.]+", "%X", msg)
norm = re.sub(r"\bi\d+\b", "iN", norm)
norm = re.sub(r"\b\d+\b", "N", norm)
```

### Extract a failing IR sample with an env var

The sqlite3 codegen test supports `DUMP_IR_FOR="<substring>"`. Set it
to a bucket name and it prints the first failing function's IR and
breaks out of the loop:

```bash
DUMP_IR_FOR="multiple definition" pytest dcc/test/test_c_frontend.py::TestSqlite3::test_codegen_sqlite3 -s -m slow
```

Good for turning "850 undefined-value errors" into "here's one
concrete IR that fails, figure out the pattern, write a 1-line C
repro".

### Ratchet assertions

The sqlite3 test has thresholds (`assert lowered >= 2560`,
`assert verified >= 1180`). Nudge them up every time you fix a bucket.
They catch regressions but don't block experimentation.

## Test infrastructure

### dgen is installed editable — worktrees won't isolate

`dgen` is installed as an editable package pointing at `/home/user/dgen`.
A `git worktree` at `/tmp/dgen_verify` will NOT pick up its own
`dgen/` source; it still loads from the editable install.

**Fix**: set `PYTHONPATH=/tmp/dgen_verify` when running pytest from the
worktree. Also clear `__pycache__` directories — stale bytecode will
lie to you.

This burned a lot of time during the "verify each fix by reverting"
pass. Tests appeared to pass against reverted source because they
were actually running against the unreverted install.

### Ir snapshots are graph-equivalence, not text

`ir_snapshot` uses `IRSnapshotExtension`, which parses both the actual
and snapshot text, then calls `graph_equivalent`. Semantic no-op
reorderings (op ordering, SSA renaming) don't force a snapshot update.
So the `.ir` text can drift without breaking tests.

This means:
- You can regenerate `.ir` files with `--snapshot-update` without
  worrying about cosmetic churn.
- Add a **structural assertion** alongside every snapshot. The
  snapshot catches cosmetic drift; the structural assertion catches
  semantic regressions. `_contains_op(m, FooOp)` and
  `_count_ops(m, FooOp) == 2` are cheap and precise.

### `_codegen_verifies` vs `emit(func)`

Per-function emission via `prepare_function(func, ctx)` + `emit(func)`
does NOT emit `declare` lines for external functions. If the C code
calls anything not defined inline (`__builtin_bswap16`, libc), LLVM
parse fails with "undefined value".

Use `emit_llvm_ir(value)` instead — it discovers `ExternOp`s and emits
the declarations. For `llvm.CallOp` / `function.CallOp` with *string*
callees, extern auto-discovery was removed in commit 7e6b689 and
hasn't been restored. Your test may still fail on string-named libc
calls; keep test cases self-contained (no external calls) until that's
fixed.

### `strict=True` on xfail

Always use `@pytest.mark.xfail(reason=..., strict=True)` for
outstanding bug repros. When the bug is fixed, the test unexpectedly
passes, pytest reports it as a failure, and you're forced to delete
the marker. Without strict, fixed bugs silently stay xfailed forever.

## Python / pytest gotchas

### f-string assertion messages eagerly dereference attributes

```python
assert stats.skipped_functions == 0, (
    f"lowering skipped functions: {stats.skip_reasons}"
)
```

Python only evaluates the message on assertion failure, but f-strings
still call `__format__` on every expression inside, which calls
`getattr`. If `stats.skip_reasons` doesn't exist (e.g. you're testing
against an older version of `LoweringStats`), you get `AttributeError`
instead of the real assertion failure, and the test output misleads.

Prefer a lazy form or check attribute existence defensively in test
helpers.

### pycache will lie to you

After reverting source files in a worktree, always clear
`__pycache__` directories. `.pyc` files are timestamped but module
discovery caches aggressively.

```bash
find . -name "__pycache__" -type d -not -path "./.venv/*" -exec rm -rf {} +
```

## Useful patterns

### LoweringStats-based introspection

`stats.function_ops` (list of all lowered FunctionOps) and
`stats.skip_reasons` (`{error_string -> count}`) are both cheap to
compute and enormously useful. Add more fields here when you need
visibility into frontend behavior — don't reach for ad-hoc logging.

### Verify each fix against its baseline

For regression tests of fixes, use a worktree + `git apply --reverse`
of each fix commit to verify the test actually fails when the fix is
gone. This catches:

- Tests that pass for the wrong reason (caching, other fixes)
- Tests whose repro doesn't actually trigger the bug
- Cross-coupling between fixes

Caveat: see the editable-install gotcha above. Use `PYTHONPATH`.
