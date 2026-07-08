# dgen design review — idealized goals vs current state

*Compiled 2026-07-08 from the design docs (`docs/`), the core framework (`dgen/`),
the examples (`examples/`), `TODO.md`, the 10 open PRs, and a fresh full-suite test
run on `main` (`5df0acb`, 2026-06-20).*

---

## 1. The idealized design — two distinct visions

dgen carries **two layers of ambition**, and they are not the same project.

### 1a. The original charter (CLAUDE.md, `docs/dialect-files.md`, `docs/specs/dialect-files.md`)

A **TableGen/MLIR replacement**:

- Target-language-independent dialect definitions (no `extraClassDefinition` C++ escape hatches)
- A **formal grammar** for the dialect-definition language (unlike TableGen's implementation-defined format)
- Default memory representations optimized for JIT: **wire format = memory format**, mmap/memcpy-friendly
- Not bound to MLIR's data model, but with an **MLIR generation backend**
- Ops with exactly one output; types parameterized on values; types as values

### 1b. The current north stars (`docs/roadmap.md`, PR #187 — unmerged)

Four objectives, dependency-ordered into horizons H0–H5:

1. **dcc → sqlite** (immediate flagship): compile real C at scale; demo *performance* and *C-with-dependent-types*
2. **Actor framework** with a real in-process runtime (mailboxes, scheduler, supervision) — the formal payoff of effects + linearity
3. **Self-hosting**: passes expressible as dialect-described IR (dual-mode with Python passes)
4. **High-level collections**: Scala-style trait hierarchy, immutable + mutable variants

These stand on a shared "spine": traits + subtyping, origins (linear memory),
cross-function effects, FatPointer/List, type staging.

### The gap between the two visions

The original charter has quietly become vestigial:

- `docs/Milestone: TableGen Replacement.md` is a **0-byte empty file**.
- The MLIR backend **does not exist anywhere** in the codebase, and `docs/ongoing-work.md` §5
  explicitly lists "MLIR backend / TableGen-replacement positioning — is it on this horizon
  at all, or deferred?" as an *unresolved* strategic question.
- The roadmap's horizons H0–H5 contain **no TableGen-replacement or MLIR-backend work at all**.

The de-facto project today is a **staged-compilation research compiler** (a value-centric IR
with dependent types, effects, and linearity, JITed via llvmlite) — the dialect-generation
framing survives mainly in the `.dgen` file format and the repo name.

---

## 2. Current state of `main` (verified 2026-07-08)

### Suite health

- **`main` is red**: 10 failed, 824 passed, 3 skipped, 9 xfailed, 1 xpassed (~7s).
  All 10 failures are dcc loop tests hitting
  `dgen/passes/control_flow_to_goto.py:312` ("body result must be a tuple of carried
  values; got Nil()") — the regression the #184 merge introduced and unmerged **PR #186 fixes**.
- No commits since 2026-06-20; the June-21 PR cluster (#185–#190) has sat for ~2.5 weeks.

### What genuinely works

- **Value-centric IR core** (`Value`/`Type`/`Op`/`Block`): closed blocks with explicit
  captures, enforced by the verifier; ASM round-trip heavily tested; `Module` deleted.
- **The staging engine is real and is the headline differentiator.** Both paths of the
  described two-path system are implemented (`dgen/passes/staging.py`):
  stage-0 subgraph extraction → isolated JIT → patch back as `ConstantOp`
  (`staging.py:133-145`), and runtime-callback thunks that re-JIT stage-2
  specializations against live values (`staging.py:164-192`, `codegen.py:1171-1247`),
  including cross-function thunk symbol registration.
- **Effects v1** (try/raise CPS-lowered to goto), generic divergence detection
  (`Diverge`, `Value.totality`), and a **linearity verifier** wired into every pass.
- **toy** is a mature end-to-end reference: source → AST → Toy IR → structured →
  goto → memory → LLVM → JIT, with 105 passing tests and 49 IR snapshots; it also
  serves as the fixture dialect for core tests.
- **dcc** is substantial (802-line lowering, 363-line type resolver) and JITs real C
  for arithmetic/locals/if-else — but all loops are broken on `main` (above), and
  greenfield bricks 8–10 (struct layout, implicit conversions, `c_to_llvm`) are
  commented-out stubs in `examples/dcc/cli.py:29-36`.
- **actor** is a sketch: 3 ops, one inlining pass, hand-written IR strings, both JIT
  tests xfail. **dependent_types** is `.dgen.asm` test data only.

### Known holes in the core (all confirmed in code)

- **No memory is ever freed**: `memory.deallocate` and `buffer_deallocate` lower to
  no-ops (`dgen/llvm/memory_to_llvm.py:57-79`); every heap allocation leaks.
- **`alloca`/`gep`/buffer codegen hardcode `double`** and 8-byte strides
  (`dgen/llvm/codegen.py:833,838,886,898`) — a toy/f64-era coupling.
- **Constraint verification is mostly off**: only `HasTraitConstraint` is checked;
  `ExpressionConstraint` and `HasTypeConstraint` are parsed and silently skipped
  (`dgen/ir/verification.py:267-274`, xfails in `test/test_trait.py`).
- **The linearity verifier is sound-but-permissive**: `_has_known_block_semantics`
  returns `False` unconditionally (`verification.py:359-372`).
- **Recursion is broken**: no `func.recursive`, so recursive calls violate the DAG
  property; recursive staging xfails (`test/test_peano.py:496`);
  `transitive_dependencies` is recursive and dcc needs `setrecursionlimit(50000)`.
- **`Executable` is documented as legacy** (`codegen.py:1029-1037`) with a known
  GC-lifetime bug (`TODO.md:54`); `staging._jit_evaluate` works around it by
  round-tripping through JSON.
- Placeholder no-op passes kept for pipeline-name stability: `RecordToMemory`,
  `ConstantFold`.

---

## 3. Ongoing workstreams

### W1 — Loop memory-effect threading (the live workstream; June 21, all open)

The #184 merge (linear `State`/`Reference` skeleton) broke dcc's loops. The response
is a five-PR cluster establishing a **loop concurrency contract** (a carry that
threads a dataflow/effect token ⇒ sequential loop; no such carry ⇒ parallelizable):

- **#186** — `ThreadLoopMemory` pass for dcc loops; fixes the 10 red tests; states the contract.
- **#188** — threads an accumulator token through toy's `CountNonzero`; generalizes
  `lower_for` to real loop carries.
- **#190** — carry/result type-consistency preconditions in `ControlFlowToGoto` (stacked on #188).
- **#185** — renames `chain(lhs, rhs)` → `chain(result, effect)`; fixes the docs' wrong
  "establishes ordering" story.
- **#189** — validates emitted LLVM in snapshot tests (closes the "snapshots bless
  malformed IR" gap).

These have explicit inter-PR merge-order notes (#186 → #188 → #185; #189 independent)
and have been stalled since June 21. **Landing #186 is the single action that turns
`main` green.**

### W2 — Memory-model migration (mem-tokens → Reference → origins)

Three-state migration, currently stuck in the middle: `main` merged #184's linear
`Reference`/`State` skeleton, but **every real consumer (dcc, toy, ndbuffer, record)
was parked on the renamed legacy mem-token type `memory.Buffer`**, and mem tokens were
not removed. The remainder — alias forest + sub-origins, destructor obligations,
ownership transfer, per-op linearity contracts — is `docs/effects.md` steps 3–7,
essentially unbuilt (roadmap H2.2). The roadmap deliberately **decouples dcc from
origins**: C keeps `Buffer` permanently; only the safe dialects migrate.

### W3 — The dcc chain (stale since April)

**#165** (cross-function calls) → **#169** (early `return` via `Never`-typed
terminators, with the known every-branch-diverges merge-elision gap). Both predate
#184 and need rebasing. This is roadmap H0/H1 critical path toward sqlite.

### W4 — The call/recursion design fork (blocking, undecided)

**#139** makes `call`'s callee an operand (Option A); `docs/block-scoping.md` §3.6
recommends **Option C** (block parameters). Unresolved since April. It gates
`func.recursive`, which gates recursion in *both* frontends and self-hosting passes.
The roadmap calls it "overdue and cheap to decide."

### W5 — Documentation reconciliation (itself unmerged)

**#187** adds `docs/roadmap.md` + `docs/ongoing-work.md` (the project's own
ideal-vs-actual analysis) and loosens the CLAUDE.md pytest-only rule. Until it merges,
the strategic direction exists only in a PR. #102 (toy autodiff) is off the critical
path and rotting.

---

## 4. Inconsistencies: idealized goal vs design docs vs code

### 4a. Charter vs code

| Stated principle | Reality |
|---|---|
| "Formal grammar specification (not implementation-defined like TableGen)" | A complete EBNF exists **on paper** (`docs/specs/dialect-files.md`), but `dgen/spec/parser.py` is a hand-rolled line/indentation recursive-descent parser, not generated from or validated against the grammar. The format is, today, exactly as implementation-defined as TableGen. |
| "Wire format = memory format, mmap/memcpy-friendly" | Holds only for flat scalar/record types. Pointer-bearing layouts (`Pointer`, `Span`, `String`, `Some`, `TypeValue`) serialize the *dereferenced* data, and codegen bakes **live host-process addresses** into emitted IR via `inttoptr` (`codegen.py:677`, `ffi.py:49`), keeping Python `bytearray`s alive through `Memory.origins`. Emitted IR is process-bound, not mmap-able. |
| "Target language independent" | `.dgen` files declare only op/type **signatures**, parameters, constraints, and block structure. All semantics — parsers, shape inference, lowerings, codegen — are Python `Pass` subclasses. Language independence is realized for declarations only. The LLVM layer additionally hardcodes f64 memory cells. |
| "Provides an MLIR generation backend" | Does not exist; no code, no plan entry, milestone doc empty. |
| "Ops always have exactly one output" | Honored (multi-results modeled as `Tuple` aggregates — e.g. `memory.load → Tuple<T, Reference<T>>`). |
| Closed blocks + captures | Honored and verifier-enforced. |

### 4b. CLAUDE.md vs reality

- Claims "110 tests, runs in ~1s" — the suite is **834+ tests in ~7s**.
- References `dgen/spec/importer.py`, which doesn't exist (the import hook is `dgen/imports.py`).
- States an absolute "never write throwaway scripts / pytest only" rule that #187 loosens —
  governance is forked across a PR boundary.
- Prescribes jj for VCS; fine locally, but remote/CI sessions run plain git.

### 4c. Doc vs doc (the docs stratify into three eras)

- **Oldest era** (`plan.md`, `dialect-files.md`, `staging.md`, `staged-computation.md`,
  parts of `block-scoping.md`): Module-based pipeline, `affine` dialect naming,
  `infer`/`lower` callbacks, symbols.
- **Middle era** (`passes.md`, `control-flow.md`, `pass-management.md`, `codegen.md`,
  `effects.md`, `linear_types.md`): goto dialect, `Compiler`, captures/`%self`.
- **Newest era** (`docs/plans/*`, `docs/specs/*`, PR #187's roadmap): Value-as-primitive,
  explicit "these older docs are stale" lists.

Concrete conflicts:

1. **Symbols vs `%self` parameters** — `passes.md` proposes an `unstructured_cf` dialect
   with `symbol`/`forward_declare`/`link`; `block-scoping.md` §1.4/§5.1 explicitly rejects
   symbol tables and `asm.md` says there is no symbol/label syntax. The `%self` design won;
   `passes.md` was never updated. Biggest doc-vs-doc conflict.
2. **Module vs Value** — `pass-management.md` says its redesign is "complete" yet still
   shows `Module`-based signatures; `value-compilation.md` declares Module deleted and
   lists five docs as needing updates that haven't happened.
3. **`infer`/`lower` callbacks** — `staged-computation.md` documents them as current;
   `pass-management.md` says they were eliminated.
4. **String/FatPointer status** — `memory-and-layout.md` presents `String → FatPointer`
   as the layout; `plan.md` Phase 2 and `staging.md` §2.1 say it's future work / not implemented.
5. **`plan.md`'s "Not planned" list** includes `.dgen` dialect files, SSA function calls,
   and block-syntax functions — all of which later docs made core. `plan.md` is superseded
   but still in the tree unmarked.
6. **Two dialect-file docs** with different constraint syntax (`require T ~= Pattern` vs
   `requires ... has trait/has type`); `docs/specs/dialect-files.md` is the authoritative one.
7. **Terminology drift**: `Diverge` vs `Diverges`; `toy_to_affine`/`affine` (old docs) vs
   `ToyToStructured`/`ndbuffer` (code); layout vocab `FatPointer` vs `Span`/`String`
   (`TODO.md:38` confirms `Span` naming is a known large refactor).

### 4d. Doc vs code

1. **`%self`/`%exit` convention** — `control-flow.md` describes the clean convention;
   `docs/plans/goto-self-convention.md` documents that codegen overloads `%self` for the
   if-merge case ("The doc was never updated. The convention diverged") with a 5-step fix
   plan whose codegen steps appear not yet fully landed.
2. **`value-compilation.md` says "Implemented"** while leaving the Executable model and
   mutual recursion open, and while `staging-cleanup.md` still catalogs Module-shaped
   staging code (`_jit_evaluate`, mini-module wrappers) awaiting deletion.
3. **Constraint language** — the spec grammar defines expression/type constraints that the
   verifier silently ignores.
4. **The strategic docs themselves live in an unmerged PR** (#187) whose snapshot
   ("9 open PRs," "#184 is the freshest PR") predates the June-21 cluster — the
   project's self-description is one migration behind the code, which is one migration
   behind the design.
5. **`examples/dcc/docs/greenfield-plan.md` describes a `dcc2/` clean-room rewrite that
   does not exist** — the plan was instead partially executed in place in `dcc/`.

### 4e. Meta-pattern

The same lifecycle repeats at every level: **design doc → partial implementation →
consumers parked on a legacy shim → doc describing the ideal as if done.** Examples:
origins (ideal) → #184 Reference skeleton (landed, unused) → `memory.Buffer` (shim,
carrying all real traffic); `%exit` convention (ideal) → `%self` overload (shim);
generic `constant` (landed) vs `Span` field annotations (stale shim); formal grammar
(spec) vs hand-rolled parser (shim). The project is disciplined about *writing down*
the ideal and the shim, but the backlog of "finish the migration" items is growing
faster than it is retiring — 10 open PRs, no merges June 20 → July 8, and `main` red.

---

## 5. Highest-leverage observations

1. **Merge #186** (or equivalent) — `main` has been red for ~2.5 weeks; everything else
   stacks behind a green baseline. Then land the rest of the June-21 cluster in its
   stated order (#186 → #188 → #190, #185, #189) and **merge #187** so the roadmap is
   actually in-tree.
2. **Decide the call/recursion fork** (#139 Option A vs block-scoping Option C). It's the
   cheapest undecided question with the widest blast radius (recursion in both frontends,
   self-hosting).
3. **Reconcile or retire the superseded docs** (`plan.md`, `passes.md` symbols section,
   `staged-computation.md` callbacks, CLAUDE.md's stale claims). The docs' three-era
   stratification is now actively misleading to both humans and agents working in the repo.
4. **Make the charter honest**: either schedule the TableGen/MLIR-backend milestone or
   rewrite the project framing around what dgen actually is now — a staged-compilation
   compiler framework. The empty milestone file and roadmap open-question #5 show this
   decision is being deferred by default.
5. The **wire-format and no-free gaps** are fine for a JIT demo but are silent
   contradictions of stated principles; worth either fixing or re-scoping the principle
   (e.g. "wire = memory for POD types; pointer-bearing types are process-local").
