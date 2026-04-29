## Staging
- Batch multiple subgraphs in the same staging pass rather than serializing them

## Constraint verification
- Implement expression constraint evaluation in `verify_constraints`. Requires an evaluator that can resolve operand/parameter references, attribute access on resolved types (`input.shape.rank`), comparisons, and arithmetic. Currently only `HasTraitConstraint` is verified; `ExpressionConstraint` and `HasTypeConstraint` are stored but silently skipped. See xfail tests in `test/test_trait.py`.
- Implement `HasTypeConstraint` verification (`requires X has type Tensor`): resolve subject to its type, look up the pattern name in the dialect type registry, check `isinstance`.

## General pass infrastructure
- Passes should guarantee they can lower all ops in their input dialect — add validation that no un-lowered ops survive a pass
- Canonicalization

## Block / scope invariants
- Implement `func.recursive` op for recursive functions (see `docs/block-scoping.md` §3.1). Currently recursive functions like `%natural` calling itself via `call<%natural>` violate the DAG property — `block.ops` follows the callee parameter edge back into the function, creating a cycle. `func.recursive` breaks the cycle by providing `%self` as a block argument.

## Actor framework
- Add a loop fusion optimization pass — currently `ActorToAffine` emits separate loops per actor; a general fusion pass would subsume the fused-pipeline special case

## Test infrastructure
- Run verifier on IR test inputs (parsed from ASM) — currently roundtrip tests don't verify captures/scoping invariants on the parsed IR
- Fuzz testing on dependency ordering: tests should verify correctness under any valid topological sort of the use-def graph, not just the one produced by the current `transitive_dependencies` walk. A random topo-sort iterator would catch ordering bugs where ops are accidentally independent (missing use-def edge) but happen to be emitted in the right order by the default DFS.

## Parser / type values
- Parser stores Type objects (e.g. `Index()`) as constant values for TypeType fields. It should store their JSON dict form (`{"tag": "builtin.Index"}`) so `TypeValue.from_json` doesn't need special-casing.

## Experiments / scope creep
- Update `toy.Tensor` to use `Pointer<Array<...>>` — removes the need for the runtime `llvm.load` to extract the data pointer (shape is compile-time, so no indirection is needed)
- DTypes

## Parser / formatting improvements
- Support `%_ = %ref` syntax (alias an SSA ref without an op). Currently blocks whose result is a block argument need `chain(%arg, ())` as a workaround.
- Massively simplify / clean the asm parser and formatter code
- Add parser failure tests
- Restore Python-style `#` line comments in ASM. The parser should skip from `#` to end-of-line in `_skip_ws`/`_skip_all`. Tried writing a comment in `examples/dependent_types/existential_any.dgen.asm` and it choked on `#`.

## Cleanup
- Remove spurious utf8 decoding stuff from Memory/Value
- Move `type_asm` to `Type.asm`, `op_asm` to `Op.asm`
- Reconcile unparameterized `Span` operand declarations across `.dgen` files. Sites like `record.PackOp.values: Span`, `goto.{Region,Label,Branch,ConditionalBranch}Op.{initial_,true_,false_,}arguments: Span`, `function.CallOp.arguments: Span`, `llvm.CallOp.args: Span`, `control_flow.{For,While,If}Op.*_arguments: Span` are not actually `Span`s (heap-allocated runtime sequences) — they're fixed-N positional value bundles. `pack()` now produces `Array<T, n>` (homogeneous) or `Tuple<types>` (heterogeneous) for these slots; the field annotations are stale. The parser's `_pack_list` currently keys off `field_type is Span` to coerce bare-list literals (e.g. `[42, 43]`) using a homogeneity heuristic on the first element. To fix: introduce a wildcard "value bundle" type (or extend the parser to handle the wildcard via a different signal) and update every operand declaration listed above. Also fix the parameterized `Span<Type>` declarations on `Tuple<types>` and `Function<arguments>`: they're compile-time fixed-N homogeneous lists of Type values, so honestly `Array<Type, n>` — but switching them adds an `n: Index` parameter to both type signatures and breaks every callsite. Doing both fixes is one larger refactor that lands together with a "value-bundle wildcard" decision.

## Block / value infrastructure
- Have values track their uses for forward iteration and fast `replace_uses`
- Make `Block` a `Value`

## Existentials
- Add a runtime `pack_existential` (or extend `algebra.cast`) op that boxes a runtime value into `Some<X>` / `Any`. Currently `Some`/`Any` only have constant construction; the cast example can't yet do "math, then return `Some<IntegralType>`" because there's no op that takes a runtime SignedInteger and produces a Some at runtime.
- Symmetric `unpack_existential` op for compile-time projection — a stage-boundary that resolves the witness type and yields a typed value.
- Compound constants (`Any`, `Some<X>`, future record-shaped existentials) currently store their rich payload as a Python dict produced by `Layout.Record.to_native_value`, and `Constant.required_dialects` peeks through dicts/lists to find embedded `Value`s. The structurally clean version is to make compound constants' rich form a tree of real `Value`s (e.g. a synthetic `pack_existential(witness=<Type>, value=<Value>)` op) so `transitive_dependencies` walks them naturally and the dict-peeking helper goes away. Probably falls out of the runtime `pack_existential` work above.

## Recursive types
- The layout system can't currently express recursive type definitions: `type Foo: data: Bar` / `type Bar: data: Foo` would loop forever in `_make_layout` since each side eagerly expands the other's layout. To support this we'd need either auto-boxing on cycle detection (insert a `Pointer` indirection when a layout would recurse into a type already on the stack) or an explicit "boxed" annotation in `.dgen`. The current `Some<bound>` / `Any` definition sidesteps this by using `Pointer<Nil>` as the value field — a fixed-size opaque slot — rather than a true mutually-recursive `value: Any`.

## Codegen
- `SignedInteger`/`UnsignedInteger` layout vs bit-width mismatch: the `.dgen` definition uses `data: Index` (always 64-bit) regardless of `bits`. `llvm_type` currently works around this by using `max(declared_bits, layout_bits)`, but the proper fix is parameterizing the layout on `bits` so a 32-bit integer actually has a 32-bit layout.
- `Executable.run()` lifetime bug: when raw Python values are passed as args, `run()` creates temporary `Memory` objects that can be GC'd before the result is read. For non-register-passable types (e.g. `TypeType`), the JIT returns a pointer into the input Memory's buffer — if that Memory is collected, the result reads garbage. Fix: `run()` should attach input memories to the result's `host_refs`. Workaround in `staging._jit_evaluate` creates memories outside the call.
- Replace `ControlFlowToGoto._make_branch_label`'s `isinstance(body.result.type, Never)` proxy with the proper "is this op a terminator" check. `Value.totality is PARTIAL` plus a `Never` result-type test covers the cases the proxy catches today (`RaiseOp`-derived terminators); `NormalizeRegionTerminators` can use the same.

## Type system / effects
- Add type subtyping. ``Raise<E>`` should be a sub-effect of ``Diverge`` and ``RaiseHandler<E>`` should automatically satisfy ``Handler<Diverge>`` via that subtyping. Until then, ``RaiseHandler`` redundantly declares both ``Handler<Raise<E>>`` and ``Handler<Diverge>`` (see `dgen/dialects/error.dgen`) so the generic divergence-detection query (`Value.totality`, see `docs/control-flow.md`) finds it without a custom trait-resolution rule.
- Per-op linearity contract framework. ``verify_linearity`` currently uses the predicate ``_has_known_block_semantics(op)`` which returns ``False`` unconditionally — every block-holding op is treated as unknown, with captures going to ``MaybeAvailable``. A real contract framework (likely an opt-in trait or method on ``Op``) lets specific ops declare precise capture-consumption semantics; the verifier then transitions captures to ``Consumed`` (or stays at ``Available`` if the op borrows) for those ops. Until that lands, the verifier is sound but permissive on block-holding ops.
- Partial-op linearity rule. ``docs/linear_types.md`` is missing a rule about partial ops needing to consume / drain in-scope linear values before they execute (otherwise divergence leaks the unmet obligation). Right shape isn't obvious yet — pin down once a real test case forces the issue.
- Mark ``Origin`` (see `docs/effects.md`) as ``has trait Linear`` when origins land. The verifier (``verify_linearity`` in `dgen/ir/verification.py`) is already wired into every pass's pre/post hooks; it'll pick up the trait on the type and start enforcing single-consume with no further plumbing.
- Loop-carry linearity. ``docs/linear_types.md`` describes a "yield-as-consume" rule for loop carries (the body's branch back to ``%self`` consumes the old carry by yielding a fresh one of the same type). The current verifier doesn't model the carry-pair relationship between block args and branch operands, so a linear loop-carry value would be flagged as ``LinearLeakError`` even when correctly threaded. No real loop carry is linear today; revisit when one appears.

## Misc
- test_peano's `test_call_jit` _should not_ call the jit, let's verify that it doesn't and put it somewhere more sensible
- `ChainOp` type forwarding: when a pass mutates `op.type` (e.g. shape inference resolves `InferredShapeTensor → Tensor`), wrapping ChainOps keep the old type. Needs a general solution.

## Control flow dialect
- `control_flow.IfOp` and `goto.ConditionalBranchOp` should require `Boolean` conditions. Currently codegen handles non-Boolean via `icmp ne 0`, but conditions should be explicitly Boolean at the dialect level.

## C compiler requests
- Add `mod`, `shift_left`, `shift_right` to the algebra dialect — these are universal integer operations, not C-specific. Every language with integers needs them; currently the C frontend keeps a separate pass just for three ops.
- Make `transitive_dependencies` iterative (explicit stack instead of recursion). Memory token chains create O(n)-depth recursion for n memory ops in a function. Python's default recursion limit (1000) is too shallow for large functions; the C frontend needs `sys.setrecursionlimit(50000)` for sqlite3.
