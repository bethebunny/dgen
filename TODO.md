## Pass framework: scope-bridging footgun
- When a handler creates new values that reference ops from the original block and places them in a *new* block, replacements on the original block don't reach the new block's values. This causes nested lowering to silently miss the new scope.
- **Pattern to follow**: reuse the original block in-place (modify captures/result) instead of creating a new block. See `lower_for` and `lower_if` in `control_flow_to_goto.py` — they reuse `op.body`/`op.then_body`/`op.else_body`.
- **Remaining**: `lower_while` still uses `_lower_block` for explicit recursive lowering because it has two input blocks (condition + body) that need merging into a new structure. Refactor to reuse original blocks.
- Eventually the pass framework should handle this automatically — the pre-computed block list should discover blocks created by handlers, not just the original blocks. This would make `_lower_block` unnecessary.

## Make the JIT work in the general case
- Remove any "if stage0/stage1" logic
- Batch multiple subgraphs in the same staging pass rather than serializing them

## Constraint verification
- Implement expression constraint evaluation in `verify_constraints`. Requires an evaluator that can resolve operand/parameter references, attribute access on resolved types (`input.shape.rank`), comparisons, and arithmetic. Currently only `HasTraitConstraint` is verified; `ExpressionConstraint` and `HasTypeConstraint` are stored but silently skipped. See xfail tests in `test/test_trait.py`.
- Implement `HasTypeConstraint` verification (`requires X has type Tensor`): resolve subject to its type, look up the pattern name in the dialect type registry, check `isinstance`.

## General pass infrastructure
- Passes should guarantee they can lower all ops in their input dialect — add validation that no un-lowered ops survive a pass
- Canonicalization
- Consider removing the generic `builtin.ChainOp` in favor of a monadic effects design (explicit effect tokens threaded through the use-def graph). ChainOp causes dangling chain ops after passes that remove their operands, and the monadic design makes ordering dependencies explicit and composable.

## Block / scope invariants
- Implement `func.recursive` op for recursive functions (see `docs/block-scoping.md` §3.1). Currently recursive functions like `%natural` calling itself via `call<%natural>` violate the DAG property — `block.ops` follows the callee parameter edge back into the function, creating a cycle. `func.recursive` breaks the cycle by providing `%self` as a block argument.

## Actor framework
- Add a loop fusion optimization pass — currently `ActorToAffine` emits separate loops per actor; a general fusion pass would subsume the fused-pipeline special case

## Test infrastructure
- Run verifier on IR test inputs (parsed from ASM) — currently roundtrip tests don't verify captures/scoping invariants on the parsed IR

## Parser / type values
- Parser stores Type objects (e.g. `Index()`) as constant values for TypeType fields. It should store their JSON dict form (`{"tag": "builtin.Index"}`) so `TypeValue.from_json` doesn't need special-casing.

## Experiments / scope creep
- Update `toy.Tensor` to use `Pointer<Array<...>>` — removes the need for the runtime `llvm.load` to extract the data pointer (shape is compile-time, so no indirection is needed)
- Tuple type (`Nil` → `Tuple<[]>`, `Sequence` trait)
- DTypes
- See if we can make `Block` an op in a generic way
- Formally support symbols
- Parser / lowering support for forward references and cyclic references

## Parser / formatting improvements
- Support `%_ = %ref` syntax (alias an SSA ref without an op). Currently blocks whose result is a block argument need `chain(%arg, ())` as a workaround.
- Massively simplify / clean the asm parser and formatter code
- Remove anything that's thinking about "origin"s and generic python types or annotations
- Add parser failure tests
- Parser should reject references to dialects/types that weren't declared via `import`
- Formatter: clearly delineate names in sub-blocks
- Formalize the formatter's op scheduling (currently uses shared `_formatted` dedup set)

## Cleanup
- Remove inline imports
- Remove cases where operands are non-Values (e.g. raw Python ints/strings passed as operand fields)
- Remove spurious utf8 decoding stuff from Memory/Value
- Move `type_asm` to `Type.asm`, `op_asm` to `Op.asm`
- Disambiguate `Type` — it means 3 things: "type value" (in `__params__`), "any type" wildcard (in `__operands__`), and "polymorphic return" (in `-> Type`)

## Block / value infrastructure
- Have values track their uses for forward iteration and fast `replace_uses`

## Codegen
- `SignedInteger`/`UnsignedInteger` layout vs bit-width mismatch: the `.dgen` definition uses `data: Index` (always 64-bit) regardless of `bits`. `llvm_type` currently works around this by using `max(declared_bits, layout_bits)`, but the proper fix is parameterizing the layout on `bits` so a 32-bit integer actually has a 32-bit layout.
- `Executable.run()` lifetime bug: when raw Python values are passed as args, `run()` creates temporary `Memory` objects that can be GC'd before the result is read. For non-register-passable types (e.g. `TypeType`), the JIT returns a pointer into the input Memory's buffer — if that Memory is collected, the result reads garbage. Fix: `run()` should attach input memories to the result's `host_refs`. Workaround in `staging._jit_evaluate` creates memories outside the call.
- Refactor `_emit_func` — 500-line closure with seven dicts of mutable state. Extract into a class or separate functions with explicit state passing.
- Replace isinstance dispatch chain in `_emit_op` with a dispatch table keyed on `(dialect_name, asm_name)`.
- `ChainOp` type forwarding: when a pass mutates `op.type` (e.g. shape inference resolves `InferredShapeTensor → Tensor`), wrapping ChainOps keep the old type. Needs a general solution.

## Misc
- test_peano's `test_call_jit` _should not_ call the jit, let's verify that it doesn't and put it somewhere more sensible

## C compiler requests
- Add `mod`, `shift_left`, `shift_right` to the algebra dialect — these are universal integer operations, not C-specific. Every language with integers needs them; currently the C frontend keeps a separate pass just for three ops.
- Module-level declarations: global variables (`module.global_variable`) and external function declarations (`module.extern`). The C frontend can't reference file-scope variables or forward-declare functions. 1644 of 2569 sqlite3 functions skip because they reference globals.
- Make `transitive_dependencies` iterative (explicit stack instead of recursion). Memory token chains create O(n)-depth recursion for n memory ops in a function. Python's default recursion limit (1000) is too shallow for large functions; the C frontend needs `sys.setrecursionlimit(50000)` for sqlite3.
- Statement sequencing: support ordered sequences of ops where each op implicitly depends on the previous. Currently, side-effecting ops that aren't connected through use-def are invisible to `block.ops`. The C frontend works around this with memory tokens, but a general `sequence` or `block-as-instruction-list` mechanism would be cleaner.
