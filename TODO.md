## Make the JIT work in the general case
- Remove any "if stage0/stage1" logic
- Batch multiple subgraphs in the same staging pass rather than serializing them

## General pass infrastructure
- Passes should guarantee they can lower all ops in their input dialect — add validation that no un-lowered ops survive a pass
- Canonicalization
- Consider removing the generic `builtin.ChainOp` in favor of a monadic effects design (explicit effect tokens threaded through the use-def graph). ChainOp causes dangling chain ops after passes that remove their operands, and the monadic design makes ordering dependencies explicit and composable.
- `Rewriter.replace_uses` only walks the immediate block — it misses uses inside nested blocks (sub-regions of ops). Fix it to recurse, and add tests for nested-block replacement (the actor `_inline_actor` workaround is a concrete example of this gap)

## Actor framework
- Add an affine loop fusion optimization pass — currently `ActorToAffine` emits separate loops per actor; a general affine-level fusion pass would subsume the fused-pipeline special case

## Parser / type values
- Parser stores Type objects (e.g. `Index()`) as constant values for TypeType fields. It should store their JSON dict form (`{"tag": "builtin.Index"}`) so `TypeValue.from_json` doesn't need special-casing. Affects `value_expression` returning bare types from `_named_type` — these should go through `__constant__.to_json()` when used as constant values.

## IR convenience
- Add a `pack()` helper function (analogous to Python's `set()`) that creates a `PackOp` from a list of values, inferring the element type. Currently every call site manually constructs `PackOp(values=..., type=builtin.List(element_type=...))`.

## Experiments / scope creep
- Update `toy.Tensor` to use `Pointer<Array<...>>` — removes the need for the runtime `llvm.load` to extract the data pointer (shape is compile-time, so no indirection is needed)
- Tuple type
  - `Nil` becomes an alias for `Tuple<[]>`
  - Add a `Sequence` trait so Tuple's type parameter could accept `Array<Type, ...>` or `List<Type>`
- DTypes
- See if we can make `Block` an op in a generic way.
- Try to generalize binary operations, eg. can we have a `builtin.add` op? Does it make sense to model types explicitly as being in a group/field/ring? There's certainly generic optimizations that can be done.
- Formally support symbols
- Parser / lowering support for forward references and cyclic references

## Parser / formatting improvements
- Empty function bodies should format as `%_: Nil = ()` for clarity; truly empty bodies (no ops, just the block header line) aren't supported by the parser/formatter roundtrip
- Formalize the formatter as an explicit scheduling pass. Ambient ops (no block-argument dependencies) are reachable from every block that references them via `walk_ops`. The formatter must assign each op to exactly one textual location. Currently this is handled by a shared `_formatted` dedup set (first-encounter-wins), which is an implicit scheduling policy. Design an explicit scheduler that decides op placement before emission.
- Formalize scheduling in codegen (`dgen/codegen.py`) — same ambient-op placement problem applies when emitting LLVM IR
- Massively simplify / clean the asm parser and formatter code
- Remove anything that's thinking about "origin"s and generic python types or annotations
- Add parser failure tests
- Parser should reject references to dialects/types that weren't declared via `import` — currently `affine.Shape` resolves as long as the dialect is registered at the Python level, even without an `import affine` line in the IR text
- Formatter: clearly delineate names in sub-blocks

## Easy cleanup
- Do a pass and remove inline imports
- Remove cases where operands are non-Values (e.g. raw Python ints/strings passed as operand fields); all operands should be `Value` instances
- Update parser and formatter to support multi-block ops
- Remove spurious utf8 decoding stuff from Memory/Value
- Move `type_asm` to `Type.asm`
- Move `op_asm` to `Op.asm`

## Harder cleanup
- Go through and rename files
- Read, understand, clean passes
- Rewrite passes to generate good IR from the start. Axe `chain_body` and block grouping. `chain_body` currently destroys return value information — the codegen `_find_return_value` heuristic exists solely to recover it by scanning backward for an op whose LLVM type matches the function signature. Once `chain_body` is gone and `block.result` reliably tracks the return value through lowering, `_find_return_value` and the type-matching scan can be deleted.
- Function calls and GOTOs should use the SSA name, not a string
- Label and function values violate closed block semantics. Design this cleanly.
- Disambiguate `Type` — it means 3 things: "type value" (in `__params__`), "any type" wildcard (in `__operands__`), and "polymorphic return" (in `-> Type`). The `__operands__` wildcard and `-> Type` should use a different name or mechanism so `Type` consistently means "type value" per `docs/dialect-files.md`

## Block / value infrastructure
- Eliminate `walk_ops`; implement directly in `Block.ops`
- Have values track their uses for forward iteration and fast `replace_uses`
- Reimplement `Block.ops` as a generator, iterating in topological order from the block arguments following usage

## Misc
- Write more down into design docs
- test_peano's `test_call_jit` _should not_ call the jit, let's verify that it doesn't and put it somewhere more sensible
