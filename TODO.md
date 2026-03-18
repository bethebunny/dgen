## Make the JIT work in the general case
- ~~Generalize `compile_and_run_staged` to not need an `infer` stage~~ Done: `Compiler` owns the full pipeline; staging no longer takes `infer`/`lower` callbacks
- Remove any "if stage0/stage1" logic
- Batch multiple subgraphs in the same staging pass rather than serializing them
- ~~Delete `DimSizeOp.resolve_constant` monkey-patch~~ Done: `DimSizeOp` now has a proper lowering in `toy_to_affine.py`

## General pass infrastructure
- Passes should guarantee they can lower all ops in their input dialect — add validation that no un-lowered ops survive a pass

## Parser / type values
- Parser stores Type objects (e.g. `Index()`) as constant values for TypeType fields. It should store their JSON dict form (`{"tag": "builtin.Index"}`) so `TypeValue.from_json` doesn't need special-casing. Affects `value_expression` returning bare types from `_named_type` — these should go through `__constant__.to_json()` when used as constant values.

## Experiments / scope creep
- Rename `List` / `FatPointer` to `Span`
- Update `toy.Tensor` to use `Pointer<Array<...>>` — removes the need for the runtime `llvm.load` to extract the data pointer (shape is compile-time, so no indirection is needed)
- Tuple type
  - `Nil` becomes an alias for `Tuple<[]>`
  - Add a `Sequence` trait so Tuple's type parameter could accept `Array<Type, ...>` or `List<Type>`
- DTypes
- See if we can make `Block` an op in a generic way.
- Make `call` a generic op in builtin
- Generalized notion of chains
  - `builtin.chain(%lhs : A, %rhs : B) -> A` which just retuns `%lhs`
  - no chain type! The following is fine:
    ```
      %0 : () = ...
      %1 : () = ...
      %2 : () = chain(%0, %1)
      %_ : () = return(%2)
    ```
- Generalized dead code elimination (mutable ops _must_ be chained)
- Try to generalize binary operations, eg. can we have a `builtin.add` op? Does it make sense to model types explicitly as being in a group/field/ring? There's certainly generic optimizations that can be done.
- Pass input/output types, pre/post validation
- Simple pass infrastructure
- Formally support symbols
- Parser / lowering support for forward references and cyclic references

## Parser / formatting improvements
- Massively simplify / clean the asm parser and formatter code
- Remove anything that's thinking about "origin"s and generic python types or annotations
- What are "special fields" in the parser? They shouldn't exist
- Add parser failure tests
- Parser should reject references to dialects/types that weren't declared via `import` — currently `affine.Shape` resolves as long as the dialect is registered at the Python level, even without an `import affine` line in the IR text
- Formatter: use %_ for unused outputs
- Formatter: clearly delineate names in sub-blocks

## Easy cleanup
- Migrate `toy/test/test_end_to_end.py` to snapshot testing (using `IRSnapshotExtension` / `graph_equivalent`) so that expected IR strings don't need manual updates when codegen changes
- Remove cases where operands are non-Values (e.g. raw Python ints/strings passed as operand fields); all operands should be `Value` instances


- Figure out why some ASM still doesn't have types, these should fail to parse
- Update parser and formatter to support multi-block ops
- Remove spurious utf8 decoding stuff from Memory/Value
- Move `type_asm` to `Type.asm`
- Move `op_asm` to `Op.asm`

## Harder cleanup
- Go through and rename files
- Read, understand, clean passes
- Rewrite passes to generate good IR from the start. Axe `chain_body` and block grouping.
- Function calls and GOTOs should use the SSA name, not a string
- Label and function values violate closed block semantics. Design this cleanly.
- Disambiguate `Type` — it means 3 things: "type value" (in `__params__`), "any type" wildcard (in `__operands__`), and "polymorphic return" (in `-> Type`). The `__operands__` wildcard and `-> Type` should use a different name or mechanism so `Type` consistently means "type value" per `docs/dialect-files.md`

## Block / value infrastructure
- Remove the `ops=` constructor from `Block`
- Eliminate `walk_ops`; implement directly in `Block.ops`
- Have values track their uses for forward iteration and fast `replace_uses`
- Reimplement `Block.ops` as a generator, iterating in topological order from the block arguments following usage

## Misc
- Write more down into design docs
- test_peano's `test_call_jit` _should not_ call the jit, let's verify that it doesn't and put it somewhere more sensible
