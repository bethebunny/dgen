## Make the JIT work in the general case
- Memory layout for types
  - Make `type: Type` an honorary `__params__` on `Op`
- Generalize `compile_and_run_staged` to not need an `infer` stage
- Remove any "if stage0/stage1" logic
- Batch multiple subgraphs in the same staging pass rather than serializing them
- Delete `DimSizeOp.resolve_constant` monkey-patch and the `getattr(op, "resolve_constant")` pattern in `staging.py` — declare it as an optional method on `Op` or remove entirely

## Experiments / scope creep
- Tuple type
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

## Parser / formatting improvements
- Massively simplify / clean the asm parser and formatter code
- Remove anything that's thinking about "origin"s and generic python types or annotations
- What are "special fields" in the parser? They shouldn't exist
- Add parser failure tests
- Formatter: use %_ for unused outputs
- Formatter: clearly delineate names in sub-blocks

## Easy cleanup
- Figure out why some ASM still doesn't have types, these should fail to parse
- Update parser and formatter to support multi-block ops
- Remove spurious utf8 decoding stuff from Memory/Value
- Move `type_asm` to `Type.asm`
- Move `op_asm` to `Op.asm`

## Harder cleanup
- Go through and rename files
- Read, understand, clean passes
- Function calls and GOTOs should use the SSA name, not a string

## Misc
- Write more down into design docs
