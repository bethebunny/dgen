## Make the JIT work in the general case
- Polish `Memory` and `Layout` a bit
  - Remove ctype buffer/pointer from `Memory`
  - Remove `_format` from `Type`, it can just have the `Struct` as a field directly.
  - Add tests for types
    - Round trip through ASM literals
    - Round trip through JIT
- Generalize `compile_and_run_staged` to not need an `infer` stage
- SSA types via JIT
- Remove any "if stage0/stage1" logic
- Right now "_jit_evaluate" assumes the result is an int. Should be able to return any type according to `__format__`.
- Batch multiple subgraphs in the same staging pass rather than serializing them
- Figure out what `resolve_constant` does and whether we need it

## dgen dialect definition files + generation
- Create dialect files
- Generate dialects from dialect files

## Experiments / scope creep
- Struct literals
- DTypes
- See if we can make `Block` an op in a generic way.
- Make `call` a generic op in builtin
- Generalized notion of chains
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
- Support broader set of literals

## Easy cleanup
- `affine.load` / `affine.mul_f` / `affine.add_f` should return `f64` (not `()`)
- Fix other ops which should return types but return `Nil` instead
- Figure out why some ASM still doesn't have types, these should fail to parse
- Go through and rename files
- Read, understand, clean passes
- Remove the janky `if args` stuff from `cli.run`
- Remove `hasattr` checks
- Remove `cast`s
- Remove `Any`s
- Remove `type: ignore`s

## Misc
- Write more down into design docs
