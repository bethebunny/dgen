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

## Block / value infrastructure
- Have values track their uses for forward iteration and fast `replace_uses`
- Make `Block` a `Value`

## Constraint verification
- Extend `_verify_has_type` to resolve pattern names against op parameters/operands as well as global type names. The motivating case: `existential.pack<bound: Type>(value) requires value has type bound` — the pattern `bound` is a parameter name, not a dialect type. Today the verifier crashes with `unknown type 'bound'`. Fix is small: try `_resolve_subject(pattern.name, op)` first; if it returns a Value[TypeType], compare via `constant(...)` equality; otherwise fall through to the dialect lookup with `isinstance`. Once landed, re-add the constraint to `existential.pack` (currently dropped).

## Existentials
- Pack lowering currently heap-allocates the Some struct (and the inner value buffer) so the bytes outlive the producing function frame, but byte-level round-trip is still broken because the witness `TypeType` constant gets stored with one extra level of indirection. The proximate fix would be making small `Record` layouts register-passable so non-register-passable value handling doesn't conflate "the bytes" with "a pointer to the bytes" — see also the bullet below. Once that's resolved, drop the heap leak in favour of stack alloc (or keep heap until we have a destructor story).
- Make small `Record` layouts (≤16 bytes — typically two pointers or two i64s) register-passable. Today every `Record` is non-register-passable, which means the LLVM-level value is `ptr` (the address of the bytes) and functions return such records by writing into a caller-supplied buffer. That convention prevents constructing a Record inside a function and returning it by value. Returning small aggregates as `{ptr, ptr}` (or `{i64, i64}`, etc.) would let `pack` flow Some<bound> through registers, drop the heap allocation, and unblock value-level round-trip tests. Touches: `Layout.register_passable`, `value_reference`, `llvm_type`, ctypes return-type construction in `Executable.run`, and `Memory.from_value`/`from_raw` for aggregate returns.
- Heap-allocated pack leaks every Some struct + every inner-value buffer. No destructor or GC. Fine for prototyping; revisit when we have a lifetime story.
- The existing `memory.heap_allocate` lowering hardcodes a `count * 8`-byte malloc — every allocation is sized in 8-byte units. Pack rounds up the inner-value count to the next 8-byte chunk to compensate, but small types (`Byte`) over-allocate and types whose size isn't a multiple of 8 lose their tail bytes. Real fix: pass the byte size through `heap_allocate` instead of hardcoding doubles.
- Symmetric `unpack_existential` op for compile-time projection — a stage-boundary that resolves the witness type and yields a typed value.
- Spec-syntax support for `has trait <param-name>`. The `existential.Some<bound: Type>` projection over its `bound` parameter is currently a hand-written monkey-patch in `dgen/dialects/__init__.py` because the spec parser/builder only resolves trait names to classes at build time. Lifting this into the spec syntax would let any parameterised type declare a trait that depends on its parameters, removing the special case.
- Compound constants (`Any`, `Some<X>`, future record-shaped existentials) currently store their rich payload as a Python dict produced by `Layout.Record.to_native_value`, and `Constant.required_dialects` peeks through dicts/lists to find embedded `Value`s. The structurally clean version is to make compound constants' rich form a tree of real `Value`s (e.g. a synthetic `pack_existential(witness=<Type>, value=<Value>)` op) so `transitive_dependencies` walks them naturally and the dict-peeking helper goes away. Probably falls out of the runtime `pack_existential` work above.

## Recursive types
- The layout system can't currently express recursive type definitions: `type Foo: data: Bar` / `type Bar: data: Foo` would loop forever in `_make_layout` since each side eagerly expands the other's layout. To support this we'd need either auto-boxing on cycle detection (insert a `Pointer` indirection when a layout would recurse into a type already on the stack) or an explicit "boxed" annotation in `.dgen`. The current `Some<bound>` / `Any` definition sidesteps this by using `Pointer<Nil>` as the value field — a fixed-size opaque slot — rather than a true mutually-recursive `value: Any`.

## Codegen
- `SignedInteger`/`UnsignedInteger` layout vs bit-width mismatch: the `.dgen` definition uses `data: Index` (always 64-bit) regardless of `bits`. `llvm_type` currently works around this by using `max(declared_bits, layout_bits)`, but the proper fix is parameterizing the layout on `bits` so a 32-bit integer actually has a 32-bit layout.
- `Executable.run()` lifetime bug: when raw Python values are passed as args, `run()` creates temporary `Memory` objects that can be GC'd before the result is read. For non-register-passable types (e.g. `TypeType`), the JIT returns a pointer into the input Memory's buffer — if that Memory is collected, the result reads garbage. Fix: `run()` should attach input memories to the result's `host_refs`. Workaround in `staging._jit_evaluate` creates memories outside the call.

## Misc
- test_peano's `test_call_jit` _should not_ call the jit, let's verify that it doesn't and put it somewhere more sensible
- `ChainOp` type forwarding: when a pass mutates `op.type` (e.g. shape inference resolves `InferredShapeTensor → Tensor`), wrapping ChainOps keep the old type. Needs a general solution.

## Control flow dialect
- `control_flow.IfOp` and `goto.ConditionalBranchOp` should require `Boolean` conditions. Currently codegen handles non-Boolean via `icmp ne 0`, but conditions should be explicitly Boolean at the dialect level.

## C compiler requests
- Add `mod`, `shift_left`, `shift_right` to the algebra dialect — these are universal integer operations, not C-specific. Every language with integers needs them; currently the C frontend keeps a separate pass just for three ops.
- Make `transitive_dependencies` iterative (explicit stack instead of recursion). Memory token chains create O(n)-depth recursion for n memory ops in a function. Python's default recursion limit (1000) is too shallow for large functions; the C frontend needs `sys.setrecursionlimit(50000)` for sqlite3.
