# TODO: Implementation Plan

## Phase 1 — Easy cleanup

Low risk, immediate value. Fix rough edges that don't require architectural changes.

### 1a. Fix ops returning Nil that should return actual types

`affine.LoadOp`, `affine.ArithMulFOp`, `affine.ArithAddFOp` all default to `Nil()`.
LoadOp should return the element type (f64 for tensors), and the arith ops should
return `F64Type()`. The corresponding LLVM dialect ops (`llvm.LoadOp`, `llvm.FAddOp`,
`llvm.FMulOp`) also default to `Nil()` and should be fixed.

Files: `toy/dialects/affine.py`, `dgen/dialects/llvm.py`

Check: does `codegen.py`'s `_OP_RESULT_TYPE` table still need to exist after this?
Currently it maps op types to LLVM type strings because the ops themselves don't
carry the right type. If the ops carry correct types, codegen can derive LLVM types
from `op.type.__layout__` instead.

### 1b. Remove the janky `if args` from cli.run

`cli.py:run()` infers parameter types from Python values at runtime (if arg is a
list, set param type to TensorType). This should be handled by the Toy parser —
function parameter type annotations, or at minimum a cleaner separation.

File: `toy/cli.py`

### 1c. Remove hasattr checks

Done — all types are now registered with `@dialect.type()` and have `asm_name`
set. The `format_expr` Type branch directly calls `type_asm()`.

File: `dgen/asm/formatting.py`

### 1d. Remove cast, Any, type: ignore

- `codegen.py:291`: `type: ignore` on `cfunc._argtypes_` — use the already-computed
  param_ctypes from `self.ctype` instead of accessing private ctypes internals.
- `dialect.py:25,34`: `Any` on decorator return types — use TypeVar for proper typing.
- Test file `test_layout.py:56`: `cast(Array, layout)` — use `assert isinstance`.

### 1e. Fix Bytes layout UTF-8 encoding

`layout.py:Bytes.parse()` hardcodes `obj.encode("utf-8")`. This couples a generic
byte layout to a specific encoding. Move encoding responsibility to `String.for_value`
or `Memory.from_value` — `Bytes.parse()` should accept `bytes` directly.

File: `dgen/layout.py`

---

## Phase 2 — FatPointer layouts for String and List

Medium risk, unblocks later work. Currently String and List use static inline layouts
(`Bytes(n)` and `Array(elem, n)`), requiring compile-time-known sizes. Switch to
`FatPointer` (pointer + length) for runtime-flexible sizes.

### 2a. Switch String.__layout__ and List.__layout__ to FatPointer

```python
class String(Type):
    @property
    def __layout__(self) -> FatPointer:
        return FatPointer(BYTE)

class List(Type):
    @property
    def __layout__(self) -> FatPointer:
        return FatPointer(self.element_type.__layout__)
```

This means `length` and `count` are no longer needed as `__params__` on the types
themselves — the length lives in the fat pointer at runtime.

### 2b. Update Memory.from_value for fat pointer types

Currently `Memory.from_value` packs values inline. For FatPointer, it needs to:
1. Allocate a backing buffer for the data
2. Pack (pointer_to_data, length) into the 16-byte FatPointer memory

### 2c. Update codegen.py for fat pointer emission

`_llvm_type(FatPointer(...))` should emit a struct type `{ptr, i64}`. Need to
generate extractvalue/insertvalue or GEP to access pointer and length fields.

### 2d. Update lowering passes

`affine_to_llvm.py` needs to generate alloc + pointer construction for fat pointer
values instead of inline arrays. All places that call `.unpack()` on String/List
memories need updating.

---

## Phase 3 — Parser/formatter cleanup

Medium risk. Simplify the parser by removing implicit type inference.

### 3a. Remove for_value

The parser currently calls `f_type.for_value(raw_value).constant(raw_value)` to
wrap raw literals. Instead, the parser should construct constants directly from
the declared field type. For parameterized types (String, List), the ASM text
should include full type annotations rather than raw literals.

Files: `dgen/asm/parser.py`, `dgen/type.py`, all `for_value` implementations

### 3b. Simplify parser internals

- Remove "special field" concepts — the `__params__`/`__operands__`/`__blocks__`
  system is clean, the parser just needs to drive off it consistently
- Clean up `pending_ops` side-channel (make `_expand_list_sugar` a method)
- Better error messages for unknown types/ops

### 3c. Support multi-block ops in parser/formatter

Current parser only handles `cls.__blocks__[0]`. Loop over all blocks, emit with
labels. Not urgent since no current ops use multiple blocks, but needed for
future if/else, switch, etc.

---

## Phase 4 — Type staging (design TBD)

High risk, high reward. Make `type: Type` a staged value so the staging system
can resolve types along with other params, subsuming shape inference.

### 4a. Make type an honorary __params__

Add `type` to the staging system's view of op fields. `compute_stages` would
assign a stage to each op's type, and `_unresolved_boundaries` would find ops
with unresolved types.

### 4b. Subsume shape inference into staging

If types are staged values, shape inference becomes "resolve the type param"
via the same JIT mechanism used for other params. The `infer` callback to
`compile_and_run_staged` becomes unnecessary.

### 4c. Remove resolve_constant

With type staging, `DimSizeOp` resolves naturally through the staging loop
rather than needing a special `resolve_constant` callback.

### 4d. Make infer optional

Once types are staged, `compile_and_run_staged` no longer requires an `infer`
callback. Default to identity.

---

## Phase 5 — Optimizations

### 5a. Batch subgraph resolution

Group independent boundaries at the same stage into a single mini-module.
One JIT call resolves all of them instead of one-at-a-time.

### 5b. List fields to List type

Make list-valued op fields (e.g., `CallOp.args`, `PhiOp.values`) use the `List`
type natively. Requires FatPointer (Phase 2) to be practical.

### 5c. Less packing/unpacking

Minimize Memory allocation churn in the staging loop. Stream bytes directly
where possible instead of pack → unpack → repack cycles.

---

## Not planned (experiments / scope creep)

These are tracked for future consideration but not part of the current plan:

- Struct literals, Tuple type, DTypes
- Make Block an op, make call a generic builtin op
- Generalized chains and dead code elimination
- Group/field/ring type algebra for binary ops
- Pass infrastructure with input/output types and validation
- dgen dialect definition files + generation
- Function calls via SSA names instead of strings
- Function definitions using normal block syntax
