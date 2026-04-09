# Explicit Staging: `stage` as an Op

## Context

This document proposes replacing the implicit staging machinery (`compute_stages`,
`_is_unresolved`, `_unresolved_boundaries`, callback thunks) with an explicit
`stage` op that passes can emit. The current system has accumulated special cases
that this design eliminates.

### Problem: implicit staging is tangled with op semantics

Today, whether a value triggers staged compilation is determined by its position
in an op's `__params__` vs `__operands__` fields. This conflates two concerns:

1. **Data flow semantics**: does this value flow through at runtime (operand) or
   must it be inspected at compile time (parameter)?
2. **Compilation timing**: does the compiler need to defer work until this value
   is known?

These are not the same thing. `call(f, args)` changed `f` from parameter to
operand because call doesn't need to inspect the function body — it just needs
the pointer. But `differentiate(expr, var)` also takes its inputs as operands
(they're values in the expression graph), yet needs the expression structure
visible at compile time to transform it.

The current staging system also requires special-case exemptions in
`_is_unresolved` for `FunctionOp` — without the exemption, staging would
JIT-evaluate every function reference to a bare pointer, losing the structural
information that passes like autodiff need. This exemption exists because the
system can't distinguish "this is already in a useful form" from "this needs
evaluation."

### Insight: staging is eval, and passes should control when it happens

In the existing staging model (see `staging.md` §2.6–2.7), `constant` is quote
(embed compile-time data as a runtime value) and JIT evaluation is eval
(execute compile-time code to produce a constant). But eval is triggered
*automatically* based on `__params__` analysis, before passes run.

The proposal: let passes emit an explicit `stage` op that means "compile this
block when its inputs are known." Constant folding — which already exists —
evaluates stage ops whose inputs are all constants. Stage ops whose inputs
depend on runtime values survive until runtime, where they become callback
thunks.

## Design

### The `stage` op

```
op stage(args: Span) -> T:
    block body
```

`stage` takes a block (the quoted computation) and a Span of argument values.
When evaluated, it compiles and executes the block with the given arguments,
producing a result of type T.

Semantics: "compile `body` when `args` are known, substituting the block
arguments with the provided values."

**This is explicit quote/eval:**
- The **block** is the quoted form — IR structure, not yet compiled
- **Evaluation** compiles the block through the pass pipeline and executes it
- The **result** is a Constant (the compile-time result of running the block)

### How passes use it

A pass that needs deferred compilation emits a `stage` op. It doesn't need to
know whether the inputs are available now or later — it always emits the same
code, and constant folding handles the timing.

**Example: tile lowering**

```python
@lowering_for(TileOp)
def lower_tile(self, op):
    # Always emit a stage — "compile the allocation when count is known"
    count_arg = BlockArgument(name="count", type=Index())
    data_arg = BlockArgument(name="data", type=op.data.type)
    alloc = allocate_tiled_buffer(count_arg, data_arg)
    return StageOp(
        body=Block(result=alloc, args=[count_arg, data_arg]),
        args=pack([op.count, op.data]),
        type=op.type,
    )
```

If `op.count` is a constant, constant folding evaluates the stage immediately —
equivalent to today's `resolve_stage0`. If `op.count` depends on a runtime
value, the stage survives and becomes a callback thunk at codegen — equivalent
to today's `_build_callback_thunk_for`.

**Example: autodiff lowering**

```python
@lowering_for(DifferentiateOp)
def lower_differentiate(self, op):
    # Always emit a stage — "differentiate when the expression is known"
    val_arg = BlockArgument(name="value", type=Float64())
    var_arg = BlockArgument(name="var", type=Float64())
    diff = DifferentiateOp(value=val_arg, var=var_arg, type=Float64())
    return StageOp(
        body=Block(result=diff, args=[val_arg, var_arg]),
        args=pack([op.value, op.var]),
        type=Float64(),
    )
```

When both `op.value` and `op.var` are block-local expressions (stage 0),
constant folding evaluates the stage immediately — the AD pass runs inside the
evaluation, producing the derivative expression. When `op.value` depends on a
runtime function argument, the stage defers to runtime.

### Constant folding replaces the staging pass

The current staging infrastructure (`compute_stages`, `_unresolved_boundaries`,
`resolve_stage0`, `_build_callback_thunk_for`) is replaced by a single rule:

> **Fold any `stage` op whose args are all constants.**

"Constant" means `isinstance(value, Constant)` — a value with known data,
including types (which should be Constant subtypes). Folding a stage op means:
compile its block through the pass pipeline, execute it, and replace the stage
op with a `ConstantOp` holding the result.

Stage ops whose args are NOT all constants survive through compilation and
lower to callback thunks at codegen — the same mechanism as today's runtime-
dependent staging, but triggered by the explicit op rather than by parameter
analysis.

### What `__params__` becomes

With explicit staging, `__params__` no longer drives compilation timing. It
becomes purely a **data flow annotation**: "this field is a compile-time
attribute of the op, not a runtime SSA operand." The distinction still matters
for:

- **ASM syntax**: parameters use `<angle brackets>`, operands use `(parens)`
- **Serialization**: parameters are part of the op's identity
- **Type derivation**: an op's result type may depend on its parameters

But parameters no longer trigger staging. A pass that needs compile-time
resolution of a value emits `stage` explicitly. This eliminates:

- `_is_unresolved` and its `FunctionOp` exemption
- `compute_stages` and stage number computation
- `_unresolved_boundaries`
- The `+1` stage bump rule for parameters
- `resolve_stage0` as a pre-pass
- The distinction between "stage-0 evaluable" and "runtime-dependent" in the
  staging infrastructure

### What happens to `call`

`call(callee, args)` — callee is an operand (current design, unchanged). No
staging implications. The callee is a runtime value (function pointer).

If a pass needs to inline or inspect a function, it checks `isinstance(callee,
FunctionOp)` directly. If the callee isn't known yet, the pass can emit a
`stage` to defer until it is.

### What happens to `FunctionOp`

`FunctionOp` is no longer special-cased in staging. It's an op that produces a
`Value[Function]`. Its block body is structural IR data, inspectable by passes.

When a `FunctionOp` flows into a `stage` op as an argument and the stage is
folded, the FunctionOp is available to the compilation pipeline inside the
stage — passes can inspect its body, inline it, differentiate through it, etc.

No `_is_unresolved` exemption needed. No `isinstance(value, FunctionOp)` check
in staging. Functions are just values.

## Comparison

| Aspect | Current (implicit) | Proposed (explicit) |
|---|---|---|
| Staging trigger | `__params__` field analysis | Pass emits `stage` op |
| When to evaluate | `compute_stages` + boundary detection | Constant folding: args all constant? fold. |
| Runtime deferral | Automatic callback thunk for stage-1+ params | `stage` op survives to codegen → thunk |
| Function exemption | `FunctionOp` in `_is_unresolved` | None needed |
| `__params__` meaning | "staging boundary" + compile-time attribute | Compile-time attribute only |
| Pass involvement | Passes don't know about staging | Passes emit `stage` when they need deferred compilation |
| Complexity | `compute_stages`, `_unresolved_boundaries`, `resolve_stage0`, `_build_callback_thunk_for` | Constant folding + codegen thunk lowering |

## Open questions

### Is `stage` a builtin or per-dialect?

`stage` could be a builtin op (like `constant`) since any dialect might need
deferred compilation. Or it could be that each dialect emits its own deferred
patterns and `stage` is just a common one.

### Recursive staging

A `stage` body may itself contain `stage` ops (e.g., `tile` inside a
`differentiate`). Constant folding handles this naturally — the inner stage
folds when its args become known during the outer stage's evaluation. This is
the multi-stage computation model, driven by folding rather than by stage
numbers.

### Relationship to `Constant`

This design assumes `Constant` is the right contract for "resolved." As
discussed separately, `Type` should likely be a `Constant` subclass (it's a
compile-time-known value of TypeType). The `stage` folding rule becomes
simply: fold when all args are `Constant`.

### Block as compile-time representation of Function

A `FunctionOp`'s block body is the compile-time representation of a Function
value, while `Pointer<Nil>` is the runtime representation. Staging (the `stage`
op) is the boundary between these representations — it compiles the block to
produce the pointer. This is the two-faces model from `staging.md` §2.1,
made explicit as an op rather than implicit infrastructure.

### Interaction with the `differentiate` op

The value-level `differentiate(value, var)` op operates on expression graphs
within a block. It composes naturally with `stage`: if differentiate can see the
full expression (all ops between value and var are known), it lowers directly.
If part of the expression is opaque (e.g., a call through an unknown function),
the autodiff pass wraps the differentiate in a `stage` — deferring until the
opaque value becomes concrete.

This eliminates the need for function-level gradient ops (`grad<f>`), the
FunctionOp exemption in staging, and the special-case inlining logic. The
autodiff pass simply says "compile this differentiation when the expression is
available" and the infrastructure handles the rest.

## Summary

Replace implicit staging (parameter analysis → automatic JIT evaluation) with
explicit staging (passes emit `stage` ops → constant folding evaluates them).
This eliminates the FunctionOp special case, simplifies the staging
infrastructure, and gives passes direct control over deferred compilation. The
Lisp analogy: quote/eval become explicit ops rather than implicit consequences
of field annotations.
