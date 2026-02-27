# Staged Computation: Implementation Architecture

This document describes the implementation of staged computation in DGEN — the mechanism by which compile-time (`Comptime`) values are resolved, including values that depend on runtime input.

See `staging.md` for the conceptual model. This document covers the concrete implementation.

## Overview

Some op fields are annotated `Comptime`, meaning their concrete value must be known before the next compilation phase can proceed (e.g., shape inference needs a tile count). There are two cases:

1. **Stage 0**: The comptime value depends only on constants. Extract the dependency subgraph, JIT it in isolation, replace the field with a `ConstantOp`.

2. **Stage 1**: The comptime value depends on a function parameter (runtime input). The function must be split: stage 1 computes the comptime value at runtime, then stage 2 is compiled with that value baked in as a constant.

```
             Stage 0                          Stage 1
   ┌───────────────────────┐    ┌──────────────────────────────────┐
   │ add_index(2, 3) → 5   │    │ nonzero_count(%param) → ???     │
   │                       │    │                                  │
   │ Self-contained.       │    │ Depends on runtime input.        │
   │ JIT in isolation.     │    │ Must split function into two     │
   │ Replace with const 5. │    │ stages and JIT stage 1 first.    │
   └───────────────────────┘    └──────────────────────────────────┘
```

## Key Types and Functions

All staging logic lives in `dgen/staging.py`.

### Comptime Annotation

`Comptime` (defined in `dgen/value.py`) is a type-level marker on op fields. It's a subclass of `Value` used purely as a type hint — at runtime the field holds a regular `Value`. The staging evaluator inspects field annotations to find which ops need resolution.

```python
@toy.op("tile")
@dataclass(eq=False, kw_only=True)
class TileOp(Op):
    input: Value
    count: Comptime      # ← must be resolved before shape inference
    type: Type
```

### Stage Classification

`_is_stage0_evaluable(target)` walks the dependency tree of a `Value`. If any node is a `BlockArgument` (function parameter), the subgraph depends on runtime input and is NOT stage-0 evaluable.

```python
def _is_stage0_evaluable(target: Value) -> bool:
    # Walk operands recursively
    # Return False if any BlockArgument is found
```

### Dependency Tracing

`_trace_dependencies(target, func)` backward-walks from a target value through its operands, returning the ops from the function body in topological order. This gives the minimal subgraph needed to compute `target`.

## Stage 0: Isolated JIT

When a comptime subgraph is self-contained (all leaves are `ConstantOp`s), `compile_and_run_staged` resolves it in isolation:

1. Extract the dependency subgraph
2. Build a mini-module: the subgraph ops + `ReturnOp(target)`
3. Lower through the pipeline (toy → affine → LLVM)
4. JIT-compile and call — get the result as a Python value
5. Create a `ConstantOp` with the result and patch the comptime field

This handles cases like `tile(%data, add_index(2, 3))`.

## Stage 1: Runtime-Dependent Comptime Values

When the comptime subgraph reaches a `BlockArgument`, isolated JIT is impossible — the value depends on runtime input. The solution: split the function at the stage boundary.

### Function Splitting

Given:
```
function (%x: Tensor([4], f64)) -> ():
    %1 = nonzero_count(%x)                        ← comptime subgraph
    %2 = [7.0, 8.0, 9.0]                          ← stage 2
    %3 = tile(%2, %1)      # count is Comptime     ← stage boundary
    %4 = print(%3)                                 ← stage 2
    return()
```

The function is split into:

**Stage 1** — computes the comptime value:
```
function (%x: Tensor([4], f64)) -> index:
    %1 = nonzero_count(%x)
    return(%1)
```

**Stage 2 template** — the rest of the function, with the comptime value as a placeholder:
```
function () -> ():
    %2 = [7.0, 8.0, 9.0]
    %count = <resolved at runtime>
    %3 = tile(%2, %count)
    %4 = print(%3)
    return()
```

### Execution Flow

There are two compilation paths:

**`compile_and_run_staged`** — one-shot: compiles and runs in one call. Stage-1 values are resolved Python-side with runtime args before final compilation.

**`compile_staged`** — compile-once, run-many: returns an `Executable` that can be called multiple times with different arguments. When unresolved `__params__` remain after stage-0, builds a callback-based executable: the compiled code calls a host callback that JIT-compiles stage-2 with the resolved values at runtime.

```
compile_staged(module, infer, lower)
│
├─ 1. _resolve_all_comptime (stage-0 only, no args needed)
│     Scan for __params__ fields
│     If stage-0 evaluable → resolve in isolation
│     If stage-1 needed → skip
│
├─ 2. Check for remaining unresolved __params__
│
├─ [if none] 3a. Compile directly: infer → lower → codegen
│
└─ [if some] 3b. Build callback-based executable:
      ├─ Build stage-2 template (ops with __params__ as placeholders)
      ├─ Create host callback (ctypes CFUNCTYPE):
      │     Receives all function args at runtime
      │     Patches __params__ with constants from args
      │     JIT-compiles stage-2: infer → lower → codegen → run
      ├─ Register callback with llvmlite (add_symbol)
      └─ Compile stage-1 thunk: calls callback, returns result
```

Each `exe.run(...)` call triggers the callback, which JIT-compiles a specialized stage-2 for the given runtime values.

### Argument Passing

Function parameters flow through the entire pipeline:

1. **Toy IR**: `Block.args` carries `BlockArgument`s with types
2. **toy_to_affine**: Block args registered in `alloc_map` as themselves (a parameter tensor is already an allocation)
3. **affine_to_llvm**: Block args registered in `value_map`, `alloc_shapes`, `alloc_sizes`
4. **codegen**: Block args emitted as LLVM function parameters (`ptr %0`, `i64 %1`, etc.)
5. **JIT call**: `_func_param_ctypes` maps block arg types to ctypes; `compile_and_run`/`jit_eval` accept an `args` list

Type mapping for parameters:

| IR Type | LLVM Type | ctypes |
|---------|-----------|--------|
| `TensorType` (has `shape`) | `ptr` | `c_void_p` |
| `IndexType` | `i64` | `c_int64` |
| `F64Type` | `double` | `c_double` |

Python-side argument preparation (`_prepare_ctypes_args`):
- Lists of floats → `(c_double * N)(*values)` → cast to `c_void_p`
- Integers and floats → passed directly

## TileOp Lowering

`TileOp` lowers in `toy_to_affine` by generating nested `ForOp` loops:

```
tile(input: Tensor([3], f64), count: 2) → Tensor([2, 3], f64)
```

Generates:
```
alloc output[2, 3]
for i in 0..2:           # outer: tile count
    for j in 0..3:       # inner: input shape
        output[i, j] = input[j]
```

The count must be a `ConstantOp` by the time lowering runs — staging has resolved it. The output shape is `[count] + input_shape`.

## Resolved Extensions

The staging system supports three capabilities beyond the basic stage-0/stage-1 split:

### Pointer-crossing

Stage-2 code can access the original function parameters (e.g., the tensor passed to the function). `compile_and_run_staged` preserves block args through compilation and passes runtime args at the run boundary.

### Multiple Comptime Fields

Both `_resolve_all_comptime` (stage-0) and the stage-1 loop in `compile_and_run_staged` use an iterative `while changed` loop that rescans after each resolution. Multiple independent boundaries in the same function are resolved one at a time, with subgraph ops removed and replaced by constants after each step.

### Arbitrary Stages (Interleaved Shape Inference)

After each comptime resolution, `infer()` is called to propagate types with the newly-known constants. Ops that implement `resolve_constant()` (e.g., `DimSizeOp`) are then replaced with `ConstantOp`s. This enables chained dependencies where a second comptime value depends on the *shape* resolved by the first.

### Runtime JIT Callbacks (`compile_staged`)

When `compile_staged` finds unresolved `__params__` that depend on runtime values (BlockArguments), it builds a callback-based executable. The compiled LLVM code calls a host callback (`ctypes.CFUNCTYPE`) that:

1. Reads the runtime `__params__` values from the callback arguments
2. Deep-copies the stage-2 template and patches the `__params__` fields with `ConstantOp`s
3. Compiles and runs stage-2 through the full pipeline (infer → lower → codegen → JIT)

This enables compile-once, run-many: the same `Executable` can be called with different arguments, each time JIT-compiling a specialized stage-2.

### Remaining Restrictions

The current `compile_staged` callback path handles the case where unresolved `__params__` values are directly `BlockArgument`s (function parameters). The `compile_and_run_staged` path handles the general case (computed stage-1 values like `nonzero_count`). Unifying these — native stage-1 computation followed by a callback — is a future extension.

## File Map

| File | Role |
|------|------|
| `dgen/staging.py` | Stage analysis, dependency tracing, function splitting, `compile_and_run_staged` |
| `dgen/codegen.py` | LLVM IR emission with function parameters, JIT with args |
| `dgen/value.py` | `Comptime` type marker |
| `toy/passes/toy_to_affine.py` | Block arg propagation, TileOp lowering |
| `toy/passes/affine_to_llvm.py` | Block arg propagation with shape tracking |
| `toy/passes/shape_inference.py` | Resolves tile shapes after comptime fields are constants |
| `toy/test/test_staging.py` | Tests for all staging scenarios |
