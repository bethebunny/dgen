# Staged Computation: Implementation Architecture

This document describes the implementation of staged computation in DGEN — the mechanism by which compile-time (`__params__`) values are resolved, including values that depend on runtime input.

See `staging.md` for the conceptual model. This document covers the concrete implementation.

## Overview

Some op fields are declared in `__params__`, meaning their concrete value must be known before the next compilation phase can proceed (e.g., shape inference needs a tile count). The staging system assigns every value a **stage number** and resolves `__params__` boundaries in stage order.

## Stage Numbers

Every value in a function is assigned a stage number via `compute_stages(func)`.

**Base cases:**
- `Constant` / `ConstantOp`: stage 0
- `BlockArgument` (function input): stage 1

**Ops:**
```
stage = max((
    *(1 + stage(p) for p in __params__),
    *(stage(v) for v in __operands__),
))
```

The `+1` applies only to `__params__` — these are stage boundaries where a compile-time value must be resolved before the op can proceed. `__operands__` (runtime SSA values) flow through without adding a stage.

**Example:**
```
%0 = constant(2)            # stage 0
%1 = constant([1,2,3])      # stage 0
%2 = add_index(%0, %0)      # no __params__, max(0,0) = 0
%x = block_arg              # stage 1
%3 = tile<%2>(%1)           # has __params__, max(0, 0+1) = 1
%n = nonzero_count(%x)      # no __params__, max(1) = 1
%4 = tile<%n>(%1)           # has __params__, max(0, 1+1) = 2
```

### Op.ready

`Op.ready` checks whether all `__params__` fields are resolved `Constant` instances:

```python
@property
def ready(self) -> bool:
    return all(isinstance(getattr(self, name), Constant) for name, _ in self.__params__)
```

An op's stage number tells you *when* it can become ready. `Op.ready` tells you whether it *is* ready right now.

## Key Functions

All staging logic lives in `dgen/passes/staging.py`.

### Stage Computation

`compute_stages(func)` returns `dict[int, int]` mapping `id(value) → stage_number` for every value in the function. Uses recursive memoization over the dataflow graph.

`_unresolved_boundaries(func, stages)` finds ops with unresolved `__params__` (non-Constant Value fields), returns `(stage, op, field_name, param_value)` tuples sorted by stage number.

### Stage Classification

`_is_stage0_evaluable(target)` walks the dependency tree of a `Value`. If any node is a `BlockArgument` (function parameter), the subgraph depends on runtime input and is NOT stage-0 evaluable.

### Dependency Tracing

`_trace_dependencies(target, func)` backward-walks from a target value through its operands, returning the ops from the function body in topological order. This gives the minimal subgraph needed to compute `target`.

## Resolution: `_resolve_all_comptime`

Resolves `__params__` fields in stage order:

1. Compute stage numbers via `compute_stages`
2. Find unresolved boundaries via `_unresolved_boundaries` (sorted by stage)
3. Process the lowest-stage boundary:
   - If stage-0 evaluable → extract subgraph, JIT in isolation, replace with `ConstantOp`
   - If not stage-0 evaluable → stop (remaining boundaries need runtime args)
4. After each resolution, run `infer()` to propagate types, then `_resolve_constant_ops` to replace ops like `DimSizeOp` with constants
5. Re-compute stages (the graph has changed) and repeat

This replaces an earlier fixpoint-scan approach with structured stage-ordered processing.

### Isolated JIT (Stage-0 Boundaries)

When a `__params__` subgraph is self-contained (all leaves are `ConstantOp`s):

1. Extract the dependency subgraph
2. Build a mini-module: the subgraph ops + `ReturnOp(target)`
3. Lower through the pipeline (toy → affine → LLVM)
4. JIT-compile and call — get the result as a Python value
5. Create a `ConstantOp` with the result and patch the `__params__` field

This handles cases like `tile(%data, add_index(2, 3))`.

### Runtime-Dependent Boundaries

When the `__params__` subgraph reaches a `BlockArgument`, isolated JIT is impossible — the value depends on runtime input. The solution: build a callback-based executable.

Given:
```
function (%x: Tensor([4], f64)) -> ():
    %1 = nonzero_count(%x)                    ← stage 1 (depends on block arg)
    %2 = [7.0, 8.0, 9.0]                      ← stage 0
    %3 = tile<%1>(%2)    # count in __params__ ← stage 2 (boundary)
    %4 = print(%3)                             ← stage 2
    return()
```

## Compilation Paths

```
compile_staged(module, infer, lower)
│
├─ 1. _resolve_all_comptime (stage-ordered)
│     Compute stages → find lowest unresolved boundary
│     If stage-0 evaluable → resolve in isolation
│     If runtime-dependent → stop
│
├─ 2. Check for remaining unresolved boundaries
│
├─ [if none] 3a. Compile directly: infer → lower → codegen
│
└─ [if some] 3b. Build callback-based executable:
      ├─ Build stage-2 template (ops with __params__ as placeholders)
      ├─ Create host callback (ctypes CFUNCTYPE):
      │     Receives all function args at runtime
      │     Resolves __params__ in stage order with runtime args
      │     JIT-compiles stage-2: infer → lower → codegen → run
      ├─ Register callback with llvmlite (add_symbol)
      └─ Compile stage-1 thunk: calls callback, returns result
```

Each `exe.run(...)` call triggers the callback, which JIT-compiles a specialized stage-2 for the given runtime values.

The callback closure uses the same `compute_stages` / `_unresolved_boundaries` loop, but without the stage-0-evaluable gate — runtime args are available, so all boundaries can be resolved.

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

### Pointer-crossing

Stage-2 code can access the original function parameters (e.g., the tensor passed to the function). `compile_and_run_staged` preserves block args through compilation and passes runtime args at the run boundary.

### Chained Stage Boundaries

After each `__params__` resolution, `infer()` is called to propagate types with the newly-known constants. Ops that implement `resolve_constant()` (e.g., `DimSizeOp`) are then replaced with `ConstantOp`s. This enables chained dependencies where a second `__params__` value depends on the *shape* resolved by the first. Stage numbers are re-computed after each resolution step to account for these graph changes.

### Remaining Restrictions

The current `compile_staged` callback path handles the case where unresolved `__params__` values are directly `BlockArgument`s (function parameters). The `compile_and_run_staged` path handles the general case (computed stage-1 values like `nonzero_count`). Unifying these — native stage-1 computation followed by a callback — is a future extension.

## File Map

| File | Role |
|------|------|
| `dgen/passes/staging.py` | Stage computation, dependency tracing, staged resolution, `compile_staged` |
| `dgen/op.py` | `Op.ready` property |
| `dgen/llvm/codegen.py` | LLVM IR emission with function parameters, JIT with args |
| `dgen/value.py` | `Constant` / `Value` base classes |
| `toy/passes/toy_to_affine.py` | Block arg propagation, TileOp lowering |
| `toy/passes/affine_to_llvm.py` | Block arg propagation with shape tracking |
| `toy/passes/shape_inference.py` | Resolves tile shapes after comptime fields are constants |
| `toy/test/test_staging.py` | Tests for all staging scenarios |
