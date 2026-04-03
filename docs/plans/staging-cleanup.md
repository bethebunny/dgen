# Plan: Staging System Cleanup

## Problem

The staging system (`dgen/staging.py`, ~560 lines) has accumulated codegen
internals, hand-rolled optimization passes, and duplicated IR infrastructure.
Most of this exists because `_jit_evaluate` bypasses the compiler's continuation
and hardcodes `codegen.compile` — forcing staging to manage LLVM linking,
ctypes marshaling, and IR construction that should live elsewhere.

Fixing the `lower` callback to be the full compiler continuation (passes + exit)
removes the need for most of this code.

## Root Cause

`_jit_evaluate` takes `lower: Callable[[Module], Module]` (passes only), then
calls `codegen.compile` directly:

```python
def _jit_evaluate(subgraph, target, lower, ...):
    module = Module(ops=[func])
    lowered = lower(module)          # passes only
    exe = codegen.compile(lowered)   # hardcoded exit
    result = exe.run(...)
    return result.to_json()
```

Callers pass `compiler.run` as `lower`, which applies passes but not the exit
pass. So staging reimplements the exit step, pulling in codegen internals.

The fix: `lower` should be the full continuation — passes + exit. With
`compile(value) -> Constant`, `_jit_evaluate` is subsumed entirely by
`compiler.compile(target)`.

## What Goes Away

### 1. `_jit_evaluate` (lines 91-112) — subsumed by `compiler.compile`

Builds a mini-module, calls `lower`, calls `codegen.compile`, runs the result,
converts to JSON. This is exactly what `compiler.compile(value)` should do.
The entire function disappears.

### 2. `_specialize_ifs` (lines 241-293) — becomes a normal pass

Hand-rolled branch elimination. `isinstance(op, control_flow.IfOp)`, JIT-
evaluates conditions, inlines the taken branch, manually rewires references
with `setattr`. Staging shouldn't know about IfOp. This is an optimization
pass that runs after boundaries are resolved — the compiler's pass pipeline
handles it.

### 3. `_extern_declarations` (lines 61-88) — codegen handles its own linking

Generates LLVM `declare` strings for function calls in staging subgraphs.
Only exists because `_jit_evaluate` calls `codegen.compile` directly and needs
to provide extern declarations. If staging uses the full compiler continuation,
codegen discovers and emits its own externs.

### 4. `_build_callback_thunk` LLVM construction (lines 402-494)

Staging manually constructs `llvm.CallOp`, `FunctionOp`, `Module` to build a
thunk that calls a ctypes callback. Directly calls `codegen.compile` (line 491),
`codegen._ensure_initialized()`, `llvmlite_binding.add_symbol`. Imports
`_ctype`, `_llvm_type` from codegen internals. This is codegen logic that
shouldn't live in staging.

### 5. `compile_module` exit-pass duplication (lines 497-559)

Manually calls `compiler.run(resolved)` then `compiler.exit.run(lowered)` in
two separate code paths (lines 519-521 and 555-558). This reimplements what
`compiler.compile` should do after staging completes. Also directly calls
`codegen._jit_engine` and `llvmlite_binding.add_symbol`.

### 6. `_raw_to_json` / `_make_memories` (lines 51-58, 314-326)

ctypes-to-Python value conversion and Python-to-Memory marshaling. Only exists
because staging manually calls codegen and manages JIT results. With the full
continuation, marshaling stays in codegen/Executable.

### 7. `_trace_dependencies` (lines 37-48) — reimplements existing IR infra

Backward-walks from target to collect needed ops. The IR already has `block.ops`
(transitive dependencies from `block.result`) and `transitive_dependencies`.
This exists because `_jit_evaluate` builds a mini-module from a subgraph rather
than compiling a value directly.

## What Stays

| Function | Role |
|----------|------|
| `compute_stages` | Assign stage numbers to values — core staging logic |
| `_unresolved_boundaries` | Find ops with unresolved `__params__` |
| `_resolve_comptime_field` | Resolve one boundary (calls `compiler.compile`) |
| `resolve_stage0` | Loop: find lowest boundary, resolve, repeat |
| `ConstantFold` | Pass wrapper around `resolve_stage0` |

These are the actual staging algorithm. Everything else is infrastructure that
leaked in because staging bypasses the compiler.

## Target State

```python
# ~50 lines instead of ~560

class ConstantFold(Pass):
    allow_unregistered_ops = True

    def run(self, module, compiler):
        return resolve_stage0(module, compiler)


def resolve_stage0(module, compiler):
    module = deepcopy(module)
    while True:
        # find lowest-stage unresolved boundary
        # compiler.compile(boundary_value) -> Constant
        # patch result back
    return module
```

Staging computes stages, finds boundaries, and calls `compiler.compile` on
subgraphs. It never imports codegen, ctypes, llvmlite, or control_flow.

## Imports That Go Away

```python
# Current imports in staging.py that should not be there:
import ctypes
import llvmlite.binding as llvmlite_binding
from dgen import codegen
from dgen.codegen import Executable, _ctype, _llvm_type
from dgen.dialects import control_flow, llvm
from dgen.dialects.builtin import String
```

## Dependencies

This cleanup depends on:

1. **Fix `lower` to be the full continuation** — the immediate bug. Change
   `ConstantFold.run` and `compile_module` to pass a callback that includes
   the exit pass, not just `compiler.run`.

2. **`compile(value) -> Constant`** — the full design (see
   `docs/plans/value-compilation.md`). Subsumes `_jit_evaluate` entirely.
   Without this, step 1 still requires `_jit_evaluate` to exist but it
   delegates to the compiler instead of hardcoding codegen.

3. **Branch elimination as a pass** — `_specialize_ifs` moves out of staging
   into a pass that the compiler runs. Can be done independently.

4. **Callback thunks in codegen** — `_build_callback_thunk`'s LLVM construction
   moves to codegen. The staging system just says "this function needs a
   runtime callback" and codegen builds the thunk.

## Implementation Order

1. Fix the `lower` callback (immediate, small). `_jit_evaluate` stops calling
   `codegen.compile` directly. Removes `_extern_declarations` import.

2. Extract `_specialize_ifs` into a pass (independent).

3. Move callback thunk construction to codegen (medium).

4. Implement `compile(value) -> Constant` and delete `_jit_evaluate`,
   `_trace_dependencies`, marshaling helpers (the big payoff).
