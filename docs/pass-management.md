# Redesign PassManager → Compiler

## Motivating Example

Consider a Toy program that tiles a tensor by a computed count:

```
def main() {
    var data = [1, 2, 3];
    var t = tile(data, add_index(2, 2));
    print(t);
    return;
}
```

After lowering to IR (before any passes):

```
%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<affine.Shape<1>([3]), F64> = [1.0, 2.0, 3.0]
    %1 : Index = 2
    %2 : Index = 2
    %3 : Index = add_index(%1, %2)
    %4 : toy.InferredShapeTensor<F64> = toy.tile<%3>(%0)
    %5 : Nil = toy.print(%4)
    %_ : Nil = return(())
```

`%4` (TileOp) has a compile-time parameter `count = %3`, where `%3` is an `add_index` op — a Value, not a Constant. So `%4.ready` is **False**: its shape depends on a computation that hasn't been evaluated yet.

ShapeInference cannot resolve `%4`'s type either — `_resolve_index_value(%3)` returns None because `%3` is not a ConstantOp. The tile stays `InferredShapeTensor`, and ToyToAffine can't lower it (no shape → no loop bounds).

The staging evaluator resolves this: it JIT-compiles the subgraph `{%1, %2, %3}` to get `4`, replaces `%3` with `ConstantOp(4)`, and now TileOp's count is a Constant. After staging:

```
%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<affine.Shape<1>([3]), F64> = [1.0, 2.0, 3.0]
    %1 : Index = 4
    %2 : toy.InferredShapeTensor<F64> = toy.tile<%1>(%0)
    %3 : Nil = toy.print(%2)
    %_ : Nil = return(())
```

Now `%2.ready` is True (count is a Constant). ShapeInference can resolve `_resolve_index_value(%1) → 4` and compute the output shape `[4, 3]`:

```
%main : Nil = function<Nil>() ():
    %0 : toy.Tensor<affine.Shape<1>([3]), F64> = [1.0, 2.0, 3.0]
    %1 : Index = 4
    %2 : toy.Tensor<affine.Shape<2>([4, 3]), F64> = toy.tile<%1>(%0)
    %3 : Nil = toy.print(%2)
    %_ : Nil = return(())
```

ToyToAffine can now lower everything.

The same pattern occurs at runtime. When the count depends on a function argument:

```
%main : Nil = function<Nil>() (%x: toy.Tensor<affine.Shape<1>([4]), F64>):
    %0 : Index = toy.nonzero_count(%x)
    %1 : toy.Tensor<affine.Shape<1>([3]), F64> = [7.0, 8.0, 9.0]
    %2 : toy.InferredShapeTensor<F64> = toy.tile<%0>(%1)
    %3 : Nil = toy.print(%2)
    %_ : Nil = return(())
```

Here `%0` depends on `%x` (a BlockArgument) — staging can't resolve it at compile time. Instead it builds a callback thunk: at runtime, when `%x` is known, the callback resolves `%0`, then runs the remaining compiler pipeline (ShapeInference → ToyToAffine → codegen) on the specialized IR.

**Today this requires the caller to pass separate `infer` and `lower` callbacks to `compile_staged`.** The Compiler design eliminates this by owning the full pipeline — staging, inference, lowering, and codegen are all passes in a single sequence.

## Context

PassManager is a thin sequential runner. Staging (`compile_staged`) takes ad-hoc `infer` and `lower` callbacks, fragmenting the pipeline across `cli.py`, `staging.py`, and `codegen.py`. The goal: unify into a `Compiler` that owns the full pipeline, with staging built in rather than bolted on.

Key design insight: **inference passes are regular lowering passes** — ShapeInference lowers `InferredShapeTensor` to `Tensor`, just as ToyToAffine lowers Toy ops to Affine ops. There is no special `infer` concept. Staging runs automatically before any pass when the IR contains non-ready values.

## Design

### Pass Types

```python
class Pass:  # Module → Module (existing, unchanged)
    def run(self, module: Module) -> Module: ...

class ExitPass(Generic[T]):  # Module → T (new)
    def run(self, module: Module) -> T: ...
```

`ExitPass[Executable]` wraps what `codegen.compile` does today. Entry passes (parsers) stay as plain functions — they don't interact with staging or compose with passes, so formalizing them adds no value.

### Compiler

```python
class Compiler(Generic[T]):
    passes: list[Pass]
    exit: ExitPass[T]

    def compile(self, module: Module) -> T:
        for i, pass_ in enumerate(self.passes):
            module = self._stage(module)
            module = pass_.run(module)
        module = self._stage(module)
        return self.exit.run(module)
```

### Staging Trigger

**Readiness is a property of the IR, not the pass.** `Value.ready` returns True when the value's type is ready and all its `__params__` are ready (recursively — a param is ready when it's a Constant or a Type). Staging checks the IR directly:

```python
def _needs_staging(self, module: Module) -> bool:
    for func in module.functions:
        for op in func.body.ops:
            if not op.ready:
                return True
    return False
```

No per-pass introspection needed. When the IR has non-ready ops, staging resolves what it can before the next pass runs.

In the TileOp example, `%4.ready` is False because its `count` param (`%3 = add_index(...)`) is not a Constant. Staging fires, resolves `%3` to `ConstantOp(4)`, and now all ops are ready. Subsequent passes (ShapeInference, ToyToAffine) see a fully ready IR.

This is correct even for optimization passes that could tolerate non-ready ops. Staging before optimization is not wasted — resolving comptime params early gives the optimizer more information. For example, if TileOp.count resolves to 1, the optimizer could eliminate the tile entirely. In practice, staging is a no-op when all ops are already ready (the common case for optimization), so the overhead is a single walk of the op graph.

### Staging Algorithm

`_stage` resolves all stage-0 comptime boundaries, then checks for remaining runtime-dependent boundaries:

1. Walk the IR, find ops where `__params__` fields are non-Constant Values.
2. For each such boundary, check if it's stage-0 evaluable (no BlockArgument dependencies). If so, JIT-compile the dependency subgraph using `self` (the full compiler) and replace with a ConstantOp.
3. Repeat until no more stage-0 boundaries remain.
4. If runtime-dependent (stage-1+) boundaries remain → build callback thunks and return early.
5. Otherwise → return the staged module for the next pass.

In the motivating example, step 2 resolves `add_index(2, 2)` → `ConstantOp(4)`. The dependency subgraph `{%1, %2, %3}` is wrapped in a mini-module and compiled through the full compiler pipeline (which is correct — the mini-module contains only builtin ops, so toy passes are no-ops).

### Callback Thunks

When runtime-dependent boundaries remain (stage-1+ example: `nonzero_count(%x)` where `%x` is a BlockArgument), build a callback:

```python
def _callback(*runtime_args):
    template = deepcopy(captured_module)
    resolve_boundaries_with_runtime_args(template, runtime_args)
    return compiler.compile(template)  # re-run FULL pipeline
```

The callback re-runs the **full compiler** on the specialized IR. ShapeInference runs naturally as a regular pass — no `infer` callback. Optimization passes re-run (idempotent, fast). This eliminates the entire `infer`/`lower` callback distinction.

For the runtime example: `nonzero_count(%x)` can't be resolved at compile time. At runtime with `%x = [1.0, 0.0, 3.0, 0.0]`, the callback resolves `%0` to `ConstantOp(2)`, then runs the full pipeline: staging (no-op — all ops now ready), ToyOptimize (no-op), ShapeInference (resolves tile shape to `[2, 3]`), ToyToAffine, AffineToLLVM, codegen.

### Toy Pipeline

```python
toy_compiler = Compiler(
    passes=[
        ToyOptimize(),
        ShapeInference(),
        ToyToAffine(),
        AffineToLLVMLowering(),
    ],
    exit=LLVMCodegen(),
)

# cli.py
def run(source: str, *, args=()):
    ast = parse_toy(source)
    ir = lower(ast)
    exe = toy_compiler.compile(ir)
    return exe.run(*args)
```

### Codegen as ExitPass

```python
class LLVMCodegen(ExitPass[Executable]):
    def run(self, module: Module) -> Executable:
        module = lower_builtin_to_llvm(module)
        ir, host_buffers = emit_llvm_ir(module)
        main = module.functions[0]
        ...
        return Executable(ir=ir, ...)
```

This is `codegen.compile()` wrapped as an ExitPass.

## Files to Modify

| File | Change |
|------|--------|
| `dgen/passes/pass_manager.py` | Rename to `compiler.py`, replace PassManager with Compiler |
| `dgen/codegen.py` | Extract `LLVMCodegen` ExitPass from `compile()` function |
| `dgen/staging.py` | Refactor: `compile_staged` → method on Compiler. Remove `infer`/`lower` callbacks. Staging uses `compiler.compile()` for JIT and callbacks. |
| `dgen/passes/__init__.py` | Update exports |
| `toy/cli.py` | Use Compiler instead of ad-hoc pipeline |
| `toy/test/test_staging.py` | Update to use Compiler API |
| `test/test_peano.py` | Update to use Compiler API |
| Any other callers of `compile_staged` / `compile_and_run_staged` / `PassManager` | Update |

## Implementation Steps

### 1. Add ExitPass protocol and Compiler class
- Create `dgen/compiler.py` with `ExitPass`, `Compiler`
- Compiler.compile implements staging-aware pass execution
- `_needs_staging` checks IR readiness directly (no pass introspection)

### 2. Extract LLVMCodegen as ExitPass
- Wrap `codegen.compile()` as `LLVMCodegen(ExitPass[Executable])`
- Keep `codegen.compile()` as a convenience function that delegates

### 3. Move staging into Compiler
- `Compiler._stage()` replaces `_resolve_all_comptime`
- Stage-0 resolution uses `self.compile()` for JIT sub-expressions
- Callback thunks call `self.compile()` on resolved template
- Remove `infer` and `lower` parameters from staging functions

### 4. Update Toy pipeline
- Define `toy_compiler` in `toy/compiler.py` (or `toy/cli.py`)
- `cli.run` uses `toy_compiler.compile(ir)`
- Remove ad-hoc `_lower`, `infer_shapes` callback plumbing

### 5. Update tests
- All tests using `compile_staged`/`compile_and_run_staged` switch to Compiler
- All tests using `PassManager` switch to Compiler (or Compiler without exit pass)

### 6. Remove dead code
- Delete `PassManager`
- Delete `compile_staged`, `compile_and_run_staged` free functions
- Clean up `staging.py` (staging logic moves into Compiler)

## Verification

```bash
# All existing tests must pass
pytest . -q

# Type checking
ruff format && ruff check --fix && ty check

# Verify end-to-end
python -m toy.cli toy/test/testdata/constant.toy
python -m toy.cli toy/test/testdata/tile_add_index.toy
```

## Design Decisions

1. **No-exit-pass case**: Compiler has an optional `run()` method that returns Module (no exit pass needed). `compile()` requires an exit pass. This covers the PassManager use case cleanly.

2. **`lower_builtin_to_llvm`**: Stays inside LLVMCodegen — it's always needed before LLVM IR emission and doesn't need to be visible in the pipeline.

3. **Sub-expression JIT**: Uses the full compiler (self). Correct by construction, simpler. Running optimization on tiny sub-expressions is cheap. The tail-only approach would be more efficient but adds complexity around determining the right tail position.

4. **Staging is unconditional**: Staging runs before every pass (and before the exit pass). When the IR is already fully ready — the common case — `_needs_staging` is a single walk that returns False immediately. This is simpler than having passes declare whether they need ready ops, and avoids the pass framework needing to know about staging at all.

5. **Why `infer` was unnecessary**: The `infer` callback in today's `_resolve_all_comptime` is called after each comptime resolution, but it doesn't affect boundary detection. `_unresolved_boundaries` operates on the param dependency graph — it checks whether params are Constants, not whether types are resolved. Resolving one boundary replaces an Op with a ConstantOp, directly changing the dependency graph for downstream boundaries. The resolution loop already handles cascading without `infer`. The only place `infer` actually matters is before lowering — but that's just "ShapeInference runs before ToyToAffine," which pipeline ordering guarantees.

## Open Questions

1. **Staging in sub-expression JIT**: When staging resolves a comptime field, it builds a mini-module and compiles it via the full compiler. This means the mini-module goes through ToyOptimize, ShapeInference, etc. — is that desirable or wasteful? It's correct but could be slow for deeply chained resolutions.
