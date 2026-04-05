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
class Pass:  # Value → Value
    def run(self, value: Value, compiler: Compiler) -> Value: ...

class ExitPass(Generic[T]):  # Value → T
    def run(self, value: Value) -> T: ...
```

`Pass.run` operates on a single value (typically a FunctionOp). `ExitPass[Executable]` wraps codegen. Entry passes (parsers) stay as plain functions.

### Compiler

```python
class Compiler(Generic[T]):
    passes: list[Pass]
    exit: ExitPass[T]

    def compile(self, value: Value) -> T:
        """Full pipeline: staging → passes → exit."""
        return compile_value(value, self)

    def run(self, value: Value) -> T:
        """Run passes + exit (no staging)."""
        for i, p in enumerate(self.passes):
            continuation = Compiler(self.passes[i + 1:], self.exit)
            value = p.run(value, continuation)
        return self.exit.run(value)
```

Each pass receives a `continuation` Compiler representing the remaining pipeline — this allows passes (and the staging system) to invoke sub-compilations.

### Staging Trigger

**Readiness is a property of the IR, not the pass.** `Value.ready` returns True when the value's type is ready and all its `__params__` are ready (recursively — a param is ready when it's a Constant or a Type). Staging checks the IR directly by walking every op reachable from the root value (via `all_values`) and testing `op.ready`.

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
    template = deepcopy(captured_value)
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
        ToyToStructured(),
        ControlFlowToGoto(),
        NDBufferToMemory(),
        MemoryToLLVM(),
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
    def run(self, value: Value) -> Executable:
        # If value is a FunctionOp, use it as entry; otherwise wrap.
        entry = value if isinstance(value, FunctionOp) else _wrap(value)
        ir, host_buffers = emit_llvm_ir(entry)
        ...
        return Executable(ir=ir, ...)
```

This is `codegen.compile()` wrapped as an ExitPass.

## Implementation Status

This redesign is complete. The key files are:

| File | Role |
|------|------|
| `dgen/compiler.py` | `Compiler` class with staging-aware `compile()` and per-value `run()` |
| `dgen/passes/pass_.py` | `Pass` base class: `run(value, compiler) -> Value`, `@lowering_for`, `_dispatch_handlers` |
| `dgen/codegen.py` | `LLVMCodegen` ExitPass, `compile()` convenience function |
| `dgen/staging.py` | `compile_module()`, `resolve_stage0()`, callback thunks |
| `dgen/type.py` | `Value.replace_operand`, `Value.replace_uses_of` |
| `dgen/block.py` | `Block.replace_uses_of` |
| `dgen/graph.py` | `all_blocks`, `interior_blocks`, `inline_block` |

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

1. **Pass.run operates on Value, not Module**: `Compiler.run(value)` threads a single Value through every pass. This enables `compile(value) -> T` as the fundamental compilation primitive.

2. **Continuation compiler**: Each pass receives a Compiler with the remaining passes. This allows passes and the staging system to invoke sub-compilations (e.g., JIT-compiling a compile-time expression through the full pipeline).

3. **Sub-expression JIT**: Uses the full compiler (self). Correct by construction, simpler. Running optimization on tiny sub-expressions is cheap.

4. **Staging is unconditional**: `Compiler.compile` runs staging before the pass pipeline. When the IR is already fully ready — the common case — staging is a quick no-op.

5. **Why `infer` was unnecessary**: ShapeInference is a regular pass in the pipeline, not a special callback. Pipeline ordering guarantees it runs before lowering.

## Open Questions

1. **Staging in sub-expression JIT**: When staging resolves a comptime field, it builds a mini-module and compiles it via the full compiler. This means the mini-module goes through ToyOptimize, ShapeInference, etc. — is that desirable or wasteful? It's correct but could be slow for deeply chained resolutions.
