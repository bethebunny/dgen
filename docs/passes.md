# Pass Framework

## Pass.run

A `Pass` transforms a `Value` into a new `Value`. Handlers are registered via `@lowering_for` and return `Value | None`:

```python
from dgen.passes.pass_ import Pass, lowering_for
from dgen.dialects import control_flow, goto

class ControlFlowToGoto(Pass):
    allow_unregistered_ops = True

    @lowering_for(control_flow.ForOp)
    def lower_for(self, op: control_flow.ForOp) -> Value | None:
        # Build goto.RegionOp replacement...
        return goto.RegionOp(...)

    @lowering_for(control_flow.WhileOp)
    def lower_while(self, op: control_flow.WhileOp) -> Value | None:
        return goto.RegionOp(...)
```

When a handler returns a `Value`, the framework calls `block.replace_uses_of(old, result)` automatically. When it returns `None`, the op is unchanged.

### Handler dispatch

The `_PassMeta` metaclass collects `@lowering_for` handlers at class definition time. At runtime, `_dispatch_handlers(v)` looks up handlers by `type(v)` and tries them in order.

If `allow_unregistered_ops = False` and no handler matches an `Op`, the pass raises `TypeError`. This is useful for terminal lowering passes that must handle every op.

## Compiler[T]: Pipeline with Staging

`Compiler[T]` chains a list of `Pass` instances with an `ExitPass[T]` (typically `LLVMCodegen`):

```python
from dgen.passes.compiler import Compiler
from dgen.llvm.codegen import LLVMCodegen

compiler = Compiler(
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

exe = compiler.compile(ir)  # staging → passes → codegen
result = exe.run()
```

### Continuation model

Each pass receives a `continuation` Compiler representing the remaining pipeline. This enables sub-compilations (staging JIT-compiles subexpressions through the full pipeline):

```python
def run(self, value: Value) -> T:
    for i, p in enumerate(self.passes):
        continuation = Compiler(self.passes[i + 1:], self.exit)
        value = p.run(value, continuation)
    return self.exit.run(value)
```

### compile vs run

- `compiler.compile(value)` -- full pipeline: staging + passes + exit
- `compiler.run(value)` -- passes + exit only (no staging)

`compile` calls `compile_value` from `dgen/passes/staging.py`, which resolves stage-0 boundaries before entering the pass pipeline.

## Value.replace_uses_of / Block.replace_uses_of

Replacement is in-place mutation, not a separate Rewriter:

**Value.replace_uses_of(old, new):** Updates operands, parameters, type, and cascades into owned blocks.

**Block.replace_uses_of(old, new):** Sweeps `block.values` first (while captures still contain `old` so the walk finds all references), then updates block metadata (captures, arg types, parameter types, result).

```python
# The pass framework does this automatically:
result = handler(self, old_op)
if result is not None:
    block.replace_uses_of(old_op, result)
```

## Walk Order

Within `Pass._lower_block`, ops are visited in topological order (from `block.values`). For each op:

1. Dispatch handlers
2. Recurse into the effective op's nested blocks (the replacement if one was returned, otherwise the original) **before** processing downstream uses
3. If a replacement was returned, call `block.replace_uses_of`

This ensures handler-created blocks are lowered before they're referenced elsewhere.

## Verification

Passes can declare pre/post-conditions. The base `Pass` provides:

| Check | What it verifies |
|-------|-----------------|
| `verify_all_ready` | Every op has resolved parameters |
| `verify_dag` | No cycles in the use-def graph |
| `verify_closed_blocks` | Closed-block invariant holds; unique ownership |
| `verify_constraints` | Trait constraints on ops are satisfied |

Verification runs when `verify_passes` context var is `True` (off by default for performance):

```python
from dgen.passes.compiler import verify_passes
token = verify_passes.set(True)
# ... run pipeline ...
verify_passes.reset(token)
```

Preconditions run all four checks. Postconditions skip `verify_all_ready` and `verify_constraints` (the pass may have introduced unresolved ops that the next pass will handle).

## Staging: resolve_stage0

`resolve_stage0` runs before the pass pipeline via `Compiler.compile`. It iteratively finds the lowest-stage unresolved `__params__` boundary that is stage-0 evaluable (no `BlockArgument` dependencies), JIT-compiles it in isolation, and patches the result as a `ConstantOp`. See [values-and-types.md](values-and-types.md) for the full staging model.

> **Not yet implemented:** `forward_declare` / `link` for irreducible CFGs. Currently all control flow must be reducible (structured loops lowered to goto regions with `%self` back-edges).

## Key Files

| File | Role |
|------|------|
| `dgen/passes/pass_.py` | `Pass` base class, `@lowering_for`, `_PassMeta` |
| `dgen/passes/compiler.py` | `Compiler[T]`, `ExitPass` protocol, `verify_passes` |
| `dgen/passes/staging.py` | `compile_value`, `resolve_stage0`, callback thunks |
| `dgen/ir/verification.py` | `verify_closed_blocks`, `verify_dag`, `verify_all_ready`, `verify_constraints` |
