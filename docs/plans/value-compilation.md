# Design: Value as the Compilation Interface

## Context

Module is the only IR construct that isn't a Value. Everything else — ops, types, blocks —
is in the dataflow graph. The compilation interface should be `compile(value) -> Constant`:
compile any Value, get back its resolved constant. Module becomes ASM-only.

This also surfaces a deeper issue: cross-function references use name-based placeholders
instead of actual Value references, so the use-def graph doesn't capture function
dependencies. Fixing that makes Module unnecessary in the pipeline.

## Core Insight: `compile(value) -> Constant`

The natural compilation primitive is:

```python
class Compiler:
    def compile(self, value: Value) -> Constant:
        # 1. If value is already a Constant, return it (short circuit)
        # 2. Resolve stage-0 boundaries (constant fold)
        # 3. Run passes on the dependency subgraph
        # 4. Lower + codegen + JIT execute
        # 5. Return result as Constant
```

This subsumes `_jit_evaluate` — it already does exactly this (build a mini-module, run
passes, JIT, return Python value). The difference: `_jit_evaluate` returns `object`
(Python value); `compile` returns `Constant` (typed Memory wrapper).

**Short circuit in `Compiler.compile`:** If the value is already a `Constant`, return it
immediately. No passes, no codegen. This is the natural base case.

## The Executable Question

Currently `Compiler[T]` is generic on exit pass return type, and the toy compiler is
`Compiler[Executable]`. `Executable` wraps LLVM IR + type metadata + a `run()` method for
CLI invocation.

**Tension:** `compile(value) -> Constant` says "evaluate this value." But a FunctionOp
with block args (parameters) can't be evaluated without supplying those args. It's a
function, not a value-to-be-computed.

**Three options for how Executable fits:**

### Option A: Executable is outside `compile`

`compile(value) -> Constant` is the core primitive. `Executable` is a separate concept —
it's the CLI's way of wrapping a compiled function for invocation.

```python
# Core primitive
constant = compiler.compile(some_expression)  # -> Constant

# CLI entry point (separate from compile)
exe = compiler.build_executable(function_op)  # -> Executable
result = exe.run(arg1, arg2)  # -> Memory
```

`build_executable` doesn't evaluate — it compiles the function body and returns a callable.
`compile` evaluates.

### Option B: Executable IS a Constant[Function]

A compiled function is a constant whose value is a function pointer. `Function` type gets
a layout (pointer-sized), and `Constant[Function]` holds the JIT'd address.

```python
func_constant = compiler.compile(function_op)  # -> Constant[Function]
# func_constant.value is Memory[Function] containing a function pointer
result = call(func_constant, arg1, arg2)       # -> Memory
```

This is elegant but requires:
- `Function` type to have a real layout (currently `data: Nil`)
- A `call` mechanism that marshals args and invokes via ctypes
- Executable's `run()` logic to move into a generic `call` function

### Option C: `compile` returns `Constant | Executable` depending on arity

```python
compiler.compile(zero_arg_expr)       # -> Constant (evaluated)
compiler.compile(function_with_args)  # -> Executable (callable)
```

This is pragmatic but loses type uniformity. The caller needs to know what it'll get back.

**Status:** Left open. The core `compile(value) -> Constant` primitive doesn't depend on
this choice.

## Two Layers of Change

### Layer 1: Functions as captured dependencies

Currently, cross-function calls use **name-based placeholders**:

```python
# toy/parser/lowering.py
callee_ref = dgen.Value(name=call.callee, type=builtin.Nil())  # dangling!
op = function.CallOp(callee=callee_ref, arguments=p, type=...)
```

`function.CallOp.callee: Value[Function]` is a compile-time parameter (in `__params__`),
so it IS in the use-def graph — but it points to a placeholder, not the actual FunctionOp.
Resolution happens by name lookup in ShapeInference, codegen, and staging.

**Fix:** Link `CallOp.callee` directly to the FunctionOp. Then:
- Use-def graph captures cross-function dependencies
- Compiling one value discovers all needed callees via `transitive_dependencies`
- ShapeInference doesn't need `_func_map`

**Structural question:** How does a FunctionOp reference another FunctionOp?

The callee FunctionOp should be an **op inside the caller's block** — it's defined in the
scope where it's used. This is function nesting (closures). `%helper` is defined inside
`%main`'s body block.

```
%main = function.function<Nil>() body():
    %helper = function.function<Float64>() body(%a):
        ...
    %result = function.call<%helper>([%x])
```

Alternatively, captures from an implicit top-level scope. But nesting is simpler — it
uses existing block scoping rules with no new concepts.

For recursion: the function appears as a `BlockParameter %self`, same as loops today.
For mutual recursion: needs shell-first construction (forward declarations).

### Layer 2: Module → Value in compilation interfaces

Once functions are linked via use-def, remove Module from the compilation pipeline.

**Current → New interfaces:**

| Current | New |
|---------|-----|
| `Compiler.compile(module: Module) -> T` | `Compiler.compile(value: Value) -> Constant` |
| `Compiler.run(module: Module) -> Module` | `Compiler.run(value: Value) -> Value` |
| `Pass.run(module, compiler) -> Module` | `Pass.run(value, compiler) -> Value` |
| `ExitPass.run(module: Module) -> T` | Removed (folded into compile) |
| `codegen.compile(module: Module) -> Executable` | `codegen.build_executable(func: FunctionOp) -> Executable` |
| `staging.compile_module(module, compiler)` | `staging.compile_value(value, compiler)` |
| `staging._jit_evaluate(subgraph, target, ...)` | `compiler.compile(target)` (subsumed) |

**Module survives** as ASM-only: serialization, parsing, test infrastructure.

## Open Questions

1. **Executable model:** How does `Executable` relate to `compile(value) -> Constant`?
   See options A/B/C above.

2. **Pass interface on Values:** Currently passes transform a Module (collection of
   functions). If passes operate on a single Value, how do multi-function passes work?
   E.g., ShapeInference needs to process callees before callers. Answer: the pass walks
   `transitive_dependencies` to find callees and processes them in topological order.

3. **`Compiler.run` return type:** If `compile(value) -> Constant` evaluates, what does
   `run` do? Currently `run` applies passes without the exit pass. Maybe `run` stays as
   "apply passes" and `compile` adds "then evaluate."

4. **Recursion in use-def:** If `%main` calls `%helper` which calls `%main`, the
   dependency graph has a cycle. This is handled by BlockParameter `%self` for
   self-recursion, but mutual recursion needs a new pattern.

## Design Docs to Update

When this design is implemented, the following existing design documents need revision:

### `docs/staging.md` — Compile-Time Types and Staging

- §2.7 (Staging Model): Currently describes stages in terms of a module-level view.
  Update to describe staging as operating on a single Value's dependency subgraph.
- `constant` as stage boundary (§2.3, §2.7): The `compile(value) -> Constant` primitive
  makes the relationship explicit — compiling a value IS the stage boundary crossing.
  `_jit_evaluate` is subsumed by `compile`. Update the conceptual framing.
- Add a section connecting `compile(value) -> Constant` to the quote/eval analogy:
  `compile` is `eval` — it takes a value expression and produces a constant.

### `docs/staged-computation.md` — Implementation Architecture

- Stage resolution loop: Currently described as iterating `_unresolved_boundaries` on a
  Module's functions. Update to describe resolution on a single Value's dependency graph.
- `_jit_evaluate`: Currently builds a mini-Module, wraps in FunctionOp, compiles through
  full pipeline. Update to describe as `compiler.compile(target_value)` — the mini-module
  wrapper is gone.
- Callback thunks: Currently receive a `Module` template and find functions by name.
  Update to describe receiving a single FunctionOp with callees reachable via use-def.

### `docs/pass-management.md` — Redesign PassManager → Compiler

- `Compiler.compile(module: Module) -> T`: Update signature to `compile(value: Value) -> Constant`.
- `ExitPass[T]` protocol: May be removed (folded into compile). Or may evolve —
  depends on Executable resolution.
- `Pass.run(module, compiler) -> Module`: Update to `run(value, compiler) -> Value`.
- Staging trigger (`_needs_staging`): Currently walks `module.functions`. Update to walk
  the target value's dependency graph.
- Callback thunks section: Same as staged-computation.md updates.

### `docs/passes.md` — Pass Framework Design

- Walk behavior: Currently described as walking "each block's result value" within a
  Module. The Module wrapper goes away — passes receive a Value and walk its dependencies.
- Verification (pre/post): Currently takes `Module`. Update to take `Value` (or
  `FunctionOp` as the common case).
- Examples (§Examples): Update `ShapeInference` example to show use-def traversal for
  cross-function inference instead of `_func_map`.
- Multi-function passes: Add a section explaining how passes that need cross-function
  context (like ShapeInference) discover callees via `transitive_dependencies` and process
  them in topological order.

### `docs/codegen.md` — Three-Phase LLVM IR Emission

- `emit_llvm_ir(module)`: Update to describe receiving functions directly (discovered
  via use-def from the entry function) rather than a Module.
- `compile(module) -> Executable` → `build_executable(func) -> Executable` (or whatever
  the Executable model resolves to).
