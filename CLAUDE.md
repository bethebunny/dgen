# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DGEN is a dialect generation tool inspired by LLVM's TableGen, designed as a replacement that addresses MLIR's performance, language-independence, and plugin limitations. It targets performant JIT compilation scenarios where MLIR's overhead is impractical.

Key design principles:
- Target language independent (no `extraClassDefinition`-style C++ hardcoding)
- Specifies default memory representations optimized for JIT (wire format = memory format, mmap/memcpy-friendly)
- Formal grammar specification (not implementation-defined like TableGen)
- Not bound to MLIR's data model, but provides an MLIR generation backend
- Ops always have exactly one output value
- Types may be parameterized on values; types may themselves be values

## Current Focus: Toy Dialect

The current working demonstration is a Toy dialect (inspired by the MLIR Toy tutorial), implemented as a full pipeline: source → AST → Toy IR → structured IR → LLVM IR → JIT execution via llvmlite.

Pipeline stages (in `toy/cli.py`):
1. **Parse** `.toy` source → AST (`toy/parser/toy_parser.py`, `toy/parser/ast.py`)
2. **Lower** AST → Toy IR (`toy/parser/lowering.py`)
3. **Optimize** Toy IR → Toy IR (`toy/passes/optimize.py` — transpose folding, reshape elimination, dead code)
4. **Shape inference** (`toy/passes/shape_inference.py`)
5. **Lower** Toy → structured (`toy/passes/toy_to_structured.py` — loops, memory, arithmetic)
6. **Lower** control flow → goto (`dgen/passes/control_flow_to_goto.py` — for/while → labels)
7. **Lower** ndbuffer → memory (`dgen/passes/ndbuffer_to_memory.py`)
8. **Lower** memory → LLVM (`dgen/passes/memory_to_llvm.py`)
9. **Codegen** → LLVM IR text → JIT via llvmlite (`dgen/codegen.py`, three-phase: see `docs/codegen.md`)

Implementation language: **Python**.

## Repository Structure

- `dgen/` — Core IR framework (dialect-independent)
  - `op.py`, `type.py`, `value.py`, `block.py` — Core IR types
  - `dialect.py` — Dialect class with decorator-based op/type registration
  - `asm/` — IR text formatting (`formatting.py`) and parsing (`parser.py`)
  - `dialects/builtin.pyi` — Builtin dialect type stubs (loaded at runtime from `builtin.dgen`)
  - `dialects/llvm.pyi` — LLVM dialect type stubs (loaded at runtime from `llvm.dgen`)
  - `dialects/ndbuffer.pyi` — NDBuffer dialect type stubs (loaded at runtime from `ndbuffer.dgen`)
  - `codegen.py` — LLVM IR emission and JIT compilation via llvmlite
  - `layout.py` — Memory layout descriptors for types
- `toy/` — Toy dialect implementation
  - `dialects/toy.pyi` — Toy dialect type stubs (loaded at runtime from `toy.dgen`)
  - `parser/` — Toy language frontend (lexer, parser, AST, lowering to IR)
  - `passes/` — Lowering and optimization passes
  - `test/` — All tests (pytest)
  - `test/testdata/` — `.toy` source files for CLI tests
  - `cli.py` — CLI entry point (compile and run `.toy` files)
- `docs/` — Design documents (see `staging.md` for compile-time type staging model)
- `test/` — `dgen`-level tests
- `TODO.md` — Current task list

## IR Design: Blocks and Use-Def

**Blocks are closed.** An op inside a block may only reference values defined in that same
block (local ops, block arguments, or captures). Values from an enclosing scope must be
declared as **captures** — the block explicitly lists every outer-scope value it references.
The parser and verifier enforce this.

**Three kinds of block inputs:**
- **`args`** — runtime values, receive phi nodes at entry (e.g. loop induction variables)
- **`parameters`** — compile-time values bound at block construction (e.g. `%self` for back-edges, `%exit` for loop exits)
- **`captures`** — outer-scope values referenced directly (no phi, no copy — just a declared dependency)

**Within a block, execution order is use-def order.** There is no implicit scheduling.
Ops with no use-def relationship between them may execute in any order. Side-effecting ops
must be chained via `ChainOp` to be reachable from `block.result` and to establish
ordering. `ChainOp(lhs=X, rhs=Y)` returns X's value with a use-def dependency on Y.

**All ops must be reachable from `block.result` via `transitive_dependencies`.** Unreachable ops are
dead. `block.ops` gives the complete, canonical op list for a block.
`dependencies` follows operands, parameters, types, and block captures (parent-scope
dependencies). It does NOT descend into nested block bodies — each block is its own
walk scope, with captures as boundaries.

See `docs/control-flow.md` and `docs/codegen.md` for the full design.

## Key Architecture Patterns

- **Dialect registration**: Decorators `@dialect.op("name")` and `@dialect.type("name")` register ops/types
- **Ops are dataclasses**: All ops inherit from `Op` (which inherits from `Value`). Fields annotated as `Value` are operands; fields annotated as `Block` are regions; other fields are compile-time attributes
- **Pass framework**: `Pass.run(value, compiler) -> Value` operates on values, not modules. Handlers registered via `@lowering_for(ValueType)` return `Value | None`. The framework calls `block.replace_uses_of(old, result)` automatically. `Compiler.run` handles Module iteration, dispatching per top-level op. See `docs/passes.md`.
- **Replacement cascade**: `Value.replace_uses_of(old, new)` updates operands/params/type and cascades to owned blocks. `Block.replace_uses_of(old, new)` sweeps block.values then updates metadata (captures, arg types, result). No separate Rewriter — replacement is encapsulated in instance methods.
- **ASM round-trip**: IR can be printed to text and parsed back; round-trip correctness is heavily tested
- **Generic constant op**: `builtin.ConstantOp` replaces per-dialect constant ops; the type annotation determines serialization and materialization
- **Staging model**: Types have compile-time and runtime faces; `constant` is the stage boundary (see `docs/staging.md`, `docs/staged-computation.md`)

## Build & Test

Install dependencies with `uv pip` before running tests. `uv pip` requires a virtual environment — create and activate one first if it doesn't exist:

```bash
# Create and activate a virtual environment (if one doesn't exist)
[ -d .venv ] || uv venv
source .venv/bin/activate

# Install all dependencies (including dev extras)
uv pip install -e ".[dev]"
```

```bash
# Run all tests
pytest . -q

# Run a specific test file
pytest toy/test/test_end_to_end.py -q

# Run CLI on a .toy file (must be run from repo root)
python -m toy.cli toy/test/testdata/constant.toy
```

Tests validate IR round-trips, pass correctness, and end-to-end JIT output. 110 tests, runs in ~1s.

Type checking, linting, formatting: Use `ruff` and `ty`. Always run `ruff format` directly (not `--check`).

```bash
ruff format
ruff check --fix
ty check
```

## Type checking

_Always_ strive to add correct type annotations! If the type annotations are incorrect, the code is substantially harder to reason about.
- Don't leave objects un-annotated
- Avoid `Any` or `object`. Use a protocol if necessary, prefer best-practices.
- Don't use `cast`, use `assert isinstance` or typeguards.
- Don't use `type: ignore`.
- Don't use `getattr` or `setattr` with a static string


## Generated files

`.dgen` files are loaded at **import time** via a Python import hook installed by `dgen/__init__.py`. No separate code-generation step is needed at runtime.

Files marked `# GENERATED by dgen from X.dgen — do not edit.` are `.pyi` type stubs generated by `python -m dgen.gen`. **Never hand-edit generated files.** If generated output needs to change, fix the `.dgen` source or the generator (`dgen/gen/`). Committed generated files must always be exactly the current output of the generator — if regenerating a file would produce a diff, the committed version is wrong.

To regenerate `.pyi` stubs (run from repo root):
```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.pyi
python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.pyi
python -m dgen.gen dgen/dialects/ndbuffer.dgen > dgen/dialects/ndbuffer.pyi
python -m dgen.gen toy/dialects/toy.dgen > toy/dialects/toy.pyi
```

## Debugging and Investigation

**Never write throwaway Python scripts.** All investigation, debugging, value inspection, and behavior exploration must go through pytest. Use the `debugging-with-pytest` skill — invoke it before writing any standalone `.py` file, any `python -c` command, any `if __name__ == "__main__"` block, or any script that imports project modules. No exceptions.

## Code style

- **No function-level imports**, even in tests. The only acceptable exception is breaking a genuine circular dependency between two modules (e.g. `type.py` ↔ `value.py`). If the import _can_ be top-level, it _must_ be top-level.
- **Never use `id(value)` in sets or as dict keys.** `id()` is an untyped weak reference — the object can be garbage-collected and the id reused. Use the value itself.
- Avoid `for ... in range()`
- Avoid `isinstance` specialization, this is generally design smell
- Don't special case. Refuse to add special cases that aren't explicitly called out in designs. Explicitly ask before adding any special cases to a design.

## JIT, Staging, and the Memory System

### Types and Memory

Every type declares a `__layout__: Layout` — a language-agnostic description of its binary memory representation. Layout instances (`Int`, `Float64`, `Span`, `Pointer`, etc.) own a `struct.Struct` for pack/unpack and know how to convert to/from JSON. Wire format equals memory format.

`Memory[T]` is a typed buffer for a value of type `T`. It can be initialized two ways:
- `Memory.from_json(type, value)` — from a JSON-compatible Python value
- `Memory.from_raw(type, address)` — from a raw pointer address (e.g. a JIT result)

All constants carry their data as `Memory` objects. This is how compile-time values cross the stage boundary into JIT-compiled code.

### Staging and Dependent Types

Op fields come in two kinds, accessible via properties:
- **`op.operands`** — `Value`-typed fields; runtime SSA values, pass through without adding a stage
- **`op.parameters`** — non-`Value` fields (e.g. `Shape`, `int`); compile-time, each one is a **stage boundary**

Stage numbers are computed as:
```
stage(op) = max(stage(v) for v in op.operands,
               stage(v) + 1 for v in op.parameters)
```
Constants are stage 0. Block arguments (function parameters) are stage 1.

Every op's result type (`op.type`) is itself a `Value[TypeType]` — a first-class SSA value in the same dataflow graph. Types can therefore depend on other ops (e.g. a tensor type whose shape is computed at runtime), and those type values participate in staging just like any other value. An op whose type is not yet resolved is simply at a higher stage than its type dependencies.

`builtin.ConstantOp` is the canonical stage boundary: it embeds a compile-time `Memory` value as a runtime SSA value. Unlike MLIR, which requires per-dialect constant ops (e.g. `arith.constant`, `toy.constant`), dgen's single `builtin.ConstantOp` works for every dialect — the type's `__layout__` drives serialization and materialization.

### Compile-Time Resolution and JIT Paths

`Compiler.compile(module)` runs staging before the pass pipeline. The staging engine finds unresolved `parameters` (fields that still hold a `Value` rather than a resolved `Constant`) and resolves them in ascending stage order:

1. **Stage-0 evaluable** — the dependency subgraph consists entirely of constants. The subgraph is extracted, JIT-compiled in isolation, executed immediately, and the result is patched back as a `ConstantOp`. This is the constant-folding / dependent-type resolution path.

2. **Runtime-dependent** — some `parameters` depend on block arguments. A callback thunk is built: stage-1 code calls back into the compiler at runtime, which JIT-compiles a stage-2 specialization against the actual runtime values.

This gives dgen out-of-the-box: constant folding, compile-time expression evaluation, and dependent types (types parameterized on runtime values, resolved just-in-time). See `docs/staging.md` and `docs/staged-computation.md` for details.

## Version Control

This repo uses **jj** (Jujutsu) as its VCS frontend over git. Use `jj` commands rather than `git` for day-to-day operations (e.g., `jj st`, `jj log`, `jj new`, `jj commit`, `jj describe`).

_Don't ever_ use `jj edit` or `jj abandon`, as these mutate the commit history. Use `jj new` and add an appropriate commit description. Note that `jj new` creates a new empty commit, so use `jj describe` to describe the current commit, or `jj commit -m` as sugar for `jj describe -m ... && jj new`. `jj squash` is permitted for resolving rebase conflicts, addressing review nits, or amending a commit with follow-up fixes.
