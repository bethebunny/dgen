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

The current working demonstration is a Toy dialect (inspired by the MLIR Toy tutorial), implemented as a full pipeline: source → AST → Toy IR → Affine IR → LLVM IR → JIT execution via llvmlite.

Pipeline stages (in `toy/cli.py`):
1. **Parse** `.toy` source → AST (`toy/parser/toy_parser.py`, `toy/parser/ast.py`)
2. **Lower** AST → Toy IR (`toy/parser/lowering.py`)
3. **Optimize** Toy IR → Toy IR (`toy/passes/optimize.py` — transpose folding, reshape elimination, dead code)
4. **Shape inference** (`toy/passes/shape_inference.py`)
5. **Lower** Toy → Affine dialect (`toy/passes/toy_to_affine.py`)
6. **Lower** Affine → LLVM dialect (`toy/passes/affine_to_llvm.py`)
7. **Codegen** → LLVM IR text → JIT via llvmlite (`dgen/codegen.py`)

Implementation language: **Python**.

## Repository Structure

- `dgen/` — Core IR framework (dialect-independent)
  - `op.py`, `type.py`, `value.py`, `block.py` — Core IR types
  - `dialect.py` — Dialect class with decorator-based op/type registration
  - `asm/` — IR text formatting (`formatting.py`) and parsing (`parser.py`)
  - `dialects/builtin.py` — Builtin dialect (Module, FuncOp, ConstantOp, ReturnOp, types)
  - `dialects/llvm.py` — LLVM dialect ops (alloca, gep, load, store, fadd, br, phi, etc.)
  - `codegen.py` — LLVM IR emission and JIT compilation via llvmlite
  - `layout.py` — Memory layout descriptors for types
- `toy/` — Toy dialect implementation
  - `dialects/toy.py` — Toy dialect ops (Constant, Transpose, Reshape, Mul, Add, Print, etc.)
  - `dialects/affine.py` — Affine dialect ops (ForOp, AllocOp, StoreOp, LoadOp, etc.)
  - `parser/` — Toy language frontend (lexer, parser, AST, lowering to IR)
  - `passes/` — Lowering and optimization passes
  - `test/` — All tests (pytest)
  - `test/testdata/` — `.toy` source files for CLI tests
  - `cli.py` — CLI entry point (compile and run `.toy` files)
  - `TODO.md` — Current task list
- `docs/` — Design documents (see `staging.md` for compile-time type staging model)
- `test/` — `dgen`-level tests

## Key Architecture Patterns

- **Dialect registration**: Decorators `@dialect.op("name")` and `@dialect.type("name")` register ops/types
- **Ops are dataclasses**: All ops inherit from `Op` (which inherits from `Value`). Fields annotated as `Value` are operands; fields annotated as `Block` are regions; other fields are compile-time attributes
- **ASM round-trip**: IR can be printed to text and parsed back; round-trip correctness is heavily tested
- **Generic constant op**: `builtin.ConstantOp` replaces per-dialect constant ops; the type annotation determines serialization and materialization
- **Staging model**: Types have compile-time and runtime faces; `constant` is the stage boundary (see `docs/staging.md`)

## Build & Test

```bash
# Run all tests
pytest . -q

# Run a specific test file
pytest toy/test/test_end_to_end.py -q

# Run CLI on a .toy file (must be run from repo root)
python -m toy.cli toy/test/testdata/constant.toy
```

Tests validate IR round-trips, pass correctness, and end-to-end JIT output. 110 tests, runs in ~1s.

Type checking, linting, formatting: Use `ruff` and `ty`.

```bash
ruff format
ruff check --fix
ty check
```

## Version Control

This repo uses **jj** (Jujutsu) as its VCS frontend over git. Use `jj` commands rather than `git` for day-to-day operations (e.g., `jj st`, `jj log`, `jj new`, `jj commit`, `jj describe`).
