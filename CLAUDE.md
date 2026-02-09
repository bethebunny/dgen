# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DGEN is a dialect generation tool inspired by LLVM's TableGen, designed as a replacement that addresses MLIR's performance, language-independence, and plugin limitations. It targets performant JIT compilation scenarios where MLIR's overhead is impractical.

Key design principles:
- Target language independent (no `extraClassDefinition`-style C++ hardcoding)
- Specifies default memory representations optimized for JIT (wire format = memory format, mmap/memcpy-friendly)
- Formal grammar specification (not implementation-defined like TableGen)
- Not bound to MLIR's data model, but provides an MLIR generation backend

## Current Milestone: Regex JIT (Milestone 1)

The first demonstration dialect is a regex engine targeting a subset of PCRE. The goal is to land on the pareto frontier: compile faster than PCRE-JIT, match faster than PCRE-no-JIT.

PCRE subset (Level 0): character literals, `.`, `*`/`+`/`?`, `|`, `(...)`.

Phases: DGEN grammar & dialect definitions → regex parser → NFA dialect & lowering (Thompson's construction) → NFA interpreter → benchmarks vs PCRE2 → JIT compilation.

Implementation language: Mojo.

## Repository Structure

- `regex/` — Regex dialect definitions (`.dgen` files)
- `docs/` — Milestone and design documents

## Version Control

This repo uses **jj** (Jujutsu) as its VCS frontend over git. Use `jj` commands rather than `git` for day-to-day operations (e.g., `jj st`, `jj log`, `jj new`, `jj commit`).
