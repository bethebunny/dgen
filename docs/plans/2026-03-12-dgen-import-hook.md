# Plan: .dgen Import Hook + .pyi Generation

## Context

Currently, `.dgen` dialect files must be compiled to `.py` files via `python -m dgen.gen` and checked in. This creates a two-step workflow: edit `.dgen`, run generator, commit both. The goal is to:

1. Add a Python import hook so `.dgen` files are loaded directly at import time (the `.py` files become unnecessary)
2. Change the `python -m dgen.gen` CLI to output a `.pyi` type stub instead of executable Python, used by static type checkers (ty/mypy/pyright) and IDEs

The `.pyi` files replace the `.py` files. The import hook handles runtime loading; the `.pyi` handles static analysis.

---

## Implementation Plan

### 1. New file: `dgen/gen/importer.py`

Implements `DgenFinder` (MetaPathFinder) and `DgenLoader` (Loader) and an `install()` function.

**Import resolution logic**: When loading `path/to/foo.dgen` (Python module `pkg.sub.foo`), for each import declaration (e.g. `from builtin import ...` or `import affine`), look for `builtin.dgen` or `affine.dgen` in the same directory as `foo.dgen`. If found, convert the found file path to a Python module name via `sys.path` lookup. Fall back to the hardcoded default `"builtin" → "dgen.dialects.builtin"` only if not found by relative search.

This eliminates the `-I` flag entirely—`affine` in `toy/dialects/toy.dgen` automatically resolves to `toy.dialects.affine` because `toy/dialects/affine.dgen` exists in the same directory.

**Key API**:
- `DgenFinder.find_spec(fullname, path, target)` — looks for `<name>.dgen` in `path` (or `sys.path` for top-level), returns `ModuleSpec` or `None`
- `DgenLoader.__init__(fullname, path)` — stores path; also stores `ast` and `import_map` after `exec_module` for use by the CLI
- `DgenLoader.exec_module(module)` — parses `.dgen`, resolves imports, calls `generate()`, `exec()`s the result; caches `self.ast` and `self.import_map`
- `_path_to_module(dgen_path)` — converts an absolute file path to a Python module name by iterating `sys.path`
- `install()` — inserts `DgenFinder` at front of `sys.meta_path` (idempotent)

### 2. Modify `dgen/__init__.py`

Add at the bottom:
```python
from dgen.gen.importer import install as _install_dgen_hook
_install_dgen_hook()
```

This installs the hook automatically whenever `dgen` is imported. No circular import risk: `dgen.gen.importer` → `dgen.gen.parser` → `dgen.gen.ast` (all stdlib only).

### 3. Modify `dgen/gen/python.py`

Add `as_stub: bool = False` parameter to `_generate()` and `generate()`. When `as_stub=True`:
- Replace `pass` (empty class bodies) with `...`
- Replace multi-line property body (the `return ...` line) with `...`

Also expose a `generate_pyi(ast, dialect_name, import_map)` convenience function that calls `generate(..., as_stub=True)`.

The existing `generate()` function (without `as_stub`) is unchanged—it's still used by the import hook.

### 4. Modify `dgen/gen/__main__.py`

Replace the current CLI with one that:
1. Calls `install()` to register the import hook
2. Determines the Python module name from the `.dgen` file path via `_path_to_module()`
3. Imports the module via `importlib.import_module()` (validates the hook works)
4. Retrieves `ast` and `import_map` from `module.__spec__.loader`
5. Calls `generate_pyi(ast, dialect_name, import_map)` and prints the result
6. Removes the `-I` option (no longer needed)

Error case: if `_path_to_module()` returns `None` (file not reachable via `sys.path`), raise a `ClickException` asking the user to run from the project root.

### 5. Delete generated `.py` files, generate `.pyi` files

Delete:
- `dgen/dialects/builtin.py`
- `dgen/dialects/llvm.py`
- `toy/dialects/affine.py`
- `toy/dialects/toy.py`

Generate replacements:
```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.pyi
python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.pyi
python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.pyi
python -m dgen.gen toy/dialects/toy.dgen > toy/dialects/toy.pyi
```

### 6. Update `CLAUDE.md`

Update the "Generated files" section: change `.py` → `.pyi`, remove `-I` flags from the regeneration commands, describe the new workflow.

---

## Critical Files

| File | Action |
|------|--------|
| `dgen/gen/importer.py` | **Create** |
| `dgen/__init__.py` | **Modify** (add install call) |
| `dgen/gen/python.py` | **Modify** (add `as_stub` param, `generate_pyi`) |
| `dgen/gen/__main__.py` | **Modify** (new CLI logic) |
| `dgen/dialects/builtin.py` | **Delete** |
| `dgen/dialects/llvm.py` | **Delete** |
| `toy/dialects/affine.py` | **Delete** |
| `toy/dialects/toy.py` | **Delete** |
| `dgen/dialects/builtin.pyi` | **Create** (generated) |
| `dgen/dialects/llvm.pyi` | **Create** (generated) |
| `toy/dialects/affine.pyi` | **Create** (generated) |
| `toy/dialects/toy.pyi` | **Create** (generated) |
| `CLAUDE.md` | **Modify** |

---

## Existing utilities to reuse

- `dgen/gen/parser.py`: `parse(source) -> DgenFile` — unchanged, used by `DgenLoader.exec_module`
- `dgen/gen/python.py`: `generate(ast, dialect_name, import_map) -> str` — unchanged, called by hook; extended with `as_stub` flag
- `dgen/gen/ast.py`: `ImportDecl`, `DgenFile` — used in `_resolve_imports`

---

## Tests to update

- `test/test_gen_python.py`: The existing `generate()` tests continue to pass unchanged. One test (`test_generate_trait`) asserts `"pass" in code`—this stays valid since `as_stub=False` by default. Add new tests for `generate_pyi()` asserting `"..."` instead of `"pass"`, and property stubs.
- `test/test_gen_python.py` has `test_generate_op_with_params` that passes an explicit `import_map`—this still works since `generate()` signature is unchanged.
- No changes needed to other test files: since `dgen/__init__.py` installs the hook, all imports of `dgen.dialects.*` work transparently via the hook.

---

## Verification

1. Run `pytest . -q` — all tests pass (except pre-existing llvmlite failures)
2. Run `python -m toy.cli toy/test/testdata/constant.toy` — end-to-end pipeline works
3. Run `python -m dgen.gen dgen/dialects/builtin.dgen` — outputs `.pyi` text
4. Run `python -c "from dgen.dialects.builtin import Index; print(Index())"` — works without `.py` files
5. Run `ruff format && ruff check --fix && ty check` — clean
