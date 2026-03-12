# Syrupy Migration Plan

Migrate all IR pass-output tests to `ir_snapshot` (syrupy) or `assert_ir_equivalent`.

## Background

`IRSnapshotExtension` / `ir_snapshot` and `assert_ir_equivalent` both use the same Merkle-fingerprinting graph equivalence — they differ only in where the expected IR lives: a `.ir` snapshot file vs. an inline string.

The `docs/ir_testing.md` design doc explains why string equality is wrong for pass output tests: SSA names are positionally assigned and op ordering is arbitrary, so a semantically-irrelevant change produces a large textual diff. Graph equivalence eliminates that noise.

## Current State

### Category 1 — Round-trip tests: correct as-is, not candidates

These tests verify `parse(ir_text) → format → == ir_text`. The expected string *is* the input; there is nothing to snapshot. String equality is exactly right.

| File | Pattern |
|---|---|
| `toy/test/test_ir_parser.py` | `assert asm.format(module) == ir` (input = expected) |
| `toy/test/test_affine_roundtrip.py` | same |
| `toy/test/test_llvm_roundtrip.py` | same |
| `toy/test/test_mixed_roundtrip.py` | same |

One test in `test_affine_roundtrip.py` is an exception: `test_ssa_shape_through_lowering` does a partial string check (`assert "llvm.alloca<6>()" in result`) after lowering. That is testing a specific property of lowered output — acceptable as-is since it is intentionally partial.

### Category 2 — Single-op printer tests: correct as-is

`toy/test/test_toy_printer.py` tests like `test_constant_op`, `test_transpose_op`, etc. format a single op and check a one-line string. These test the formatter contract for individual ops; string equality is appropriate and stable (no multi-op SSA numbering). `test_full_module` formats a full module and string-compares; since it is testing the printer itself (not a pass), leaving it as string equality is acceptable.

### Category 3 — Pass output tests using `assert_ir_equivalent`: migrate to `ir_snapshot`

These already use graph equivalence, so they are not wrong. But the expected IR is inline, meaning updating expected output requires editing source. Migrate to syrupy snapshots for consistency and to move expected IR out of source.

| File | Tests |
|---|---|
| `toy/test/test_optimize.py` | all 5 tests |
| `toy/test/test_toy_to_affine.py` | all 9 tests |
| `test/test_pass.py` | `test_rewriter_eager_replace`, `test_pass_run_eliminates_double_transpose` |

### Category 4 — Pass output tests using raw string equality: must migrate

These format IR to text and compare via `== expected_string`. They are brittle to SSA renaming and op reordering — exactly the problem the testing doc describes.

| File | Tests | Notes |
|---|---|---|
| `toy/test/test_lowering.py` | all 8 tests | Toy source → IR, string compare |
| `toy/test/test_shape_inference.py` | 9 of 10 tests | `test_tile_with_computed_count` uses a deliberate substring check — keep as-is |
| `toy/test/test_affine_to_llvm.py` | all 7 tests | Highest priority: 100+ SSA-numbered values, extremely brittle |

## Migration Steps

For each test in categories 3 and 4:

1. Add `ir_snapshot` parameter to the test function signature.
2. Replace `assert result == expected` / `assert_ir_equivalent(result, expected_str)` with `assert module == ir_snapshot`, where `module` is the `Module` object (not a formatted string).
3. For tests that currently call `asm.format(module)` to get a string, keep the module object and pass it directly to `ir_snapshot`.
4. Delete the inline expected IR string.
5. Run `pytest --snapshot-update` to generate the initial `.ir` snapshot files.
6. Verify the generated snapshots look correct, then commit them.

### Special case: `test_shape_inference.py::test_tile_with_computed_count`

This test uses `assert "toy.InferredShapeTensor<F64> = toy.tile" in out` to confirm that shape inference *did not* resolve a shape. This is a deliberate negative property check, not a full output comparison. Keep it as a substring check.

### Special case: `test_affine_roundtrip.py::test_ssa_shape_through_lowering`

The round-trip portion (`assert asm.format(module) == ir`) stays as-is. The lowering check (`assert "llvm.alloca<6>()" in result`) also stays as-is.

## Migration Priority

1. **`toy/test/test_affine_to_llvm.py`** — highest. Contains the largest, most SSA-name-sensitive IR strings (100+ numbered values). One small change to the codegen renumbers half the file.
2. **`toy/test/test_lowering.py`** — medium. Clean toy IR but still brittle to SSA renaming.
3. **`toy/test/test_shape_inference.py`** — medium. IR strings are smaller but the pass output is still label-sensitive.
4. **`toy/test/test_optimize.py`**, **`toy/test/test_toy_to_affine.py`**, **`test/test_pass.py`** — already correct; migrate for consistency.

## After Migration

All pass-output tests use `ir_snapshot`. The round-trip tests and single-op printer tests remain as string equality. The distinction is clear: "does this text format round-trip?" uses string equality; "does this pass produce the right computation?" uses graph equivalence via snapshot.

Snapshot update policy (from `docs/ir_testing.md`):
- **Equivalence passes, text differs**: cosmetic only — CI may auto-update.
- **Equivalence fails**: real semantic change — requires human review.
