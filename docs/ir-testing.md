# IR Equivalence Testing for Compiler Passes

## Problem

Snapshot-based IR tests are pedagogically valuable but brittle. Two sources of noise:

**Order instability.** IR is order-insensitive -- ops can appear in any valid topological order -- but text is ordered. A semantically-irrelevant change can alter the topological ordering, producing a large textual diff.

**Label instability.** SSA names are positionally assigned (`%0, %1, ...`). Inserting one op renumbers everything downstream.

## Approach: Graph Equivalence via Merkle Fingerprinting

Two IRs are **equivalent** if they are structurally isomorphic -- same computation, up to op ordering and alpha-renaming.

The fingerprint of an op is:

```
fingerprint(op) = hash(dialect, opcode, type, params, [fingerprint(v) for v in operands], [fingerprint_block(b) for b in blocks])
```

Two values are equivalent iff their root fingerprints match. This check is O(n).

### What Gets Fingerprinted

| Value kind | Fingerprint inputs |
|---|---|
| **Op** | dialect name, op name, result type, parameter fingerprints (declaration order), operand fingerprints (declaration order), block fingerprints (declaration order) |
| **Constant** | `"constant"`, type fingerprint, serialized value buffer |
| **BlockArgument** | `"arg"`, position within block, type fingerprint |
| **BlockParameter** | `"param"`, position within block, type fingerprint |
| **Type** | dialect name, type name, parameter fingerprints (declaration order) |
| **Block** | fingerprint of `block.result` |

### Design Choices

**No commutativity normalization.** `add(%a, %b)` and `add(%b, %a)` are distinct. Passes don't reorder operands arbitrarily, and commutativity annotations risk false equivalences.

**Phi nodes fingerprint directly.** `PhiOp` encodes predecessor block labels as compile-time String parameters and predecessor values as operands. No back-edge references, so no two-pass scheme is needed.

## Test Infrastructure

### Syrupy Snapshot Extension

`dgen/testing/syrupy.py` provides `IRSnapshotExtension` for pytest:

```python
# conftest.py
import pytest
from dgen.testing.syrupy import IRSnapshotExtension

@pytest.fixture
def ir_snapshot(snapshot):
    return snapshot.use_extension(IRSnapshotExtension)

# test_my_pass.py
def test_my_pass(ir_snapshot):
    result = my_pass(value)
    assert result == ir_snapshot
```

Snapshots are stored as `.ir` files. Comparison uses `graph_equivalent` (Merkle fingerprinting), not string equality. When the graph is equivalent but text differs, the snapshot update is cosmetic. When the graph differs, the failure output shows a semantic diff via `structural_diff`.

### Structural Diff

`structural_diff(actual, expected)` reports at the graph level:

- Ops present in `actual` with no matching fingerprint in `expected` -- reported as added
- Ops present in `expected` with no matching fingerprint in `actual` -- reported as removed
- Changed ops shown with both forms for comparison

CLI usage:

```bash
python -m dgen.ir.diff expected.ir actual.ir
python -m dgen.ir.diff expected.ir actual.ir -C 5
```

## Snapshot Update Policy

- **Equivalence passes, text differs**: cosmetic update only (op ordering or label renaming)
- **Equivalence fails**: real semantic change, requires review

## Key Files

| File | Role |
|------|------|
| `dgen/ir/equivalence.py` | `Fingerprinter`, `graph_equivalent` |
| `dgen/ir/diff.py` | `structural_diff`, `diff_values`, CLI |
| `dgen/testing/syrupy.py` | `IRSnapshotExtension` for pytest |
