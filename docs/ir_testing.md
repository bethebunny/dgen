# IR Equivalence Testing for Compiler Passes

## Problem

Snapshot-based IR tests are pedagogically valuable: you can read an input/output pair and understand exactly what a compiler pass does. But they are brittle in practice because the textual representation of an IR is sensitive to semantically-irrelevant changes.

Two distinct sources of noise:

**Order instability.** IR is order-insensitive — ops can appear in any valid topological order — but the text format is ordered. A semantically-unrelated change can alter which topological ordering the formatter chooses, producing a large textual diff that represents zero semantic change.

**Label instability.** SSA value names are positionally assigned (`%0, %1, %2, ...` in traversal order). Inserting or deleting one op renumbers everything downstream, producing a diff that looks like a massive change when almost nothing changed semantically.

## Approach: Graph Equivalence via Merkle Fingerprinting

Two IRs are **equivalent** if they are structurally isomorphic — same computation, up to op ordering and alpha-renaming.

The Merkle fingerprint of an op is:

```
fingerprint(op) = hash(dialect, opcode, type, params, [fingerprint(v) for v in operands], [fingerprint_block(b) for b in blocks])
```

Two blocks are equivalent iff their root fingerprints match. This check is O(n).

### What gets fingerprinted

**Ops**: hash of dialect name, op name, result type, parameter fingerprints (in declaration order), operand fingerprints (in declaration order), and block fingerprints (in declaration order).

**Constants**: hash of `"constant"`, the type fingerprint, and the serialized value buffer (`Memory.buffer`).

**Block arguments**: hash of `"arg"`, the argument's position within the block, and its type fingerprint. Block args are inputs to the block and act as leaves in the use-def graph.

**Types**: hash of dialect name, type name, and parameter fingerprints (in declaration order). Types are Values in DGEN and may themselves have computed parameters; these are included in the type fingerprint recursively.

**Blocks**: the fingerprint of the block's root value (i.e. `fingerprint(block.result)`). Because side effects in DGEN are threaded through the use-def chain (each side-effecting op produces a `Nil` that the next op consumes), the entire op sequence of a block collapses into a single root fingerprint. There is no separate "unordered set of root ops" to handle.

### Commutative ops

The fingerprint uses operand declaration order unconditionally. Commutative op annotations are explicitly not supported:

- Passes do not reorder operands arbitrarily; `add(%a, %b)` and `add(%b, %a)` are a real structural difference worth flagging.
- Commutativity annotations introduce a class of bugs (mis-annotated ops produce false equivalences) for a case that does not arise in practice.

### Phi nodes

`PhiOp` in the LLVM dialect encodes predecessor block labels as compile-time String parameters and predecessor values as normal operands. There are no back-edge value references in DGEN's phi encoding, so phi nodes fingerprint directly without any two-pass scheme. The labels become part of the hash via the parameters list.

## Test Shape

```python
def assert_ir_equivalent(actual: Module, expected: Module) -> None:
    if graph_equivalent(actual, expected):
        return
    fail(structural_diff(actual, expected))
```

On failure, `structural_diff` reports at the graph level:

- Ops present in `actual` with no matching fingerprint in `expected` → reported as added
- Ops present in `expected` with no matching fingerprint in `actual` → reported as removed or changed
- For changed ops, both the actual and expected forms are formatted and shown side-by-side

This produces actionable, noise-free failure output. A failing test means something semantically real changed.

## Snapshot Update Policy

- **Equivalence passes, text differs**: the snapshot update is cosmetic — op ordering or label renaming only. CI may auto-update without human review.
- **Equivalence fails**: a real semantic change has occurred. Snapshot update requires human review.

This restores the invariant that approving a snapshot update means something real was reviewed.

## Implementation Scope

Three components:

1. **`dgen/ir/equivalence.py`** — `fingerprint(value)`, `graph_equivalent(a, b)`, `structural_diff(a, b)`
2. **Test helper** — `assert_ir_equivalent(actual, expected)` in `dgen/testing.py` or `conftest.py`
3. **Test migration** — existing pass output tests adopt the helper

The graph walk already exists (`block.ops`). Fingerprinting is straightforward recursive hashing with memoization on object identity. `structural_diff` can start as a side-by-side format of the two modules and be refined from there.
