# IR Equivalence Testing Implementation Plan

**Goal:** Implement graph equivalence checking for IR tests so pass output tests are robust to op ordering and SSA renaming changes. See `docs/ir_testing.md` for the full design.

**Tech Stack:** Python, pytest, `hashlib`

---

### Task 1: `dgen/ir_equiv.py` — `Fingerprinter`

Create `dgen/ir_equiv.py` with a `Fingerprinter` class that hashes the use-def graph into a stable content fingerprint.

**Files:**
- Create: `dgen/ir_equiv.py`
- Create: `test/test_ir_equiv.py`

#### Step 1: Write the failing tests

Create `test/test_ir_equiv.py`:

```python
"""Tests for IR graph equivalence checking."""

from dgen.dialects import builtin
from dgen.dialects.builtin import Nil
from dgen.asm.parser import parse_module
from dgen.ir_equiv import Fingerprinter
from dgen.module import ConstantOp
from dgen.block import Block, BlockArgument
from toy.test.helpers import strip_prefix


def test_identical_ops_same_fingerprint():
    """Two independently-constructed identical ops have the same fingerprint."""
    a = ConstantOp(value=42, type=builtin.Index())
    b = ConstantOp(value=42, type=builtin.Index())
    fp = Fingerprinter()
    assert fp.fingerprint(a) == fp.fingerprint(b)


def test_different_value_different_fingerprint():
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=2, type=builtin.Index())
    fp = Fingerprinter()
    assert fp.fingerprint(a) != fp.fingerprint(b)


def test_different_type_different_fingerprint():
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=1, type=builtin.F64())
    fp = Fingerprinter()
    assert fp.fingerprint(a) != fp.fingerprint(b)


def test_op_includes_operands():
    """Ops with different operands fingerprint differently."""
    from dgen.dialects.llvm import AddOp
    x = ConstantOp(value=1, type=builtin.Index())
    y = ConstantOp(value=2, type=builtin.Index())
    z = ConstantOp(value=3, type=builtin.Index())
    add_xy = AddOp(lhs=x, rhs=y)
    add_xz = AddOp(lhs=x, rhs=z)
    fp = Fingerprinter()
    assert fp.fingerprint(add_xy) != fp.fingerprint(add_xz)


def test_op_operand_order_matters():
    """add(%x, %y) != add(%y, %x) — operand order is structural."""
    from dgen.dialects.llvm import AddOp
    x = ConstantOp(value=1, type=builtin.Index())
    y = ConstantOp(value=2, type=builtin.Index())
    fp = Fingerprinter()
    assert fp.fingerprint(AddOp(lhs=x, rhs=y)) != fp.fingerprint(AddOp(lhs=y, rhs=x))


def test_block_arg_fingerprint_by_position():
    """Two block args at the same position with same type have same fingerprint."""
    arg_a = BlockArgument(type=builtin.Index())
    arg_b = BlockArgument(type=builtin.Index())
    fp = Fingerprinter()
    fp._arg_positions[id(arg_a)] = 0
    fp._arg_positions[id(arg_b)] = 0
    assert fp.fingerprint(arg_a) == fp.fingerprint(arg_b)


def test_block_arg_different_position_different_fingerprint():
    arg_a = BlockArgument(type=builtin.Index())
    arg_b = BlockArgument(type=builtin.Index())
    fp = Fingerprinter()
    fp._arg_positions[id(arg_a)] = 0
    fp._arg_positions[id(arg_b)] = 1
    assert fp.fingerprint(arg_a) != fp.fingerprint(arg_b)


def test_fingerprint_memoized():
    """fingerprint() is called once per object even in a diamond dependency."""
    from dgen.dialects.llvm import AddOp, MulOp
    x = ConstantOp(value=5, type=builtin.Index())
    add = AddOp(lhs=x, rhs=x)
    mul = MulOp(lhs=add, rhs=add)
    fp = Fingerprinter()
    fp.fingerprint(mul)
    # x is a shared dependency — fingerprinted once
    assert id(x) in fp._cache
```

#### Step 2: Implement `Fingerprinter`

Create `dgen/ir_equiv.py`:

```python
"""IR graph equivalence via Merkle fingerprinting.

Two IRs are equivalent if their use-def graphs are structurally isomorphic
— same computation up to op ordering and alpha-renaming.

See docs/ir_testing.md for design rationale.
"""

from __future__ import annotations

import hashlib
import struct

import dgen
from dgen.block import Block, BlockArgument
from dgen.type import Constant, Value


def _h(*parts: bytes) -> bytes:
    """Hash an ordered sequence of byte strings into a single digest."""
    h = hashlib.sha256()
    for part in parts:
        h.update(struct.pack(">I", len(part)))
        h.update(part)
    return h.digest()


class Fingerprinter:
    """Computes content-addressed fingerprints for IR values.

    Fingerprints are keyed on object identity and memoized. A single
    Fingerprinter instance should be used for one coherent IR graph.
    Block arguments must be registered (via register_block) before
    fingerprinting any op that uses them.
    """

    def __init__(self) -> None:
        self._cache: dict[int, bytes] = {}
        self._arg_positions: dict[int, int] = {}

    def register_block(self, block: Block) -> None:
        """Register block argument positions and recurse into nested blocks."""
        for i, arg in enumerate(block.args):
            self._arg_positions[id(arg)] = i
        for op in block.ops:
            for _, nested in op.blocks:
                self.register_block(nested)

    def fingerprint(self, value: Value) -> bytes:
        vid = id(value)
        if vid in self._cache:
            return self._cache[vid]
        result = self._compute(value)
        self._cache[vid] = result
        return result

    def _compute(self, value: Value) -> bytes:
        match value:
            case BlockArgument(type=t):
                pos = self._arg_positions[id(value)]
                return _h(b"arg", pos.to_bytes(4, "big"), self.fingerprint(t))
            case Constant():
                mem = value.__constant__
                return _h(b"constant", self.fingerprint(value.type), bytes(mem.buffer))
            case dgen.Op() as op:
                param_fps = b"".join(self.fingerprint(v) for _, v in op.parameters)
                operand_fps = b"".join(self.fingerprint(v) for _, v in op.operands)
                block_fps = b"".join(
                    self._fingerprint_block(b) for _, b in op.blocks
                )
                return _h(
                    op.dialect.name.encode(),
                    op.asm_name.encode(),
                    self.fingerprint(op.type),
                    param_fps,
                    operand_fps,
                    block_fps,
                )
            case _:
                raise TypeError(f"Cannot fingerprint {type(value).__name__}")

    def _fingerprint_block(self, block: Block) -> bytes:
        return _h(b"block", self.fingerprint(block.result))
```

**Important ordering constraint:** `Constant` must be matched before `dgen.Op` because `ConstantOp` inherits from both. The `match` statement uses `case Constant()` to dispatch on type, so declaration order in the `match` matters.

**Type fingerprinting** falls through to the `dgen.Op` branch for `Type` subclasses (types are Ops in the IR model — they have `dialect`, `asm_name`, and `parameters`). No special case is needed.

---

### Task 2: `graph_equivalent` and `structural_diff`

Add module-level equivalence check and a diff reporter to `dgen/ir_equiv.py`.

**Files:**
- Extend: `dgen/ir_equiv.py`
- Extend: `test/test_ir_equiv.py`

#### Step 1: Write the failing tests

Add to `test/test_ir_equiv.py`:

```python
from dgen.ir_equiv import graph_equivalent, structural_diff


def test_graph_equivalent_same_ir():
    ir = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    assert graph_equivalent(parse_module(ir), parse_module(ir))


def test_graph_equivalent_different_names():
    """Same computation, different SSA names -> equivalent."""
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %x : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %y : Nil = toy.print(%x)
        |     %_ : Nil = return(%y)
    """)
    assert graph_equivalent(parse_module(a), parse_module(b))


def test_graph_not_equivalent_different_values():
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %1 : Nil = toy.print(%0)
        |     %_ : Nil = return(%1)
    """)
    assert not graph_equivalent(parse_module(a), parse_module(b))


def test_structural_diff_returns_string():
    a = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %_ : Nil = return(())
    """)
    b = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        |     %_ : Nil = return(())
    """)
    diff = structural_diff(parse_module(a), parse_module(b))
    assert "actual" in diff.lower() or "expected" in diff.lower()
```

#### Step 2: Implement `graph_equivalent` and `structural_diff`

Add to `dgen/ir_equiv.py`:

```python
from dgen import asm
from dgen.module import Module


def _fingerprint_function(func: dgen.Op) -> bytes:
    fp = Fingerprinter()
    for _, block in func.blocks:
        fp.register_block(block)
    return fp._fingerprint_block(list(func.blocks)[0][1])


def graph_equivalent(actual: Module, expected: Module) -> bool:
    """Return True if actual and expected compute the same IR graph.

    Matches functions by name. Two functions are equivalent if their
    use-def graphs are structurally isomorphic — same ops, same operand
    structure, up to op ordering and SSA name assignment.
    """
    actual_fps = {f.name: _fingerprint_function(f) for f in actual.functions}
    expected_fps = {f.name: _fingerprint_function(f) for f in expected.functions}
    return actual_fps == expected_fps


def structural_diff(actual: Module, expected: Module) -> str:
    """Return a human-readable description of the difference between two IRs."""
    return (
        "IR equivalence check failed.\n\n"
        f"Actual:\n{asm.format(actual)}\n"
        f"Expected:\n{asm.format(expected)}"
    )
```

`_fingerprint_function` uses the first block (the function body). Functions with multiple blocks (e.g. `IfOp`) are handled transitively: `_fingerprint_block` calls `fp.fingerprint(block.result)`, which recurses into nested block ops via the `block_fps` term in `_compute`.

---

### Task 3: `dgen/testing.py` — `assert_ir_equivalent`

Expose a single test helper that parses the expected IR string and checks equivalence.

**Files:**
- Create: `dgen/testing.py`

```python
"""Test helpers for IR assertions."""

from __future__ import annotations

from dgen.asm.parser import parse_module
from dgen.ir_equiv import graph_equivalent, structural_diff
from dgen.module import Module


def assert_ir_equivalent(actual: Module, expected_ir: str) -> None:
    """Assert that actual is graph-equivalent to the IR described by expected_ir.

    Parses expected_ir and compares use-def graph structure. Passes if the
    two modules compute the same thing, regardless of op ordering or SSA names.
    On failure, shows a side-by-side of both formatted IRs.
    """
    expected = parse_module(expected_ir)
    if not graph_equivalent(actual, expected):
        raise AssertionError(structural_diff(actual, expected))
```

No tests needed for this shim — it is exercised by Task 4.

---

### Task 4: Migrate pass output tests

Replace `assert asm.format(result) == expected` with `assert_ir_equivalent(result, expected)` in all pass output tests. The expected IR strings remain unchanged — they are still readable input/output pairs.

**Files:**
- Modify: `toy/test/test_optimize.py`
- Modify: `toy/test/test_toy_to_affine.py`
- Modify: `test/test_pass.py` (the tests that check formatted output, not the Rewriter unit tests)

Pattern: replace

```python
result = asm.format(opt)
expected = strip_prefix("""...""")
assert result == expected
```

with

```python
assert_ir_equivalent(opt, strip_prefix("""..."""))
```

The `asm` import and `result` local can be dropped from any test that no longer needs them. The `strip_prefix` import and the expected IR strings remain exactly as-is.

---

### Verification

After all tasks:

```bash
pytest . -q
ruff format
ruff check --fix
ty check
```

All 110+ existing tests must pass. The new tests in `test/test_ir_equiv.py` add coverage for the fingerprinting logic independently of the pass tests.
