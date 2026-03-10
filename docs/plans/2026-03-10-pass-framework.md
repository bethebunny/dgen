# Pass Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the `chain` op for side-effect ordering, change Block from list-based to graph-based (storing result value instead of op list), then build the pass infrastructure (`Pass`, `Rewriter`, `PassManager`).

**Architecture:** A builtin `chain(%lhs: $LHS, %rhs: $RHS) -> $LHS` op creates artificial data dependencies to keep side effects in the use-def graph. Block stores `result: Value` as the root; ops are derived by topological sort (a display concern). Passes mutate the graph in place via an eager `Rewriter`. The `Pass` base class in `dgen/passes/pass_.py` uses `@lowering_for` handler registration. The `PassManager` in `dgen/passes/pass_manager.py` orchestrates walk strategy and verification.

**Design doc:** `docs/passes.md`

**Tech Stack:** Python, pytest

---

### Task 1: Add `walk_ops` utility — topological sort from a root value

Before changing Block, build the graph-walking utility that derives an op list from a root value. This can be tested independently.

**Files:**
- Create: `dgen/graph.py`
- Create: `test/test_graph.py`

**Step 1: Write the failing test**

Create `test/test_graph.py`:

```python
"""Tests for use-def graph utilities."""

from dgen.dialects import builtin, llvm
from dgen.graph import walk_ops
from dgen.module import ConstantOp


def test_walk_ops_linear_chain():
    """Walk a simple linear dependency chain."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = ConstantOp(value=2, type=builtin.Index())
    c = llvm.AddOp(lhs=a, rhs=b)
    ops = walk_ops(c)
    assert ops[-1] is c
    assert set(ops) == {a, b, c}


def test_walk_ops_diamond():
    """Diamond dependency: a used by both b and c, both used by d."""
    a = ConstantOp(value=1, type=builtin.Index())
    b = llvm.AddOp(lhs=a, rhs=a)
    c = llvm.MulOp(lhs=a, rhs=a)
    d = llvm.AddOp(lhs=b, rhs=c)
    ops = walk_ops(d)
    assert ops[0] is a
    assert ops[-1] is d
    assert len(ops) == 4


def test_walk_ops_skips_block_args():
    """BlockArguments are not ops and should not appear in the result."""
    from dgen.block import BlockArgument

    arg = BlockArgument(type=builtin.Index())
    op = llvm.AddOp(lhs=arg, rhs=arg)
    ops = walk_ops(op)
    assert ops == [op]


def test_walk_ops_does_not_descend_into_blocks():
    """Ops nested inside another op's block are not included."""
    import dgen

    inner = ConstantOp(value=42, type=builtin.Index())
    func = builtin.FunctionOp(
        name="f",
        body=dgen.Block(ops=[inner], args=[]),
        result=builtin.Nil(),
    )
    ops = walk_ops(func)
    assert func in ops
    assert inner not in ops
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dgen.graph'`

**Step 3: Write minimal implementation**

Create `dgen/graph.py`:

```python
"""Use-def graph utilities."""

from __future__ import annotations

import dgen


def walk_ops(root: dgen.Value) -> list[dgen.Op]:
    """Walk the use-def graph from root, return ops in topological order.

    - Only includes Op instances (not plain Values or BlockArguments).
    - Does not descend into an op's nested blocks.
    - Dependencies appear before dependents.
    """
    visited: set[int] = set()
    order: list[dgen.Op] = []

    def visit(value: dgen.Value) -> None:
        vid = id(value)
        if vid in visited:
            return
        visited.add(vid)

        if not isinstance(value, dgen.Op):
            return

        # Visit dependencies first
        for _, operand in value.operands:
            visit(operand)
        for _, param in value.parameters:
            visit(param)

        order.append(value)

    visit(root)
    return order
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "add walk_ops: topological sort of use-def graph from a root value"
```

---

### Task 2: Add builtin `chain` op

The `chain` op creates artificial data dependencies to keep side effects in the use-def graph. `chain(%lhs: $LHS, %rhs: $RHS) -> $LHS` — returns `lhs`, but creates a dependency on `rhs`.

Example: to sequence two prints on different values:
```
%0: Index = 0
%1: Index = 1
%2: Nil = print(%0)
%3: Index = chain(%1, %2)
%4: Nil = print(%3)
%5: Nil = return(%4)
```

`%3` is `%1` at runtime, but depends on `%2` (the first print). So `print(%3)` happens after `print(%0)`.

**Files:**
- Modify: `dgen/dialects/builtin.dgen` (add chain op)
- Regenerate: `dgen/dialects/builtin.py`
- Test: `test/test_graph.py`

**Step 1: Write the failing test**

Add to `test/test_graph.py`. Test both that chain parses/formats correctly (ASM round-trip) and that `walk_ops` discovers chain dependencies:

```python
from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_chain_asm_round_trip():
    """chain op parses and formats correctly."""
    ir_text = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 0
        |     %1 : Index = 1
        |     %2 : Index = chain(%1, %0)
        |     %_ : Nil = return(())
    """)
    m = parse_module(ir_text)
    assert asm.format(m) == ir_text


def test_walk_ops_follows_chain_dependencies():
    """chain(lhs, rhs) creates dependency on rhs, walk_ops finds both."""
    ir_text = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 0
        |     %1 : Index = 1
        |     %2 : Index = chain(%1, %0)
        |     %_ : Nil = return(())
    """)
    m = parse_module(ir_text)
    func = m.functions[0]
    # walk_ops from the chain op should find both constants
    chain_op = func.body.ops[2]  # %2 = chain(...)
    ops = walk_ops(chain_op)
    assert len(ops) == 3  # two constants + chain
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_graph.py -k "chain" -v`
Expected: FAIL (chain op not recognized by parser)

**Step 3: Add chain to builtin.dgen**

Look at the existing `.dgen` format and add a `chain` op. The chain op is generic: `chain(%lhs, %rhs) -> Type`. Examine how other ops are defined in `builtin.dgen` to match the syntax.

Regenerate: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_graph.py -v`
Expected: PASS

**Step 5: Run full suite to check for regressions**

Run: `python -m pytest . -q`

**Step 6: Commit**

```bash
jj commit -m "add builtin chain op for side-effect ordering"
```

---

### Task 3: Thread chains through toy lowering

Update `toy/parser/lowering.py` to use `chain` when sequencing side effects. The key pattern: when multiple side-effecting ops appear in a function, thread them via `chain` so they're all reachable from the `return`.

**Files:**
- Modify: `toy/parser/lowering.py`
- Modify: test expectations in `toy/test/` (ASM output changes)

**Step 1: Analyze current lowering output**

Run: `python -m pytest toy/test/test_optimize.py -v` to establish baseline.

Look at the current IR patterns in test expectations. Identify all places where `return(())` should instead reference the chain of side effects.

**Step 2: Update lowering.py**

For a function body like:
```python
print(x)
print(y)
return
```

Currently produces:
```
%_ : Nil = toy.print(%x)
%_ : Nil = toy.print(%y)
%_ : Nil = return(())
```

Should produce:
```
%p1 : Nil = toy.print(%x)
%y_chained : ... = chain(%y, %p1)
%p2 : Nil = toy.print(%y_chained)
%_ : Nil = return(%p2)
```

Or if there's only one print:
```
%p1 : Nil = toy.print(%x)
%_ : Nil = return(%p1)
```

The exact threading depends on the lowering structure. Work through the cases iteratively.

**Step 3: Update test expectations**

All ASM round-trip tests that check exact IR output will need updated expected strings. Work through each test file.

**Step 4: Run full test suite**

Run: `python -m pytest . -q`
Fix failures iteratively.

**Step 5: Commit**

```bash
jj commit -m "thread chains through toy lowering for side-effect ordering"
```

---

### Task 4: Change Block to store result, derive ops via walk_ops

**Files:**
- Modify: `dgen/block.py`
- Modify: `dgen/asm/formatting.py` (if needed)
- Modify: `dgen/asm/parser.py` (if needed)
- Test: full suite must pass

Now that chains keep side effects reachable, Block can store just its result value. The `ops` property walks the use-def graph.

**Step 1: Change Block**

Modify `dgen/block.py`:

```python
@dataclass
class Block:
    result: dgen.Value
    args: list[BlockArgument] = field(default_factory=list)

    @cached_property
    def ops(self) -> list[dgen.Op]:
        from dgen.graph import walk_ops
        return walk_ops(self.result)
```

For backward compatibility during migration, accept `ops=` in the constructor and derive `result` from the last element:

```python
def __init__(self, *, result: dgen.Value | None = None,
             ops: list[dgen.Op] | None = None,
             args: list[BlockArgument] | None = None) -> None:
    if result is not None:
        self.result = result
    elif ops is not None and ops:
        self.result = ops[-1]
    else:
        raise ValueError("Block needs either result= or ops=")
    self.args = args or []
```

**Step 2: Handle ops mutation**

Many existing callers do `block.ops.append(...)` or `block.ops = [...]`. These will break since `ops` is now a derived property. Search for `block.ops` mutations and migrate them. The migration path:
- Code that builds a block: construct ops, then `Block(result=last_op)`
- Code that reads ops: `block.ops` still works (derived property)
- Code that mutates ops: must be rewritten to mutate the graph instead

**Step 3: Run full test suite iteratively**

Run: `python -m pytest . -q`
Fix failures one by one. Key areas:
- `dgen/asm/parser.py` — builds Blocks
- `dgen/asm/formatting.py` — reads block.ops
- All passes — build and manipulate Blocks
- `dgen/module.py` — `_walk_all_ops` iterates block.ops

**Step 4: Commit**

```bash
jj commit -m "change Block to store result value, derive ops via use-def walk"
```

---

### Task 5: Pass base class with @lowering_for

**Files:**
- Create: `dgen/passes/pass_.py`
- Create: `test/test_pass.py`

**Step 1: Write the failing test**

Create `test/test_pass.py`:

```python
"""Tests for the Pass base class."""

from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for


def test_lowering_for_registers_handler():
    class MyPass(Pass):
        op_domain: set[type] = set()
        op_range: set[type] = set()
        type_domain: set[type] = set()
        type_range: set[type] = set()
        allow_unregistered_ops = True

        @lowering_for(ConstantOp)
        def handle_constant(self, op, rewriter):
            return False

    assert ConstantOp in MyPass._handlers
    assert len(MyPass._handlers[ConstantOp]) == 1


def test_multiple_handlers_per_op_type():
    class MyPass(Pass):
        op_domain: set[type] = set()
        op_range: set[type] = set()
        type_domain: set[type] = set()
        type_range: set[type] = set()
        allow_unregistered_ops = True

        @lowering_for(ConstantOp)
        def handler_a(self, op, rewriter):
            return False

        @lowering_for(ConstantOp)
        def handler_b(self, op, rewriter):
            return True

    assert len(MyPass._handlers[ConstantOp]) == 2
    names = [h.__name__ for h in MyPass._handlers[ConstantOp]]
    assert names == ["handler_a", "handler_b"]
```

**Step 2: Run test, verify fail, implement, verify pass**

Create `dgen/passes/pass_.py` with `Pass` base class and `lowering_for` decorator as described in the design doc.

**Step 3: Commit**

```bash
jj commit -m "add Pass base class with @lowering_for handler registration"
```

---

### Task 6: Rewriter with eager replace_uses

**Files:**
- Modify: `dgen/passes/pass_.py`
- Modify: `test/test_pass.py`

**Step 1: Write the failing test**

Test that `replace_uses` eagerly updates all references, verified via ASM output:

```python
from dgen import asm
from dgen.asm.parser import parse_module
from dgen.passes.pass_ import Rewriter
from toy.test.helpers import strip_prefix


def test_rewriter_eager_replace():
    """replace_uses eagerly updates all referencing ops."""
    ir_text = strip_prefix("""
        | import llvm
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %2 : Index = llvm.add(%0, %0)
        |     %_ : Nil = return(())
    """)
    m = parse_module(ir_text)
    func = m.functions[0]
    old = func.body.ops[0]  # %0 = 1
    new = func.body.ops[1]  # %1 = 2

    rewriter = Rewriter(func.body)
    rewriter.replace_uses(old, new)

    result = asm.format(m)
    expected = strip_prefix("""
        | import llvm
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %2 : Index = llvm.add(%1, %1)
        |     %_ : Nil = return(())
    """)
    assert result == expected
```

**Step 2: Implement Rewriter**

```python
class Rewriter:
    """Manages in-place IR mutations during a pass."""

    def __init__(self, block: dgen.Block) -> None:
        self._block = block

    def replace_uses(self, old: dgen.Value, new: dgen.Value) -> None:
        """Eagerly replace all references to old with new in the block."""
        for op in self._block.ops:
            for name, operand in op.operands:
                if operand is old:
                    setattr(op, name, new)
            for name, param in op.parameters:
                if param is old:
                    setattr(op, name, new)
```

**Step 3: Run test, verify pass, commit**

```bash
jj commit -m "add Rewriter with eager replace_uses"
```

---

### Task 7: Pass.run — walk graph, dispatch handlers

**Files:**
- Modify: `dgen/passes/pass_.py`
- Modify: `test/test_pass.py`

**Step 1: Write the failing test**

A pass that eliminates double transposes, tested end-to-end with parse → pass → format:

```python
from dgen.passes.pass_ import Pass, Rewriter, lowering_for
from toy.dialects import toy


def test_pass_run_eliminates_double_transpose():
    """A pass that eliminates transpose(transpose(x)) -> x."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %2 : toy.Tensor<[2, 3], F64> = toy.transpose(%1)
        |     %_ : Nil = toy.print(%2)
        |     %_ : Nil = return(())
    """)

    class ElimTranspose(Pass):
        op_domain = {*toy.toy.ops.values(), ConstantOp, builtin.ReturnOp}
        op_range = {*toy.toy.ops.values(), ConstantOp, builtin.ReturnOp}
        type_domain = {toy.Tensor, builtin.Index, builtin.F64, builtin.Nil}
        type_range = {toy.Tensor, builtin.Index, builtin.F64, builtin.Nil}
        allow_unregistered_ops = True

        @lowering_for(toy.TransposeOp)
        def eliminate(self, op: toy.TransposeOp, rewriter: Rewriter) -> bool:
            if not isinstance(op.input, toy.TransposeOp):
                return False
            rewriter.replace_uses(op, op.input.input)
            return True

    m = parse_module(ir_text)
    result = ElimTranspose().run(m)
    formatted = asm.format(result)
    expected = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %_ : Nil = toy.print(%0)
        |     %_ : Nil = return(())
    """)
    assert formatted == expected
```

**Step 2: Implement Pass.run**

The pass manager walks ops (from `block.ops`, which is derived via topological sort from `block.result`), dispatches to handlers, and handlers call `rewriter.replace_uses`. Since replace_uses is eager, subsequent ops in the walk see updated operands.

```python
def run(self, module: Module) -> Module:
    for func in module.functions:
        self._run_block(func.body)
    return module

def _run_block(self, block: dgen.Block) -> None:
    rewriter = Rewriter(block)
    for op in block.ops:
        handlers = self._handlers.get(type(op), [])
        handled = False
        for handler in handlers:
            if handler(self, op, rewriter):
                handled = True
                break
        if not handled and not self.allow_unregistered_ops:
            raise TypeError(f"No handler for {type(op).__name__}")
        if not handled:
            for _, child_block in op.blocks:
                self._run_block(child_block)
```

**Step 3: Run test, verify pass, commit**

```bash
jj commit -m "add Pass.run: walk use-def graph, dispatch handlers"
```

---

### Task 8: PassManager with verification

**Files:**
- Create: `dgen/passes/pass_manager.py`
- Modify: `test/test_pass.py`

**Step 1: Implement PassManager**

```python
class PassManager:
    def __init__(self, passes: list[Pass], *, verify: bool = False) -> None:
        self._passes = passes
        self._verify = verify

    def run(self, module: Module) -> Module:
        for p in self._passes:
            if self._verify:
                p.verify_preconditions(module)
            module = p.run(module)
            if self._verify:
                p.verify_postconditions(module)
        return module
```

Add `verify_preconditions` / `verify_postconditions` to Pass (overridable, call super).

**Step 2: Write tests for verification (catches domain violations)**

```python
import pytest
from dgen.passes.pass_manager import PassManager


def test_pass_manager_verification_catches_range_violation():
    """Post-condition check detects ops outside the declared range."""
    ir_text = strip_prefix("""
        | import toy
        |
        | %main : Nil = function<Nil>() ():
        |     %0 : toy.Tensor<[2, 3], F64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], F64> = toy.transpose(%0)
        |     %_ : Nil = toy.print(%1)
        |     %_ : Nil = return(())
    """)

    class StrictPass(Pass):
        op_domain = {*toy.toy.ops.values(), ConstantOp, builtin.ReturnOp}
        op_range = {ConstantOp, builtin.ReturnOp}  # TransposeOp NOT in range
        type_domain = {toy.Tensor, builtin.F64, builtin.Nil}
        type_range = {toy.Tensor, builtin.F64, builtin.Nil}
        allow_unregistered_ops = True

    m = parse_module(ir_text)
    pm = PassManager([StrictPass()], verify=True)
    with pytest.raises(AssertionError):
        pm.run(m)
```

**Step 3: Commit**

```bash
jj commit -m "add PassManager with sequential execution and verification"
```

---

### Task 9: Port optimize.py to the pass infrastructure

**Files:**
- Modify: `toy/passes/optimize.py`
- Test: `toy/test/test_optimize.py` (existing, must still pass)

**Step 1: Run existing tests as baseline**

Run: `python -m pytest toy/test/test_optimize.py -v`

**Step 2: Rewrite optimize.py using Pass**

Port `eliminate_transpose`, `fold_constants`, `simplify_reshape` as `@lowering_for` handlers on `ToyOptimize(Pass)`. Keep DCE as a separate function for now.

**Step 3: Run tests, fix, commit**

```bash
python -m pytest toy/test/test_optimize.py -v
python -m pytest . -q
jj commit -m "port optimize.py to Pass infrastructure"
```

---

### Task 10: Full pipeline validation + lint

**Step 1: Run all tests**

```bash
python -m pytest . -q
```

**Step 2: Lint and type check**

```bash
ruff format
ruff check --fix
ty check
```

**Step 3: Commit fixes**

```bash
jj commit -m "fix lint/type issues from pass infrastructure integration"
```
