# Type Values as First-Class Citizens — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Demonstrate type values as first-class SSA citizens — ops returning types, SSA values in type parameters, SSA values as op types, type values through JIT, and type constants with dict syntax.

**Architecture:** Three small changes to parser/formatter/staging, then 10 tests showcasing the full range of type-value capabilities. Dict literal syntax enables TypeType constants in ASM. SSA refs in type position create unresolved ops that staging resolves. TypeType's existing Record layout flows through JIT as a pointer.

**Tech Stack:** Python, pytest, llvmlite (JIT)

---

### Task 1: Dict literal syntax — parser

**Files:**
- Modify: `dgen/asm/parser.py:30-91` (add `{` case to `parse_expr`)

**Step 1: Write the failing test**

Create `test/test_type_values.py`:

```python
"""Tests for type values as first-class SSA citizens."""

import pytest

from dgen import Block, Constant, Dialect, TypeType, Value, asm
from dgen.asm.formatting import format_expr, type_asm
from dgen.asm.parser import IRParser, parse_expr, parse_module
from dgen.block import BlockArgument
from dgen.codegen import compile as compile_module
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Index, Nil
from dgen.module import ConstantOp, Function, Module
from dgen.type import Memory
from toy.test.helpers import strip_prefix


def test_parse_dict_literal():
    """parse_expr handles {key: value, ...} and returns a Python dict."""
    parser = IRParser('{"tag": "builtin.Index"}')
    result = parse_expr(parser)
    assert result == {"tag": "builtin.Index"}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_type_values.py::test_parse_dict_literal -xvs`
Expected: FAIL — `parse_expr` doesn't handle `{`

**Step 3: Write minimal implementation**

In `dgen/asm/parser.py`, add a `{` case to `parse_expr` after the `[` case (around line 53):

```python
    if c == "{":
        # Dict: {key: value, key: value, ...}
        parser.expect("{")
        parser.skip_whitespace()
        result = {}
        if parser.peek() != "}":
            key = parse_expr(parser)
            parser.skip_whitespace()
            parser.expect(":")
            parser.skip_whitespace()
            val = parse_expr(parser)
            assert isinstance(key, str)
            result[key] = val
            parser.skip_whitespace()
            while parser.peek() == ",":
                parser.expect(",")
                parser.skip_whitespace()
                key = parse_expr(parser)
                parser.skip_whitespace()
                parser.expect(":")
                parser.skip_whitespace()
                val = parse_expr(parser)
                assert isinstance(key, str)
                result[key] = val
                parser.skip_whitespace()
        parser.expect("}")
        return result
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_type_values.py::test_parse_dict_literal -xvs`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat: add dict literal parsing to parse_expr"
```

---

### Task 2: Dict literal syntax — formatter

**Files:**
- Modify: `dgen/asm/formatting.py:87-115` (add dict case to `format_expr`)

**Step 1: Write the failing test**

Add to `test/test_type_values.py`:

```python
def test_format_dict_literal():
    """format_expr handles dicts."""
    result = format_expr({"tag": "builtin.Index"})
    assert result == '{"tag": "builtin.Index"}'
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_type_values.py::test_format_dict_literal -xvs`
Expected: FAIL — falls through to `str(value)`

**Step 3: Write minimal implementation**

In `dgen/asm/formatting.py`, add a dict case to `format_expr` after the `list` case (after line 106):

```python
    if isinstance(value, dict):
        items = ", ".join(
            f'"{k}": {format_expr(v, tracker)}' for k, v in value.items()
        )
        return "{" + items + "}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_type_values.py::test_format_dict_literal -xvs`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat: add dict literal formatting to format_expr"
```

---

### Task 3: Dict constant syntax in `parse_op`

**Files:**
- Modify: `dgen/asm/parser.py:570-571` (add `{` to implicit constant triggers)

**Step 1: Write the failing test**

Add to `test/test_type_values.py`:

```python
def test_typetype_constant_asm_roundtrip():
    """TypeType constant with dict literal round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_type_values.py::test_typetype_constant_asm_roundtrip -xvs`
Expected: FAIL — `parse_op` doesn't recognize `{` as constant start

**Step 3: Write minimal implementation**

In `dgen/asm/parser.py`, line 571, change:
```python
        if self.peek() in "[-0123456789":
```
to:
```python
        if self.peek() in "{[-0123456789":
```

Also in `ConstantOp.__init__` in `dgen/module.py`, the `Memory.from_value` path needs to handle dict values. Check: `Memory.from_value` calls `Memory.from_json`, and `Record.from_json` accepts dicts. But `Memory.from_value` converts `str` to `bytes` to `list[int]` before calling `from_json`. Dicts should pass through fine since they're not `str` or `bytes`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_type_values.py::test_typetype_constant_asm_roundtrip -xvs`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat: parse dict literals as implicit constants in parse_op"
```

---

### Task 4: SSA ref in type position — parser

**Files:**
- Modify: `dgen/asm/parser.py:351-358` (`parse_type`)

**Step 1: Write the failing test**

Add to `test/test_type_values.py`:

```python
def test_ssa_ref_as_op_type():
    """SSA ref in type position: %x's type is unresolved Value, op not ready."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    t_op = ops[0]  # %t = TypeType constant
    x_op = ops[1]  # %x : %t = 42
    # %x's type is the SSA value %t, not a resolved Type
    assert x_op.type is t_op
    assert not x_op.ready
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_type_values.py::test_ssa_ref_as_op_type -xvs`
Expected: FAIL — `parse_type` asserts `isinstance(result, Type)` and fails on Value

**Step 3: Write minimal implementation**

In `dgen/asm/parser.py`, change `parse_type` (lines 351-358) from:

```python
    def parse_type(self) -> Type:
        """Parse a type via the registered type table, or () for Nil."""
        if self.peek() == "(":
            self.expect("()")
            return builtin.Nil()
        result = parse_expr(self)
        assert isinstance(result, Type)
        return result
```

to:

```python
    def parse_type(self) -> Type | Value:
        """Parse a type, or an SSA ref to a TypeType value."""
        if self.peek() == "(":
            self.expect("()")
            return builtin.Nil()
        result = parse_expr(self)
        if isinstance(result, Type):
            return result
        if isinstance(result, Value) and isinstance(result.type, TypeType):
            return result
        raise RuntimeError(f"Expected type, got {result}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_type_values.py::test_ssa_ref_as_op_type -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest . -q`
Expected: All existing tests still pass (the return type widened but all existing callers pass Type which is a Value subclass)

**Step 6: Commit**

```bash
jj commit -m "feat: allow SSA refs to TypeType values in type position"
```

---

### Task 5: SSA ref as op type — ASM round-trip

**Files:**
- Test: `test/test_type_values.py`

The formatter should already handle this: `op_asm` calls `format_expr(op.type, tracker)`, and `format_expr` handles `Value` by emitting `%name`. Since `op.type` is the SSA ref (a ConstantOp), it formats as `%t`.

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_ssa_ref_as_op_type_roundtrip():
    """SSA ref in type position round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_ssa_ref_as_op_type_roundtrip -xvs`
Expected: PASS (formatter already handles Value in type position)

If it fails, debug the formatter path. The key chain is: `op_asm` → `format_expr(op.type, tracker)` → `isinstance(value, Value)` branch → `%{tracker.track_name(value)}`.

Note: `format_expr` checks `isinstance(value, Type)` before `isinstance(value, Value)`. Since `op.type` is a ConstantOp (not a Type), it hits the `Constant` branch first: `isinstance(value, Constant) and not isinstance(value, Op)` — this is False for ConstantOp (which IS an Op). Then it falls through to `isinstance(value, Type)` — also False. Then `isinstance(value, Value)` — True, formats as `%t`. Good.

**Step 3: Commit**

```bash
jj commit -m "test: SSA ref as op type round-trips through ASM"
```

---

### Task 6: TypeType Memory round-trip

**Files:**
- Test: `test/test_type_values.py`

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_typetype_memory_roundtrip():
    """TypeType constant serializes to dict and round-trips through Memory."""
    ty = TypeType(concrete=Index())
    mem = ty.__constant__
    data = mem.to_json()
    assert data == {"tag": "builtin.Index"}
    # Round-trip: dict → Memory → dict
    mem2 = Memory.from_json(ty.type, data)
    assert mem2.to_json() == data
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_typetype_memory_roundtrip -xvs`
Expected: PASS (TypeType.__constant__ and Record layout already work)

**Step 3: Commit**

```bash
jj commit -m "test: TypeType Memory round-trip"
```

---

### Task 7: Parameterized TypeType constant

**Files:**
- Test: `test/test_type_values.py`

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_parameterized_typetype_constant_roundtrip():
    """TypeType for Array<Index, 4> round-trips with nested params in dict."""
    arr_ty = builtin.Array(
        element_type=Index(),
        n=Index().constant(4),
    )
    mem = arr_ty.__constant__
    data = mem.to_json()
    assert data == {"tag": "builtin.Array", "element_type": {"tag": "builtin.Index"}, "n": 4}

    # ASM round-trip with the parameterized TypeType
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Array<Index, 4>> = {"tag": "builtin.Array", "element_type": {"tag": "builtin.Index"}, "n": 4}
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_parameterized_typetype_constant_roundtrip -xvs`
Expected: PASS

**Step 3: Commit**

```bash
jj commit -m "test: parameterized TypeType constant round-trip"
```

---

### Task 8: Array with SSA dimension

**Files:**
- Test: `test/test_type_values.py`

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_array_with_ssa_dimension():
    """Array<Index, %n> — SSA value as type parameter, round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %n : Index = 4
        |     %arr : Array<Index, %n> = [1, 2, 3, 4]
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir

    # Verify the Array type's `n` param is the SSA value %n
    ops = module.functions[0].body.ops
    n_op = ops[0]
    arr_op = ops[1]
    assert isinstance(arr_op.type, builtin.Array)
    assert arr_op.type.n is n_op
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_array_with_ssa_dimension -xvs`
Expected: PASS (parser already handles %ref in type params via parse_expr)

**Step 3: Commit**

```bash
jj commit -m "test: Array with SSA dimension round-trip"
```

---

### Task 9: Type value through JIT identity

**Files:**
- Test: `test/test_type_values.py`

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_type_value_jit_identity():
    """TypeType value survives JIT identity function (passed as pointer)."""
    ty = TypeType(concrete=Index())
    mem = ty.__constant__

    # Build: main(t: TypeType<Index>) -> TypeType<Index> { return t }
    arg = BlockArgument(name="t", type=ty.type)
    func = FunctionOp(
        name="main",
        body=Block(ops=[builtin.ReturnOp(value=arg)], args=[arg]),
        type=Function(result=ty.type),
    )
    exe = compile_module(Module(functions=[func]))
    result = exe.run(mem)
    # TypeType is pointer-passed (Record layout), verify address survives
    assert result == mem.address
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_type_value_jit_identity -xvs`
Expected: PASS (Record layout → ptr in codegen, same as String/List)

**Step 3: Commit**

```bash
jj commit -m "test: TypeType value through JIT identity"
```

---

### Task 10: Type constant JIT return

**Files:**
- Test: `test/test_type_values.py`

**Step 1: Write the test**

Add to `test/test_type_values.py`:

```python
def test_type_constant_jit_return():
    """JIT function returns a TypeType constant, read back via Memory.from_raw."""
    ty = TypeType(concrete=Index())
    const = ConstantOp(value=ty.__constant__.to_json(), type=ty.type)
    ret = builtin.ReturnOp(value=const)
    func = FunctionOp(
        name="main",
        body=Block(ops=[const, ret], args=[]),
        type=Function(result=ty.type),
    )
    exe = compile_module(Module(functions=[func]))
    raw = exe.run()
    assert isinstance(raw, int)
    result = Memory.from_raw(ty.type, raw).to_json()
    assert result == {"tag": "builtin.Index"}
```

**Step 2: Run test**

Run: `python -m pytest test/test_type_values.py::test_type_constant_jit_return -xvs`
Expected: PASS

**Step 3: Commit**

```bash
jj commit -m "test: TypeType constant JIT return"
```

---

### Task 11: Staging resolves type value

**Files:**
- Modify: `dgen/staging.py:207-224` (`_unresolved_boundaries` — also check `op.type`)
- Test: `test/test_type_values.py`

**Step 1: Write the failing test**

Add to `test/test_type_values.py`:

```python
def test_staging_resolves_type_value():
    """Staging resolves a TypeType function param used as op type.

    main(%t: TypeType<Index>, %x: Index) -> Index:
        %y : %t = add_index(%x, %x)
        return(%y)

    The staging system resolves %t to Index, then codegen proceeds normally.
    """
    from dgen.staging import compile_and_run_staged

    identity = lambda m: m  # noqa: E731

    ir = strip_prefix("""
        | %main = function (%t : TypeType<Index>, %x : Index) -> Index:
        |     %y : %t = add_index(%x, %x)
        |     %_ : () = return(%y)
    """)
    module = parse_module(ir)

    result = compile_and_run_staged(
        module,
        infer=identity,
        lower=identity,
        args=[{"tag": "builtin.Index"}, 21],
    )
    assert result == 42
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_type_values.py::test_staging_resolves_type_value -xvs`
Expected: FAIL — `_unresolved_boundaries` doesn't check `op.type`, so staging doesn't resolve `%t`

**Step 3: Write minimal implementation**

In `dgen/staging.py`, modify `_unresolved_boundaries` to also check `op.type`:

```python
def _unresolved_boundaries(
    func: FunctionOp,
    stages: dict[int, int],
) -> list[tuple[int, dgen.Op, str, dgen.Value]]:
    """Find ops with unresolved __params__ or type refs, sorted by stage number."""
    boundaries: list[tuple[int, dgen.Op, str, dgen.Value]] = []
    for op in func.body.ops:
        for field_name, value in op.parameters:
            if isinstance(value, dgen.Value) and not isinstance(
                value, (Constant, dgen.Type)
            ):
                boundaries.append((stages.get(id(op), 0), op, field_name, value))
        # Also check op.type — if it's an unresolved SSA ref (Value, not Type)
        if isinstance(op.type, dgen.Value) and not isinstance(
            op.type, (Constant, dgen.Type)
        ):
            boundaries.append((stages.get(id(op), 0), op, "type", op.type))
    boundaries.sort(key=lambda t: t[0])
    return boundaries
```

Also in `_resolve_comptime_field`, when `field_name == "type"` and the resolved value is a dict with a `"tag"` key, we need to reconstruct the concrete Type from the tag and set `op.type` to the concrete Type (not a ConstantOp). Modify `_resolve_comptime_field` to handle this:

After the existing `setattr(op, field_name, const_op)` line, add handling for `field_name == "type"`:

```python
    if field_name == "type":
        # Resolve TypeType: unwrap the concrete type from the tag
        tag = result if isinstance(result, dict) else const_op.value.to_json()
        assert isinstance(tag, dict)
        op.type = _reconstruct_type(tag)
    else:
        setattr(op, field_name, const_op)
```

Add helper `_reconstruct_type`:

```python
def _reconstruct_type(data: dict) -> dgen.Type:
    """Reconstruct a Type from its serialized TypeType dict."""
    tag = data["tag"]
    dialect_name, type_name = tag.split(".")
    dialect = dgen.Dialect.get(dialect_name)
    cls = dialect.types[type_name]
    # Simple types with no params
    params = {k: v for k, v in data.items() if k != "tag"}
    if not params:
        return cls()
    # Parameterized types: wrap each param as a constant
    kwargs = {}
    for param_name, param_value in params.items():
        for field_name, field_type in cls.__params__:
            if field_name == param_name:
                if isinstance(param_value, dict):
                    # Nested type
                    kwargs[param_name] = _reconstruct_type(param_value)
                else:
                    kwargs[param_name] = field_type().constant(param_value)
                break
    return cls(**kwargs)
```

Also need to include `op.type` in stage computation. In `compute_stages`, the `_stage` function for ops computes stage from `__params__` and `__operands__`. When `op.type` is an unresolved Value, it should contribute `1 + stage(op.type)` (same as a `__params__` entry). Modify the `assert isinstance(value, dgen.Op)` branch:

```python
        assert isinstance(value, dgen.Op)
        parts: list[int] = []
        for pv in _field_values(value, value.__params__):
            parts.append(1 + _stage(pv))
        for ov in _field_values(value, value.__operands__):
            parts.append(_stage(ov))
        # Unresolved type ref counts as a param boundary
        if isinstance(value.type, dgen.Value) and not isinstance(
            value.type, (Constant, dgen.Type)
        ):
            parts.append(1 + _stage(value.type))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_type_values.py::test_staging_resolves_type_value -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass

**Step 6: Commit**

```bash
jj commit -m "feat: staging resolves SSA type refs on ops"
```

---

### Task 12: Final validation

**Step 1: Run full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass (existing + 10 new)

**Step 2: Run linting and formatting**

Run: `ruff format && ruff check --fix && ty check`
Expected: Clean

**Step 3: Commit any fixups**

```bash
jj commit -m "chore: lint/format fixes"
```
