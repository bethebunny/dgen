# builtin.Tuple Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Tuple<types: list<Type>>` as a builtin type with Record layout, enabling heterogeneous product types.

**Architecture:** Tuple is a parameterized builtin type with a variadic `list<Type>` parameter. Its layout is a Record whose fields are numbered "0", "1", ... with layouts derived from each type. The code generator needs a new `Record` layout entry and handling for variadic-type-param → Record layout generation. The ASM parser's `_coerce_param` needs a fix to handle lists of Values for parameterized field types. `Type.__constant__` and `_type_from_dict` need list param support.

**Tech Stack:** Python, pytest, dgen code generator

---

### Task 1: Test Tuple type construction and layout

**Files:**
- Test: `test/test_tuple.py`

**Step 1: Write the failing test**

```python
"""Tests for builtin.Tuple type."""

from dgen import layout
from dgen.dialects.builtin import F64, Index, Nil, String, Tuple
from dgen.type import type_constant


def test_tuple_construction():
    """Tuple<[Index, String]> constructs with a list of types."""
    t = Tuple(types=[Index(), String()])
    assert len(t.types) == 2


def test_tuple_layout():
    """Tuple<[Index, String]> has Record layout with fields "0", "1"."""
    t = Tuple(types=[Index(), String()])
    expected = layout.Record([("0", layout.Int()), ("1", layout.String())])
    assert t.__layout__ == expected


def test_empty_tuple_layout():
    """Tuple<[]> has empty Record layout (zero bytes)."""
    t = Tuple(types=[])
    expected = layout.Record([])
    assert t.__layout__ == expected
    assert t.__layout__.byte_size == 0


def test_tuple_three_types():
    """Tuple<[Index, F64, Index]> layout has three fields."""
    t = Tuple(types=[Index(), F64(), Index()])
    expected = layout.Record([
        ("0", layout.Int()),
        ("1", layout.Float64()),
        ("2", layout.Int()),
    ])
    assert t.__layout__ == expected
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tuple.py -q`
Expected: ImportError — `Tuple` doesn't exist yet.

**Step 3: Add Tuple to builtin.dgen**

Add to `dgen/dialects/builtin.dgen` before the ops:

```
type Tuple<types: list<Type>>:
    layout Record
```

**Step 4: Add Record to \_LAYOUTS and handle Record codegen**

In `dgen/gen/python.py`:

1. Add `"Record": "layout.Record"` to `_LAYOUTS`.

2. In the `is_parametric_layout` branch (around line 272), handle the Record case. When layout is Record and the param is variadic (list<Type>), generate:

```python
    @property
    def __layout__(self) -> layout.Layout:
        return layout.Record([
            (str(i), dgen.type.type_constant(t).__layout__)
            for i, t in enumerate(self.types)
        ])
```

The current codegen for parametric layouts does:
```python
entry = _LAYOUTS[td.layout]  # "layout.Record"
args = [...]  # one arg per param
body.append(f"        return {entry}({', '.join(args)})")
```

This doesn't work for Record since Record takes `list[tuple[str, Layout]]`, not positional Layout args. Need to detect when layout is "Record" and a param is variadic+Type, then generate the list comprehension form.

**Step 5: Regenerate builtin.py**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

Verify the generated Tuple class looks correct.

**Step 6: Run test to verify it passes**

Run: `python -m pytest test/test_tuple.py -q`
Expected: PASS

**Step 7: Commit**

```
jj commit -m "add builtin.Tuple type with Record layout"
```

---

### Task 2: Test Tuple ASM round-trip

**Files:**
- Test: `test/test_tuple.py` (add tests)
- Modify: `dgen/asm/parser.py` (`_coerce_param`)
- Modify: `dgen/type.py` (`Type.__constant__`, `_type_from_dict`)

**Step 1: Write the failing test**

Add to `test/test_tuple.py`:

```python
from dgen import asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_tuple_type_asm_format():
    """Tuple<[Index, String]> formats as Tuple<[Index, String]>."""
    from dgen.asm.formatting import type_asm

    t = Tuple(types=[Index(), String()])
    assert type_asm(t) == "Tuple<[Index, String]>"


def test_empty_tuple_asm_format():
    """Tuple<[]> formats as Tuple<[]>."""
    from dgen.asm.formatting import type_asm

    t = Tuple(types=[])
    assert type_asm(t) == "Tuple<[]>"


def test_tuple_type_roundtrip():
    """Tuple type in IR round-trips through ASM."""
    ir = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %x : Tuple<[Index, String]> = [42, "hello"]
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tuple.py::test_tuple_type_roundtrip -q`
Expected: FAIL — parser can't coerce list of types for Tuple param.

**Step 3: Fix \_coerce\_param to handle list of Values**

In `dgen/asm/parser.py`, `_coerce_param` (line 294-300):

Change:
```python
if isinstance(value, list) and not field_type.__params__:
    return [_coerce_param(v, field_type) for v in value]
```

To:
```python
if isinstance(value, list):
    return [_coerce_param(v, field_type) for v in value]
```

The `not field_type.__params__` guard isn't needed — if the list elements are Values they pass through the `isinstance(value, Value)` check; if they're raw values they get wrapped as constants. No existing code path passes a raw list as a param to a parameterized type.

**Step 4: Fix Type.\_\_constant\_\_ for list params**

In `dgen/type.py`, `Type.__constant__` (line 108-113):

The current code does `param.__constant__.to_json()` for each param value. For a list param, `param` is a Python list, not a Value. Fix:

```python
@cached_property
def __constant__(self) -> Memory[TypeType]:
    data: dict[str, object] = {"tag": self.qualified_name}
    for name, param in self.parameters:
        if isinstance(param, list):
            data[name] = [p.__constant__.to_json() for p in param]
        else:
            data[name] = param.__constant__.to_json()
    return Memory.from_json(self.type, data)
```

**Step 5: Fix \_type\_from\_dict for list param values**

In `dgen/type.py`, `_type_from_dict` (line 51-72):

The current code handles dict values (nested types) and scalar values. Add list handling:

```python
if isinstance(param_value, list):
    kwargs[param_name] = [
        _type_from_dict(v) if isinstance(v, dict) else field_type().constant(v)
        for v in param_value
    ]
elif isinstance(param_value, dict):
    kwargs[param_name] = _type_from_dict(param_value)
else:
    kwargs[param_name] = field_type().constant(param_value)
```

**Step 6: Run tests to verify they pass**

Run: `python -m pytest test/test_tuple.py -q`
Expected: PASS

**Step 7: Run full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass.

**Step 8: Commit**

```
jj commit -m "Tuple ASM round-trip: fix coerce_param, Type.__constant__, _type_from_dict"
```

---

### Task 3: Test Tuple with type values

**Files:**
- Test: `test/test_tuple.py` (add tests)

**Step 1: Write the test**

Add to `test/test_tuple.py`:

```python
def test_tuple_type_values():
    """Tuple of type values: %types : Tuple<[Type, Type]> = [Index, String]."""
    from dgen.dialects.builtin import TypeTag
    from dgen.type import TypeType

    ir = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %types : Tuple<[TypeTag, TypeTag]> = [Index, String]
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

Note: This test validates that Tuple works with TypeTag values (type names as strings). The exact form depends on how type values serialize — may need adjustment based on how TypeTag constants work.

**Step 2: Run test**

Run: `python -m pytest test/test_tuple.py::test_tuple_type_values -q`

If it passes, great. If not, debug and fix.

**Step 3: Commit**

```
jj commit -m "test Tuple with type values"
```

---

### Task 4: Run lints and full test suite

**Step 1: Format and lint**

```bash
ruff format
ruff check --fix
ty check
```

**Step 2: Run full test suite**

```bash
python -m pytest . -q
```

**Step 3: Commit any fixes**

```
jj commit -m "lint fixes"
```
