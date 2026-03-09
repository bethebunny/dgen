# builtin.Tuple Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Tuple<types: List<Type>>` as a builtin type with Record layout, enabling heterogeneous product types.

**Architecture:** Tuple is a parameterized builtin type with a `List<Type>` parameter — NOT a variadic `list<Type>` parameter. `List<Type>` is the uppercase builtin List type parameterized on Type. The code generator needs to handle this new param pattern: `_annotation_for_param` produces `list[Type]`, `__params__` entry uses `List`, and a `layout Record` + `List<Type>` param combination generates a Record-building `__layout__` property. The ASM parser, type serialization, and type reconstruction all need list param support.

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
type Tuple<types: List<Type>>:
    layout Record
```

**Step 4: Code gen changes in `dgen/gen/python.py`**

1. Add `"Record": "layout.Record"` to `_LAYOUTS` dict.

2. Handle `List<Type>` in `_annotation_for_param` — when param type is `List<Type>`, generate `list[Type]`:

```python
def _annotation_for_param(param: ParamDecl) -> str:
    if param.variadic:
        inner = _resolve_param_type_ref(param.type)
        return f"list[Value[{inner}]]"
    if param.type.name == "List" and param.type.args:
        inner = _resolve_param_type_ref(param.type.args[0])
        return f"list[Value[{inner}]]"
    if param.type.name == "Type":
        return "Value[dgen.TypeType]"
    return f"Value[{param.type.name}]"
```

3. Handle `List<Type>` in `_resolve_param_type_ref` — return `List` for the `__params__` entry:

```python
def _resolve_param_type_ref(ref: TypeRef) -> str:
    if ref.name == "Type":
        return "dgen.TypeType"
    if ref.name == "List":
        return "List"
    return ref.name
```

4. Handle `layout Record` with `List<Type>` param in the parametric layout codegen (around line 272). The current codegen passes each param as a positional arg to the layout constructor. For Record + List<Type>, we need to generate a list comprehension:

```python
    @property
    def __layout__(self) -> layout.Layout:
        return layout.Record([
            (str(i), dgen.type.type_constant(t).__layout__)
            for i, t in enumerate(self.types)
        ])
```

Detect this case: `td.layout == "Record"` and any param has `type.name == "List"` with `type.args[0].name == "Type"`. Generate the comprehension instead of the generic `entry(args)` form.

**Step 5: Regenerate builtin.py**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

Verify the generated Tuple class has `types: list[Value[dgen.TypeType]]`, `__params__ = (("types", List),)`, and the Record-building `__layout__` property.

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
from dgen.asm.formatting import type_asm
from dgen.asm.parser import parse_module
from toy.test.helpers import strip_prefix


def test_tuple_type_asm_format():
    """Tuple<[Index, String]> formats as Tuple<[Index, String]>."""
    t = Tuple(types=[Index(), String()])
    assert type_asm(t) == "Tuple<[Index, String]>"


def test_empty_tuple_asm_format():
    """Tuple<[]> formats as Tuple<[]>."""
    t = Tuple(types=[])
    assert type_asm(t) == "Tuple<[]>"


def test_tuple_constant_roundtrip():
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

Run: `python -m pytest test/test_tuple.py::test_tuple_constant_roundtrip -q`
Expected: FAIL — parser can't coerce list of types for Tuple param; `Type.__constant__` can't handle list params.

**Step 3: Fix `_coerce_param` to handle list of Values**

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

The `not field_type.__params__` guard prevented recursion for parameterized types. This isn't needed — list elements that are Values pass through unchanged, and raw values get wrapped as constants.

**Step 4: Fix `Type.__constant__` for list params**

In `dgen/type.py`, `Type.__constant__` (line 108-113):

The current code does `param.__constant__.to_json()` for each param. For a list param, `param` is a Python list, not a Value. Change:

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

**Step 5: Fix `_type_from_dict` for list param values**

In `dgen/type.py`, `_type_from_dict` (line 51-72):

The current code handles dict values (nested types) and scalar values. Add list handling before the dict check:

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

**Step 6: Fix `TypeType.__layout__` for list params**

In `dgen/type.py`, `TypeType.__layout__` (line 164-175), the property iterates params to build the layout Record. For a list param, `param.type` doesn't exist on a Python list. Need to handle this:

```python
@property
def __layout__(self) -> Record:
    resolved = type_constant(self.concrete)
    fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
    for name, param in resolved.parameters:
        if isinstance(param, list):
            # List of types → FatPointer of type value Records
            # Use generic string-based serialization for now
            fields.append((name, FatPointer(StringLayout())))
        else:
            fields.append((name, type_constant(param.type).__layout__))
    return Record(fields)
```

Note: The exact layout for serialized list-of-type params may need iteration. The key is that `to_json`/`from_json` roundtrip correctly — the Record layout must match the dict structure produced by `__constant__`.

**Step 7: Run tests to verify they pass**

Run: `python -m pytest test/test_tuple.py -q`
Expected: PASS

**Step 8: Run full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass.

**Step 9: Commit**

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
    """Tuple of type values: %types : Tuple<[TypeType<Index>, TypeType<String>]> = [...]."""
    ir = strip_prefix("""
        | %main : Nil = function<Nil>() ():
        |     %types : Tuple<[TypeType<Index>, TypeType<String>]> = [{"tag": "builtin.Index"}, {"tag": "builtin.String"}]
        |     %_ : Nil = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir
```

This validates that Tuple works with TypeType values. Uses `TypeType<Index>` because bare `Type` isn't yet a registered ASM type name (future work: register `Type` as ASM alias for `TypeType`, see TODO.md).

**Step 2: Run test**

Run: `python -m pytest test/test_tuple.py::test_tuple_type_values -q`

If it passes, great. If not, debug and fix.

**Step 3: Add TODO for bare `Type` in ASM**

Add to `TODO.md`: "Register `Type` as an ASM type name (alias for `TypeType`) so `Tuple<[Type, Type]>` works instead of `Tuple<[TypeType<Index>, TypeType<String>]>`"

**Step 4: Commit**

```
jj commit -m "test Tuple with type values; TODO for bare Type in ASM"
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
