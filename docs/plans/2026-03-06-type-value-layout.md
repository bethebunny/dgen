# Type Value Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give type values a concrete memory layout so they can flow through the JIT, enabling staging and dependent types.

**Architecture:** A type value's layout is `Record(tag, *param_layouts)` where the tag is a `TypeTag` (wrapping a dialect-qualified string name) and each parameter uses its type's `__layout__`. The `type_layout` property on `Type` is derived from `__params__` — no new fields needed. For deserialization, the format is self-describing: read the tag, look up the type constructor, walk params recursively.

**Tech Stack:** Python, dgen layout system, dgen codegen

**Design doc:** `docs/plans/2026-03-06-type-value-layout-design.md`

---

### Task 1: Add TypeTag to builtin.dgen and builtin.py

Add the `TypeTag` type to the builtin dialect. It wraps a `String` layout and holds a dialect-qualified type name.

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Modify: `dgen/dialects/builtin.py`
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing test**

Add to `toy/test/test_layout.py`:

```python
def test_type_tag_layout():
    """TypeTag wraps a String layout — 16 bytes."""
    tag = builtin.TypeTag()
    assert tag.__layout__.byte_size == 16
    assert isinstance(tag.__layout__, layout.String)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest toy/test/test_layout.py::test_type_tag_layout -v`
Expected: FAIL — `TypeTag` doesn't exist yet.

**Step 3: Add TypeTag to builtin.dgen**

Add after `type String`:

```dgen
type TypeTag:
    storage: String
```

**Step 4: Add TypeTag to builtin.py**

Add after the `String` class:

```python
@builtin.type("TypeTag")
@dataclass(frozen=True)
class TypeTag(Type):
    __layout__ = layout.String()
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest toy/test/test_layout.py::test_type_tag_layout -v`
Expected: PASS

**Step 6: Commit**

```bash
jj commit -m "feat: add TypeTag builtin type with String storage"
```

---

### Task 2: Add `type_layout` property to `Type` base class

The `type_layout` property computes a `Record` layout for the type value itself, derived from `__params__`.

**Files:**
- Modify: `dgen/type.py`
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

Add to `toy/test/test_layout.py`:

```python
from dgen.layout import Record, String as StringLayout


def test_type_layout_non_parametric():
    """Non-parametric type has layout Record([("tag", String)])."""
    ty = builtin.Index()
    tl = ty.type_layout
    assert isinstance(tl, Record)
    assert len(tl.fields) == 1
    assert tl.fields[0][0] == "tag"
    assert isinstance(tl.fields[0][1], StringLayout)


def test_type_layout_parametric_value_param():
    """Shape<rank: Index> includes rank as Int layout."""
    from toy.dialects import shape_constant
    from toy.dialects.affine import Shape

    shape = Shape(rank=shape_constant([2, 3]).type.n)
    tl = shape.type_layout
    assert isinstance(tl, Record)
    # tag + rank
    assert len(tl.fields) == 2
    assert tl.fields[0][0] == "tag"
    assert tl.fields[1][0] == "rank"
    assert isinstance(tl.fields[1][1], layout.Int)


def test_type_layout_parametric_type_param():
    """List<element_type: Type> inlines the element type's type_layout."""
    list_type = builtin.List(element_type=builtin.Index())
    tl = list_type.type_layout
    assert isinstance(tl, Record)
    # tag + element_type (which is itself a Record)
    assert len(tl.fields) == 2
    assert tl.fields[0][0] == "tag"
    assert tl.fields[1][0] == "element_type"
    inner = tl.fields[1][1]
    assert isinstance(inner, Record)
    # Inner is Index's type_layout: just a tag
    assert len(inner.fields) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest toy/test/test_layout.py::test_type_layout_non_parametric toy/test/test_layout.py::test_type_layout_parametric_value_param toy/test/test_layout.py::test_type_layout_parametric_type_param -v`
Expected: FAIL — `type_layout` doesn't exist.

**Step 3: Implement `type_layout` on `Type`**

In `dgen/type.py`, add to the `Type` class:

```python
@property
def type_layout(self) -> layout.Record:
    from .layout import Record, String as StringLayout

    fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
    for name, _ in self.__params__:
        val = getattr(self, name)
        if isinstance(val, Type):
            fields.append((name, val.type_layout))
        else:
            fields.append((name, val.__constant__.type.__layout__))
    return Record(fields)
```

Note: import `Record` and `String` inside the property to avoid circular imports (layout.py doesn't import type.py, but type.py imports from layout.py at the top level already — the issue is that `String` from layout would shadow builtin's `String` type if imported at module level).

**Step 4: Run tests to verify they pass**

Run: `python -m pytest toy/test/test_layout.py::test_type_layout_non_parametric toy/test/test_layout.py::test_type_layout_parametric_value_param toy/test/test_layout.py::test_type_layout_parametric_type_param -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat: add type_layout property to Type base class"
```

---

### Task 3: Type value serialization (to_json / from_json round-trip)

Pack a type value into a `Memory` buffer and read it back as a JSON-compatible dict.

**Files:**
- Modify: `dgen/type.py` (add `type_memory` and `type_from_memory` methods)
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

Add to `toy/test/test_layout.py`:

```python
def test_type_value_roundtrip_non_parametric():
    """Pack Index() as a type value and read it back."""
    from dgen.type import Memory

    ty = builtin.Index()
    mem = Memory(ty, layout=ty.type_layout)
    ty.type_pack(mem)
    result = mem.to_json()
    assert result == {"tag": "builtin.Index"}


def test_type_value_roundtrip_parametric():
    """Pack List<Index> as a type value and read it back."""
    from dgen.type import Memory

    ty = builtin.List(element_type=builtin.Index())
    mem = Memory(ty, layout=ty.type_layout)
    ty.type_pack(mem)
    result = mem.to_json()
    assert result == {
        "tag": "builtin.List",
        "element_type": {"tag": "builtin.Index"},
    }
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest toy/test/test_layout.py::test_type_value_roundtrip_non_parametric toy/test/test_layout.py::test_type_value_roundtrip_parametric -v`
Expected: FAIL

**Step 3: Implement `type_pack`**

In `dgen/type.py`, add to the `Type` class:

```python
def type_to_json(self) -> dict[str, object]:
    """Convert this type value to a JSON-compatible dict."""
    result: dict[str, object] = {"tag": f"{self.dialect.name}.{self._asm_name}"}
    for name, _ in self.__params__:
        val = getattr(self, name)
        if isinstance(val, Type):
            result[name] = val.type_to_json()
        else:
            result[name] = val.__constant__.to_json()
    return result
```

Then `type_pack` packs this into a Memory:

```python
def type_pack(self, mem: Memory) -> None:
    """Pack this type value into a Memory buffer."""
    self.type_layout.from_json(mem.buffer, 0, self.type_to_json(), mem.origins)
```

Update the tests to use `type_to_json()` directly for the simple case, and `type_pack` + `to_json()` for the round-trip through Memory.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest toy/test/test_layout.py::test_type_value_roundtrip_non_parametric toy/test/test_layout.py::test_type_value_roundtrip_parametric -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat: type value serialization via type_to_json and type_pack"
```

---

### Task 4: Type value deserialization

Read a type value back from a Memory buffer, reconstructing the Python Type object.

**Files:**
- Modify: `dgen/type.py` (add `Type.type_from_json` class method)
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

Add to `toy/test/test_layout.py`:

```python
def test_type_from_json_non_parametric():
    """Reconstruct Index() from its JSON representation."""
    from dgen.type import Type

    result = Type.type_from_json({"tag": "builtin.Index"})
    assert result == builtin.Index()


def test_type_from_json_parametric():
    """Reconstruct List<Index> from its JSON representation."""
    from dgen.type import Type

    result = Type.type_from_json({
        "tag": "builtin.List",
        "element_type": {"tag": "builtin.Index"},
    })
    assert result == builtin.List(element_type=builtin.Index())


def test_type_value_full_roundtrip():
    """Pack a type value to Memory and reconstruct it."""
    from dgen.type import Memory, Type

    ty = builtin.List(element_type=builtin.Index())
    mem = Memory(ty, layout=ty.type_layout)
    ty.type_pack(mem)
    json_val = mem.to_json()
    reconstructed = Type.type_from_json(json_val)
    assert reconstructed == ty
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest toy/test/test_layout.py::test_type_from_json_non_parametric toy/test/test_layout.py::test_type_from_json_parametric toy/test/test_layout.py::test_type_value_full_roundtrip -v`
Expected: FAIL

**Step 3: Implement `type_from_json`**

In `dgen/type.py`, add to the `Type` class:

```python
@classmethod
def type_from_json(cls, data: dict[str, object]) -> Type:
    """Reconstruct a Type from its JSON representation.

    Uses the dialect registry to look up the type constructor by its
    qualified tag, then recursively deserializes parameters.
    """
    from .dialect import Dialect

    assert isinstance(data, dict)
    tag = data["tag"]
    assert isinstance(tag, str)
    dialect_name, type_name = tag.split(".", 1)
    dialect = Dialect.get(dialect_name)
    type_cls = dialect.types[type_name]

    kwargs = {}
    for name, param_type in type_cls.__params__:
        raw = data[name]
        if param_type is Type or (isinstance(param_type, type) and issubclass(param_type, Type)):
            # Type-kinded param: recurse
            assert isinstance(raw, dict)
            kwargs[name] = cls.type_from_json(raw)
        else:
            # Value-kinded param: wrap as Constant
            kwargs[name] = param_type().constant(raw)
    return type_cls(**kwargs)
```

The key insight: for value params (like `rank: Index`), the JSON contains a plain value (e.g. `2`) which needs to be wrapped as a `Constant` via `param_type().constant(raw)`. For type params, it's a nested dict that we recurse on.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest toy/test/test_layout.py::test_type_from_json_non_parametric toy/test/test_layout.py::test_type_from_json_parametric toy/test/test_layout.py::test_type_value_full_roundtrip -v`
Expected: PASS

**Step 5: Run the full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass.

**Step 6: Commit**

```bash
jj commit -m "feat: type value deserialization via Type.type_from_json"
```

---

### Task 5: Type value Memory round-trip through raw buffers

Test the full path: type value -> Memory buffer -> raw bytes -> reconstruct type. This validates the self-describing format works end-to-end with actual packed memory.

**Files:**
- Test: `toy/test/test_layout.py`

**Step 1: Write the tests**

Add to `toy/test/test_layout.py`:

```python
def test_type_value_memory_roundtrip_with_value_params():
    """Round-trip a type with value params through Memory."""
    from toy.dialects import shape_constant
    from toy.dialects.affine import Shape
    from dgen.type import Memory, Type

    rank_const = shape_constant([2, 3]).type.n
    ty = Shape(rank=rank_const)
    mem = Memory(ty, layout=ty.type_layout)
    ty.type_pack(mem)
    json_val = mem.to_json()
    reconstructed = Type.type_from_json(json_val)
    assert reconstructed == ty


def test_type_value_memory_roundtrip_nested_type_params():
    """Round-trip List<List<F64>> through Memory."""
    from dgen.type import Memory, Type

    inner = builtin.List(element_type=builtin.F64())
    ty = builtin.List(element_type=inner)
    mem = Memory(ty, layout=ty.type_layout)
    ty.type_pack(mem)
    json_val = mem.to_json()
    reconstructed = Type.type_from_json(json_val)
    assert reconstructed == ty
```

**Step 2: Run tests**

Run: `python -m pytest toy/test/test_layout.py::test_type_value_memory_roundtrip_with_value_params toy/test/test_layout.py::test_type_value_memory_roundtrip_nested_type_params -v`
Expected: PASS (these should work with the implementation from tasks 2-4).

**Step 3: Run full test suite + linting**

Run: `python -m pytest . -q && ruff format && ruff check --fix`
Expected: All pass, clean formatting.

**Step 4: Commit**

```bash
jj commit -m "test: type value memory round-trip tests"
```
