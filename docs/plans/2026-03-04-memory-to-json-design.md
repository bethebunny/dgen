# Design: Generic Memory.to_json() / from_json()

## Problem

`Memory.to_python()` and `Memory._from_fat_pointer()` are fragile — they use type-specific branching (FatPointer vs Array vs scalar) with manual pointer chasing and special cases for strings, nested pointers, etc. This doesn't scale as new layout types are added.

## Solution

Replace them with two generic methods driven by walking the `Layout` tree:

- **`to_json()`** — recursively unpack buffer into JSON-compatible Python structures
- **`from_json()`** — recursively pack JSON-compatible structures into a buffer

## to_json() by Layout type

| Layout | Output |
|--------|--------|
| `Int` | `int` |
| `Float64` | `float` |
| `Byte` | `int` (single byte value) |
| `Void` | `None` |
| `Array(elem, n)` | `[to_json for each element]` |
| `FatPointer(pointee)` | Dereference `{ptr, length}` via ctypes, recursively unpack each element -> `list` |
| `Pointer(pointee)` | Dereference ptr via ctypes, unpack single element |

**Type-level specialization:** Types can post-process the generic JSON output. For example, `String` converts `[int, ...]` -> `str`. This keeps the Layout walk fully generic.

## from_json()

Mirror of `to_json()`. Walks the Layout tree, allocates origins for pointer types, packs recursively. Replaces both the scalar `layout.parse()` path and `_from_fat_pointer()`.

## from_value() changes

Becomes a thin wrapper: apply type-level pre-processing (e.g., String `str` -> list of byte ints), then call `from_json()`.

## Layout cleanup

`Bytes(n)` is removed in favor of `Array(Byte(), n)`.

## Caller updates

- `format_expr` (formatting.py) — call `to_json()` instead of `to_python()`
- `string_value` (module.py) — use `to_json()`
- `_raw_to_python` (staging.py) — use `to_json()`
- `from_asm` (type.py) — call `from_json()`
- Tests — update assertions

## Deletions

- `Memory.to_python()`
- `Memory._from_fat_pointer()`
- `Bytes` layout class
- FatPointer branch in `Memory.from_value()`

## Preserved

- `Memory.from_raw()` — unchanged (copies bytes from JIT result address)
- `Memory.address` — unchanged
- `origins` list — still used for GC lifetime rooting (not for unpacking)
- `Layout.parse()` — may survive as internal helper or fold into `from_json()`

---

# Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fragile `Memory.to_python()` / `_from_fat_pointer()` with generic Layout-tree-walking `to_json()` / `from_json()`.

**Architecture:** Add `to_json()` and `from_json()` as methods on `Layout` (not `Memory`), so each layout subclass implements its own serialization. `Memory.to_json()` delegates to `self.layout.to_json(self.buffer, 0)`. `Memory.from_json()` delegates to `self.layout.from_json(mem, value, 0)`. Type-level specialization (String) is handled in `Memory.to_json()` / `Memory.from_value()` via a `__to_json__` / `__from_json__` hook on `Type`.

**Tech Stack:** Python, ctypes, struct

---

### Task 1: Add `Layout.to_json()` for scalar layouts

**Files:**
- Modify: `dgen/layout.py`
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

Add to `toy/test/test_layout.py`:

```python
def test_int_to_json():
    from dgen.type import Memory
    from dgen.dialects.builtin import IndexType
    mem = Memory.from_value(IndexType(), 42)
    assert mem.to_json() == 42

def test_float_to_json():
    from dgen.type import Memory
    from dgen.dialects.builtin import F64Type
    mem = Memory.from_value(F64Type(), 3.14)
    assert mem.to_json() == 3.14

def test_byte_to_json():
    from dgen.layout import BYTE, Byte
    from struct import Struct
    buf = bytearray(1)
    Struct("B").pack_into(buf, 0, 65)
    assert BYTE.to_json(buf, 0) == 65
```

**Step 2: Run tests to verify they fail**

Run: `pytest toy/test/test_layout.py::test_int_to_json toy/test/test_layout.py::test_float_to_json toy/test/test_layout.py::test_byte_to_json -v`
Expected: FAIL — `to_json` not defined

**Step 3: Implement `to_json()` on Layout, Void, Byte, Int, Float64**

In `dgen/layout.py`:

```python
# On Layout base:
def to_json(self, buf: bytes | bytearray, offset: int) -> object:
    raise NotImplementedError

# On Void:
def to_json(self, buf: bytes | bytearray, offset: int) -> None:
    return None

# On Byte:
def to_json(self, buf: bytes | bytearray, offset: int) -> int:
    return self.struct.unpack_from(buf, offset)[0]

# On Int:
def to_json(self, buf: bytes | bytearray, offset: int) -> int:
    return self.struct.unpack_from(buf, offset)[0]

# On Float64:
def to_json(self, buf: bytes | bytearray, offset: int) -> float:
    return self.struct.unpack_from(buf, offset)[0]
```

Also add `Memory.to_json()` in `dgen/type.py`:

```python
def to_json(self) -> object:
    result = self.layout.to_json(self.buffer, 0)
    hook = getattr(self.type, "__to_json__", None)
    if hook is not None:
        result = hook(result)
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest toy/test/test_layout.py::test_int_to_json toy/test/test_layout.py::test_float_to_json toy/test/test_layout.py::test_byte_to_json -v`
Expected: PASS

**Step 5: Commit**

`jj describe -m "feat: add Layout.to_json() for scalar layouts"`

---

### Task 2: Add `Layout.to_json()` for Array, FatPointer, Pointer

**Files:**
- Modify: `dgen/layout.py`
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

```python
def test_array_to_json():
    from dgen.type import Memory
    from toy.dialects import shape_constant
    from toy.dialects.toy import TensorType
    ty = TensorType(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]

def test_fatpointer_to_json():
    from dgen.type import Memory
    from dgen.dialects.builtin import List, IndexType
    ty = List(element_type=IndexType())
    mem = Memory.from_value(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]

def test_string_to_json():
    from dgen.type import Memory
    from dgen.dialects.builtin import String
    mem = Memory.from_value(String(), "hello")
    assert mem.to_json() == "hello"
```

**Step 2: Run tests to verify they fail**

Run: `pytest toy/test/test_layout.py::test_array_to_json toy/test/test_layout.py::test_fatpointer_to_json toy/test/test_layout.py::test_string_to_json -v`
Expected: FAIL

**Step 3: Implement**

In `dgen/layout.py`:

```python
import ctypes

# On Array:
def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
    return [
        self.element.to_json(buf, offset + i * self.element.struct.size)
        for i in range(self.count)
    ]

# On FatPointer:
def to_json(self, buf: bytes | bytearray, offset: int) -> list[object]:
    ptr, length = self.struct.unpack_from(buf, offset)
    pointee = self.pointee
    ps = pointee.struct.size
    data = bytes((ctypes.c_char * (length * ps)).from_address(ptr))
    return [pointee.to_json(data, i * ps) for i in range(length)]

# On Pointer:
def to_json(self, buf: bytes | bytearray, offset: int) -> object:
    (ptr,) = self.struct.unpack_from(buf, offset)
    pointee = self.pointee
    data = bytes((ctypes.c_char * pointee.struct.size).from_address(ptr))
    return pointee.to_json(data, 0)
```

Add `__to_json__` hook on `String` type (in `dgen/dialects/builtin.py` or via monkey-patch in `dgen/module.py`):

```python
# On String type class:
@staticmethod
def __to_json__(value: object) -> str:
    assert isinstance(value, list)
    return bytes(value).decode("utf-8")
```

**Step 4: Run tests to verify they pass**

Run: `pytest toy/test/test_layout.py::test_array_to_json toy/test/test_layout.py::test_fatpointer_to_json toy/test/test_layout.py::test_string_to_json -v`
Expected: PASS

**Step 5: Commit**

`jj describe -m "feat: add Layout.to_json() for composite layouts + String hook"`

---

### Task 3: Add `Layout.from_json()` for all layouts

**Files:**
- Modify: `dgen/layout.py`, `dgen/type.py`
- Test: `toy/test/test_layout.py`

**Step 1: Write the failing tests**

```python
def test_int_from_json_roundtrip():
    from dgen.type import Memory
    from dgen.dialects.builtin import IndexType
    ty = IndexType()
    mem = Memory.from_json(ty, 42)
    assert mem.to_json() == 42

def test_list_from_json_roundtrip():
    from dgen.type import Memory
    from dgen.dialects.builtin import List, IndexType
    ty = List(element_type=IndexType())
    mem = Memory.from_json(ty, [10, 20, 30])
    assert mem.to_json() == [10, 20, 30]

def test_string_from_json_roundtrip():
    from dgen.type import Memory
    from dgen.dialects.builtin import String
    mem = Memory.from_json(String(), "hello")
    assert mem.to_json() == "hello"

def test_nested_list_from_json_roundtrip():
    from dgen.type import Memory
    from dgen.dialects.builtin import List, IndexType
    inner = List(element_type=IndexType())
    outer = List(element_type=inner)
    mem = Memory.from_json(outer, [[1, 2], [3, 4, 5]])
    assert mem.to_json() == [[1, 2], [3, 4, 5]]
```

**Step 2: Run tests to verify they fail**

Run: `pytest toy/test/test_layout.py::test_int_from_json_roundtrip toy/test/test_layout.py::test_list_from_json_roundtrip toy/test/test_layout.py::test_string_from_json_roundtrip toy/test/test_layout.py::test_nested_list_from_json_roundtrip -v`
Expected: FAIL

**Step 3: Implement**

In `dgen/layout.py`, add `from_json(buf, offset, value, origins)` on each layout:

```python
# On Layout base:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    raise NotImplementedError

# On Void:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    pass

# On Byte:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    assert isinstance(value, int)
    self.struct.pack_into(buf, offset, value)

# On Int:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    assert isinstance(value, int)
    self.struct.pack_into(buf, offset, value)

# On Float64:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    assert isinstance(value, (int, float))
    self.struct.pack_into(buf, offset, float(value))

# On Array:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    assert isinstance(value, list)
    es = self.element.struct.size
    for i, v in enumerate(value):
        self.element.from_json(buf, offset + i * es, v, origins)

# On FatPointer:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    assert isinstance(value, list)
    pointee = self.pointee
    ps = pointee.struct.size
    origin = bytearray(ps * len(value))
    origins.append(origin)
    for i, v in enumerate(value):
        pointee.from_json(origin, i * ps, v, origins)
    ptr = _bytearray_address(origin)
    self.struct.pack_into(buf, offset, ptr, len(value))

# On Pointer:
def from_json(self, buf: bytearray, offset: int, value: object, origins: list[bytearray]) -> None:
    pointee = self.pointee
    origin = bytearray(pointee.struct.size)
    origins.append(origin)
    pointee.from_json(origin, 0, value, origins)
    ptr = _bytearray_address(origin)
    self.struct.pack_into(buf, offset, ptr)
```

Note: `_bytearray_address` needs to be moved from `type.py` to `layout.py` (or imported).

Add `Memory.from_json()` in `dgen/type.py`:

```python
@classmethod
def from_json(cls, type: Type, value: object) -> Memory:
    hook = getattr(type, "__from_json__", None)
    if hook is not None:
        value = hook(value)
    mem = cls(type)
    type.__layout__.from_json(mem.buffer, 0, value, mem.origins)
    return mem
```

Add `__from_json__` on String:

```python
@staticmethod
def __from_json__(value: object) -> list[int]:
    if isinstance(value, str):
        return list(value.encode("utf-8"))
    return value
```

**Step 4: Run tests to verify they pass**

Run: `pytest toy/test/test_layout.py -v`
Expected: PASS

**Step 5: Commit**

`jj describe -m "feat: add Layout.from_json() for all layouts"`

---

### Task 4: Wire `from_json()` into `from_value()` and delete `_from_fat_pointer()`

**Files:**
- Modify: `dgen/type.py`
- Test: `pytest . -q` (all tests)

**Step 1: Replace `from_value()` body**

```python
@classmethod
def from_value(cls, type: Type, value: object) -> Memory:
    if isinstance(value, str):
        value = value.encode("utf-8")
    if isinstance(value, bytes):
        value = list(value)
    return cls.from_json(type, value)
```

**Step 2: Delete `_from_fat_pointer()`**

Remove the entire `_from_fat_pointer` classmethod.

**Step 3: Run all tests**

Run: `pytest . -q`
Expected: PASS (some tests may need adjustment — `from_value` previously accepted raw Python values that `parse()` coerced; now `from_json` handles it)

**Step 4: Commit**

`jj describe -m "refactor: from_value() delegates to from_json(), delete _from_fat_pointer"`

---

### Task 5: Wire `to_json()` into callers and delete `to_python()`

**Files:**
- Modify: `dgen/asm/formatting.py:98,105`, `dgen/module.py:74`, `dgen/staging.py:258-268`
- Test: `pytest . -q` (all tests)

**Step 1: Update `format_expr` in `dgen/asm/formatting.py`**

Change line 98:
```python
# Before:
return format_expr(value.__constant__.to_python(), tracker)
# After:
return format_expr(value.__constant__.to_json(), tracker)
```

Change line 105:
```python
# Before:
return format_expr(value.to_python(), tracker)
# After:
return format_expr(value.to_json(), tracker)
```

**Step 2: Update `string_value` in `dgen/module.py`**

```python
def string_value(v: Value[String]) -> str:
    result = v.__constant__.to_json()
    assert isinstance(result, str)
    return result
```

**Step 3: Update `_raw_to_python` in `dgen/staging.py`**

```python
def _raw_to_python(raw: object, ty: dgen.Type) -> object:
    layout = ty.__layout__
    if _ctype(layout) is ctypes.c_void_p:
        assert isinstance(raw, int)
        return Memory.from_raw(ty, raw).to_json()
    return raw
```

**Step 4: Delete `Memory.to_python()`**

Remove the entire method from `dgen/type.py`.

**Step 5: Update tests**

Replace all `.to_python()` calls in tests with `.to_json()`.

**Step 6: Run all tests**

Run: `pytest . -q`
Expected: PASS

**Step 7: Commit**

`jj describe -m "refactor: replace to_python() with to_json() everywhere"`

---

### Task 6: Remove `Bytes` layout and `Layout.parse()`

**Files:**
- Modify: `dgen/layout.py`, `dgen/gen/python.py`, `toy/test/test_layout.py`
- Test: `pytest . -q`

**Step 1: Delete `Bytes` class from `dgen/layout.py`**

**Step 2: Remove `parse()` methods from all layouts**

These are no longer called — `from_json()` replaces them.

**Step 3: Update `dgen/gen/python.py`**

Remove `"Bytes": "Bytes"` from the layout map and `"Bytes"` from the set.

**Step 4: Update `toy/test/test_layout.py`**

Remove `Bytes` import and `test_bytes_layout` test.

**Step 5: Run all tests**

Run: `pytest . -q`
Expected: PASS

**Step 6: Run formatting/linting**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

`jj describe -m "cleanup: remove Bytes layout and Layout.parse()"`
