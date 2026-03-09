# Remove TypeType.concrete Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the `concrete` parameter from `TypeType`, making it an unparameterized type meaning "any type as a value." Register `Type` as an ASM alias.

**Architecture:** `TypeType.concrete` is purely static type information — the runtime `TypeValue` layout is self-describing via tags and doesn't need it. Removing `concrete` simplifies TypeType to a singleton-like type and lets ASM use bare `Type` instead of `TypeType<ConcreteType>`.

**Tech Stack:** Python, pytest

---

### Task 1: Remove `concrete` from `TypeType` and update core type system

**Files:**
- Modify: `dgen/type.py`

**Step 1: Write failing test**

No new test needed — existing tests will break. Verify current tests pass first:

Run: `python -m pytest test/test_type_values.py toy/test/test_layout.py -q`

**Step 2: Modify TypeType class**

In `dgen/type.py`, change `TypeType` (lines 161-182) from:

```python
@dataclass(frozen=True)
class TypeType(Type):
    """A type whose values are themselves types.

    TypeType(concrete=Index()) wraps Index as a first-class value.
    """

    concrete: Value[TypeType]
    __params__: ClassVar[Fields] = (("concrete", Type),)

    @property
    def __layout__(self) -> TypeValue:
        """Layout for this type as a value — a pointer to a self-describing Record."""
        return TypeValue()

    @property
    def ready(self) -> bool:
        return isinstance(self.concrete, Type) or self.concrete.ready

    @cached_property
    def type(self) -> TypeType:
        return self
```

To:

```python
@dataclass(frozen=True)
class TypeType(Type):
    """A type whose values are themselves types.

    TypeType() is the metatype — its values are type descriptors.
    The concrete identity of a type value is encoded in the TypeValue
    layout (self-describing via tag), not in the TypeType itself.
    """

    @property
    def __layout__(self) -> TypeValue:
        """Layout for this type as a value — a pointer to a self-describing Record."""
        return TypeValue()

    @cached_property
    def type(self) -> TypeType:
        return self
```

Key changes:
- Remove `concrete` field
- Remove `__params__` (inherits empty tuple from `Type`)
- Remove `ready` property (no params to check, inherits default)

**Step 3: Simplify `Type.type` property**

In `dgen/type.py`, line 109-111, change:

```python
@cached_property
def type(self) -> TypeType:
    return TypeType(concrete=self)
```

To:

```python
@cached_property
def type(self) -> TypeType:
    return TypeType()
```

**Step 4: Verify tests fail as expected**

Run: `python -m pytest test/ toy/test/ -q 2>&1 | head -40`

Expected: Many failures from `TypeType(concrete=...)` calls and ASM mismatches.

**Step 5: Commit**

```
jj commit -m "remove TypeType.concrete parameter, make TypeType unparameterized"
```

### Task 2: Update staging.py

**Files:**
- Modify: `dgen/staging.py`

**Step 1: Simplify `_resolve_comptime_field`**

In `dgen/staging.py`, lines 156-163, change:

```python
    # For TypeType results, reconstruct the concrete type from the dict
    # so the ConstantOp's layout correctly includes all param fields.
    # (e.g. Natural()'s layout only has the tag, but Successor needs tag + pred)
    const_type = value.type
    if isinstance(result, dict) and "tag" in result:
        concrete = dgen.type._type_from_dict(result)
        const_type = dgen.type.TypeType(concrete=concrete)
    const_op = ConstantOp(value=result, type=const_type)
```

To:

```python
    const_type = value.type
    if isinstance(result, dict) and "tag" in result:
        const_type = dgen.type.TypeType()
    const_op = ConstantOp(value=result, type=const_type)
```

The `_type_from_dict` call and `concrete=` wrapping are no longer needed — TypeType() is always the same, and the layout is self-describing via TypeValue.

**Step 2: Commit**

```
jj commit -m "simplify staging: TypeType() needs no concrete parameter"
```

### Task 3: Register `Type` as ASM alias for `TypeType`

**Files:**
- Modify: `dgen/module.py`

**Step 1: Add Type alias registration**

In `dgen/module.py`, after line 136 (`builtin.type("TypeType")(TypeType)`), add:

```python
builtin.types["Type"] = TypeType
```

This registers `Type` as an alternative ASM name that resolves to `TypeType`. The existing `_lookup_type` in the parser will find it.

**Step 2: Verify parsing works**

```python
# Quick manual check: "Type" should now parse as TypeType
python -c "from dgen.dialects import builtin; from dgen.asm.parser import parse_module; print('ok')"
```

**Step 3: Commit**

```
jj commit -m "register Type as ASM alias for TypeType"
```

### Task 4: Update test_type_values.py

**Files:**
- Modify: `test/test_type_values.py`

**Step 1: Update all ASM strings**

Replace all `TypeType<...>` type annotations in IR strings with `Type`. Examples:

- `%t : TypeType<Index> = ...` → `%t : Type = ...`
- `%tt : TypeType<%arr_ty> = ...` → `%tt : Type = ...`
- `TypeType(concrete=...)` → `TypeType()`

Specific tests to update:
- `test_typetype_constant_asm_roundtrip` (line 28): `TypeType<Index>` → `Type`
- `test_ssa_ref_as_op_type` (line 39): `TypeType<Index>` → `Type`
- `test_ssa_ref_as_op_type_roundtrip` (line 56): `TypeType<Index>` → `Type`
- `test_parameterized_typetype_constant_roundtrip` (line 79): `TypeType<Array<Index, 4>>` → `Type`
- `test_array_with_ssa_element_type` (line 122): `TypeType<Index>` → `Type`
- `test_pointer_with_ssa_pointee` (line 163): `TypeType<Index>` → `Type`
- `test_type_value_jit_identity` (line 186): use `idx.type` instead of constructing TypeType
- `test_type_constant_jit_return` (line 204): use `idx.type` instead of constructing TypeType
- `test_list_with_ssa_element_type` (line 221): `TypeType<Index>` → `Type`
- `test_fat_pointer_with_ssa_pointee` (line 243): `TypeType<F64>` → `Type`
- `test_function_with_ssa_result_type` (line 265): `TypeType<Index>` → `Type`
- `test_typetype_layout_with_block_arg_is_fixed` (line 455): `TypeType(concrete=...)` → `TypeType()`
- `test_parse_typetype_block_arg_constant_materializes` (line 474): `TypeType<Array<Index, 4>>` → `Type`, `TypeType<%arr_ty>` → `Type`

**Step 2: Run tests**

Run: `python -m pytest test/test_type_values.py -q`
Expected: All pass.

**Step 3: Commit**

```
jj commit -m "update test_type_values: TypeType<T> → Type"
```

### Task 5: Update test_layout.py

**Files:**
- Modify: `toy/test/test_layout.py`

**Step 1: Update TypeType constructions**

Replace all `TypeType(concrete=...)` with `TypeType()`:
- `test_type_value_memory_non_parametric` (line 218): `TypeType(concrete=ty)` → `TypeType()`
- `test_type_value_memory_parametric` (line 225): `TypeType(concrete=...)` → `TypeType()`
- `test_type_value_memory_pointer` (line 233): `TypeType(concrete=...)` → `TypeType()`
- `test_type_value_memory_nil` (line 241): `TypeType(concrete=...)` → `TypeType()`
- `test_type_value_memory_nested` (line 249): `TypeType(concrete=...)` → `TypeType()`
- `test_type_type_layout_non_parametric` (line 263): `TypeType(concrete=...)` → `TypeType()`
- `test_type_type_layout_parametric` (line 270): `TypeType(concrete=...)` → `TypeType()`

**Step 2: Run tests**

Run: `python -m pytest toy/test/test_layout.py -q`
Expected: All pass.

**Step 3: Commit**

```
jj commit -m "update test_layout: TypeType(concrete=T) → TypeType()"
```

### Task 6: Update test_tuple.py

**Files:**
- Modify: `test/test_tuple.py`

**Step 1: Update ASM string**

In `test_tuple_type_values` (line 89), change:
```
%types : Tuple<[TypeType<Index>, TypeType<String>]> = [Index, String]
```
To:
```
%types : Tuple<[Type, Type]> = [Index, String]
```

**Step 2: Run tests**

Run: `python -m pytest test/test_tuple.py -q`
Expected: All pass.

**Step 3: Commit**

```
jj commit -m "update test_tuple: TypeType<T> → Type"
```

### Task 7: Update test_peano.py

**Files:**
- Modify: `test/test_peano.py`

**Step 1: Update TypeType constructions and ASM strings**

- Remove `TypeType(concrete=...)` constructions, use `TypeType()` instead
- Lines using `TypeType(concrete=...)`:
  - Line 71: `type: Type = TypeType(concrete=Zero())` → `type: Type = TypeType()`
  - Line 80: `type: Value[TypeType] = TypeType(concrete=Zero())` → `type: Value[TypeType] = TypeType()`
  - Line 110: `type=TypeType(concrete=z)` → `type=TypeType()`
  - Line 122: `type=TypeType(concrete=succ)` → `type=TypeType()`
  - Line 229: `type=TypeType(concrete=nat)` → `type=TypeType()`
- `test_natural_trait_wraps_concrete` (line 187): `Natural(concrete=Zero())` — keep for now, Natural still has its own concrete field
- ASM strings with `TypeType<peano.X>`:
  - `test_peano_constant` (line 233): `TypeType<peano.Zero>` → `Type`, `TypeType<peano.Successor<...>>` → `Type`
  - `test_recursive_peano` (line 414): same pattern — all `TypeType<peano.X>` → `Type`

**Step 2: Run tests**

Run: `python -m pytest test/test_peano.py -q`
Expected: All pass.

**Step 3: Commit**

```
jj commit -m "update test_peano: TypeType<T> → Type, remove concrete= constructions"
```

### Task 8: Clean up TODO and run full test suite

**Files:**
- Modify: `toy/TODO.md`

**Step 1: Remove completed TODO items**

Remove these lines from `toy/TODO.md`:
- `- TypeType<T> -> Type in asm`
- `- Register Type as an ASM type name (alias for TypeType) so Tuple<[Type, Type]> works instead of requiring Tuple<[TypeType<Index>, TypeType<String>]>`

**Step 2: Run full test suite**

Run: `python -m pytest . -q`
Expected: All 432+ tests pass.

**Step 3: Run lints**

```bash
ruff format
ruff check --fix
```

**Step 4: Commit**

```
jj commit -m "clean up TODOs: TypeType<T> → Type is done"
```
