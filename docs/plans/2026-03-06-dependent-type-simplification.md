# Dependent Type Simplification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the codebase now that TypeType enables true dependent types — remove `for_value`, unify type params as Values, and eliminate monkey-patches.

**Architecture:** Six tasks in dependency order. Tasks 1-3 are independent cleanups. Task 4 (unify type params) is the architectural change that enables Task 5 (remove DimSizeOp.resolve_constant). Task 6 moves Tensor.__layout__ to .dgen.

**Tech Stack:** Python, pytest, ruff, ty, jj

**Baseline:** 356 passed, 2 xfailed. Run `python -m pytest . -q` after every change.

---

### Task 1: Remove `for_value`

The `for_value` classmethod on `Type` infers a type instance from a raw Python value. It's used in 3 parser call sites and has 2 monkey-patched overrides. All call sites already know the field type from `__params__`, so the indirection is unnecessary.

**Files:**
- Modify: `dgen/type.py:33-34` (delete `for_value`)
- Modify: `dgen/asm/parser.py:130,157,163` (replace `for_value` calls)
- Modify: `dgen/module.py:139-145` (delete `_list_for_value` + monkey-patch)
- Modify: `toy/dialects/__init__.py:24-33` (delete `_shape_for_value` + monkey-patch)
- Modify: `toy/test/test_layout.py:38` (replace `String.for_value("hello")`)
- Modify: `toy/test/test_type_roundtrip.py:373-374` (replace `String.for_value(...)`)

**Step 1: Update test call sites**

In `toy/test/test_layout.py:38`, change:
```python
s = builtin.String.for_value("hello")
```
to:
```python
s = builtin.String()
```

In `toy/test/test_type_roundtrip.py:373-374`, change:
```python
s3 = builtin.String.for_value("abc")
s5 = builtin.String.for_value("hello")
```
to:
```python
s3 = builtin.String()
s5 = builtin.String()
```

**Step 2: Replace `for_value` calls in the parser**

In `dgen/asm/parser.py`, the pattern `field_type.for_value(raw_value).constant(raw_value)` appears at lines 130, 157, and 163. Replace all three with:
```python
field_type().constant(raw_value)
```

Specifically:

Line 130 in `_parse_fields_from_exprs`:
```python
# Before
raw_value = field_type.for_value(raw_value).constant(raw_value)
# After
raw_value = field_type().constant(raw_value)
```

Line 157 in `parse_op_fields` (list element wrapping):
```python
# Before
f_type.for_value(v).constant(v)
# After
f_type().constant(v)
```

Line 163 in `parse_op_fields` (scalar wrapping):
```python
# Before
raw_value = f_type.for_value(raw_value).constant(raw_value)
# After
raw_value = f_type().constant(raw_value)
```

**Step 3: Delete `for_value` and its monkey-patches**

Delete from `dgen/type.py` (lines 32-34):
```python
@classmethod
def for_value(cls, value: object) -> Type:
    return cls()
```

Delete from `dgen/module.py` (lines 139-145):
```python
@classmethod  # type: ignore[misc]
def _list_for_value(cls: type[List], value: object) -> List:
    assert isinstance(value, list)
    return cls(element_type=Index())


List.for_value = _list_for_value  # type: ignore[assignment]
```

Also update the module docstring at line 5 to remove the `List.for_value` mention.

Delete from `toy/dialects/__init__.py` (lines 24-33):
```python
@classmethod  # type: ignore[misc]
def _shape_for_value(cls: type[Shape], value: object) -> Shape:
    if isinstance(value, Constant):
        assert isinstance(value.type, Index)
        return cls(rank=Index().constant(value.__constant__.to_json()))
    assert isinstance(value, list)
    return cls(rank=Index().constant(len(value)))


Shape.for_value = _shape_for_value  # type: ignore[assignment]
```

**Step 4: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 5: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "refactor: remove for_value — parser uses field type directly"
```

---

### Task 2: Move `Tensor.unpack_shape` to `.dgen`

The `unpack_shape` method is monkey-patched onto `Tensor` in `toy/dialects/__init__.py`. It's a one-liner (`self.shape.__constant__.to_json()`) that can be a generated method.

**Files:**
- Modify: `toy/dialects/toy.dgen:4-6` (add method)
- Modify: `toy/dialects/__init__.py:47-52` (delete monkey-patch)
- Regenerate: `toy/dialects/toy.py`

**Step 1: Add method to toy.dgen**

In `toy/dialects/toy.dgen`, change the Tensor type declaration from:
```
type Tensor<shape: affine.Shape, dtype: Type = F64>:
    # TODO: layout requires math.prod of shape dims, not yet expressible in .dgen.
    # Temporary workaround: __layout__ monkey-patched in toy/dialects/__init__.py
```
to:
```
type Tensor<shape: affine.Shape, dtype: Type = F64>:
    # TODO: layout requires math.prod of shape dims, not yet expressible in .dgen.
    # Temporary workaround: __layout__ monkey-patched in toy/dialects/__init__.py

    method unpack_shape(self) -> list:
        return self.shape.__constant__.to_json()
```

**Step 2: Regenerate toy.py**

Run: `python -m dgen.gen toy/dialects/toy.dgen --import-map affine=toy.dialects.affine > /tmp/toy_gen.py && cp /tmp/toy_gen.py toy/dialects/toy.py`

If there's no CLI for the generator, regenerate manually: read `toy/dialects/toy.py` and add the method to the `Tensor` class:
```python
    def unpack_shape(self):
        return self.shape.__constant__.to_json()
```

**Step 3: Delete monkey-patch**

Delete from `toy/dialects/__init__.py` (lines 47-52):
```python
def _tensor_unpack_shape(self: Tensor) -> list[int]:
    """Extract concrete shape dimensions as a list of ints."""
    return self.shape.__constant__.to_json()


Tensor.unpack_shape = _tensor_unpack_shape  # type: ignore[assignment]
```

**Step 4: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 5: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "refactor: move Tensor.unpack_shape to toy.dgen method"
```

---

### Task 3: Move `FunctionOp.asm` dispatch into `Op.asm`

The `FunctionOp.asm` property is monkey-patched in `dgen/module.py`. Instead, `Op.asm` should dispatch to `format_func` for ops with the `HasSingleBlock` trait that are top-level functions (i.e., FunctionOp).

**Files:**
- Modify: `dgen/op.py:40-43` (update `asm` property)
- Modify: `dgen/module.py:148-153` (delete monkey-patch)
- Modify: `dgen/asm/formatting.py` (import check)

**Step 1: Update Op.asm**

In `dgen/op.py`, change the `asm` property (lines 40-43) from:
```python
@property
def asm(self) -> Iterable[str]:
    from .asm.formatting import op_asm

    return op_asm(self)
```
to:
```python
@property
def asm(self) -> Iterable[str]:
    from .asm.formatting import format_func, op_asm

    if self._asm_name == "function":
        return format_func(self)
    return op_asm(self)
```

**Step 2: Delete monkey-patch**

Delete from `dgen/module.py` (lines 148-153):
```python
@property  # type: ignore[misc]
def _function_asm(self: FunctionOp) -> Iterable[str]:
    return format_func(self)


FunctionOp.asm = _function_asm  # type: ignore[assignment, misc]
```

Also remove the `format_func` import from `dgen/module.py:15` (it's only used by the monkey-patch). And update the docstring at line 4-5.

**Step 3: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 4: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "refactor: move FunctionOp.asm dispatch into Op.asm"
```

---

### Task 4: Unify type-kinded params as `Value[TypeType]`

This is the main architectural change. Type-kinded params (e.g., `element_type: Type` on `List`, `dtype: Type` on `Tensor`) become `Constant[TypeType[concrete]]` values. This eliminates the Type/Value duality throughout the codebase.

This task is large and should be broken into sub-steps. After each sub-step, tests must pass.

**Files:**
- Modify: `dgen/type.py` (type_layout, add `as_value` helper)
- Modify: `dgen/op.py` (parameters iterator type annotation)
- Modify: `dgen/asm/parser.py` (wrap parsed Types as TypeType constants)
- Modify: `dgen/asm/formatting.py` (unwrap TypeType constants for display)
- Modify: `dgen/staging.py` (_field_values now captures all params)
- Modify: `dgen/dialects/builtin.py` (param annotations on List, Array, Pointer, etc.)
- Modify: `toy/dialects/toy.py` (param annotations on Tensor)
- Modify: `toy/dialects/affine.py` (param annotations on MemRef)
- Modify: `dgen/gen/python.py` (code generator for type-kinded params)
- Modify: All lowering passes that construct types with type params
- Test: All existing tests must continue to pass

**Sub-step 4a: Add `Type.as_value()` helper**

Add to `dgen/type.py`, after the `constant` method:
```python
def as_value(self) -> Constant:
    """Wrap this type as a Constant[TypeType] value."""
    from .value import Constant
    from dgen.dialects.builtin import TypeType

    tt = TypeType(concrete=self)
    return Constant(type=tt, value=Memory.from_json(tt, self._type_to_json()))

def _type_to_json(self) -> dict[str, object]:
    """Serialize this type to a JSON-compatible dict (tag + params)."""
    cls = type(self)
    dialect = getattr(cls, "dialect", None)
    prefix = f"{dialect.name}." if dialect is not None and dialect.name != "builtin" else ""
    tag = f"{prefix}{getattr(cls, '_asm_name', type(self).__name__)}"
    result: dict[str, object] = {"tag": tag}
    for name, _ in self.__params__:
        val = getattr(self, name)
        if isinstance(val, Type):
            result[name] = val._type_to_json()
        else:
            result[name] = val.__constant__.to_json()
    return result
```

Run: `python -m pytest . -q` — should still pass (no callers yet).

**Sub-step 4b: Update parser to wrap Type results as TypeType constants**

In `dgen/asm/parser.py`, in `_parse_fields_from_exprs` (line ~129), change:
```python
if not isinstance(raw_value, (Value, Type)):
    raw_value = field_type().constant(raw_value)
kwargs[name] = raw_value
```
to:
```python
if isinstance(raw_value, Type) and not isinstance(raw_value, Value):
    raw_value = raw_value.as_value()
elif not isinstance(raw_value, Value):
    raw_value = field_type().constant(raw_value)
kwargs[name] = raw_value
```

Apply the same pattern in `parse_op_fields` (lines ~154, ~163).

**Sub-step 4c: Update formatter to unwrap TypeType constants**

In `dgen/asm/formatting.py`, in `format_expr` (before the existing `isinstance(value, Constant)` check at line 95), add:
```python
if isinstance(value, Constant) and isinstance(value.type, TypeType):
    return format_expr(value.type.concrete, tracker)
```

This ensures TypeType-wrapped types still print as type literals (e.g., `Index` not `%0`).

**Sub-step 4d: Update type declarations**

Change type-kinded param annotations from bare `Type` to `Value`. For each type with `Type`-kinded params in `__params__`:

`dgen/dialects/builtin.py` — Array, Pointer, FatPointer, List:
```python
# Array: element_type changes from `Type` to `Value`
element_type: Value  # was: Type
```

The `__params__` tuples stay the same — the field type class (`Type`) is used for parsing, not annotation.

Actually — reconsider: `__params__` entries with `Type` as the field type class tell the parser/staging what kind of param it is. We need a way to distinguish "this is a type-kinded param" from "this is a value-kinded param". Currently `Type` in `__params__` means "type-kinded".

Better approach: leave `__params__` as-is. The parser uses the field type class to decide wrapping. In `_parse_fields_from_exprs`, check if `field_type is Type` (meaning type-kinded):
```python
if field_type is Type:
    # Type-kinded param: wrap as TypeType constant
    if isinstance(raw_value, Type) and not isinstance(raw_value, Value):
        raw_value = raw_value.as_value()
elif not isinstance(raw_value, (Value, Type)):
    raw_value = field_type().constant(raw_value)
```

**Sub-step 4e: Update `type_layout` to use uniform Value access**

In `dgen/type.py`, change `type_layout` property:
```python
@property
def type_layout(self) -> Record:
    fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
    for name, field_type in self.__params__:
        val = getattr(self, name)
        if isinstance(val, Type):
            # Bare type (backwards compat during migration)
            fields.append((name, val.type_layout))
        else:
            # Value — works for both TypeType and regular values
            fields.append((name, val.__constant__.type.__layout__))
    return Record(fields)
```

This is actually unchanged — but once all constructors pass TypeType values, the `isinstance(val, Type)` branch becomes dead code. Remove it after all callers are migrated.

**Sub-step 4f: Update all type constructor call sites**

Every place that constructs a parameterized type with a bare `Type` argument needs updating. Search for these patterns and wrap with `.as_value()`:

- `List(element_type=X)` → `List(element_type=X.as_value())`
- `Array(element_type=X, n=...)` → `Array(element_type=X.as_value(), n=...)`
- `Pointer(pointee=X)` → `Pointer(pointee=X.as_value())`
- `FatPointer(pointee=X)` → `FatPointer(pointee=X.as_value())`
- `Tensor(shape=..., dtype=X)` → `Tensor(shape=..., dtype=X.as_value())`
- `MemRef(shape=..., dtype=X)` → `MemRef(shape=..., dtype=X.as_value())`
- `InferredShapeTensor(dtype=X)` → `InferredShapeTensor(dtype=X.as_value())`

Key files to update:
- `dgen/module.py` (ConstantOp, any List construction)
- `dgen/asm/parser.py` (list sugar expansion)
- `toy/dialects/__init__.py` (shape_constant, Tensor construction)
- `toy/passes/shape_inference.py` (all Tensor construction)
- `toy/passes/toy_to_affine.py` (MemRef construction)
- `toy/passes/affine_to_llvm.py` (MemRef references)
- `toy/parser/lowering.py` (initial Tensor construction from AST)
- All test files constructing these types

This is the highest-effort part. Use grep to find all sites:
```bash
grep -rn 'element_type=\|pointee=\|dtype=' dgen/ toy/ test/ --include='*.py'
```

**Sub-step 4g: Update `__layout__` properties that access type params**

Types with parametric `__layout__` access type-kinded params directly. After migration, they need to unwrap the TypeType constant:

`dgen/dialects/builtin.py` — Array.__layout__:
```python
# Before
return layout.Array(self.element_type.__layout__, self.n.__constant__.to_json())
# After — element_type is now a Constant[TypeType], extract the concrete type
return layout.Array(self.element_type.type.concrete.__layout__, self.n.__constant__.to_json())
```

Wait — this is getting ugly. Better approach: add a property `Type.unwrap_type(value)` or have TypeType constants provide easy access:

Actually, the cleanest approach: on the `Type` side, when a type has a type-kinded param, the `__layout__` property should access the concrete type. Add a helper or access pattern.

**Alternative for 4d-4g:** Instead of changing all constructors at once, introduce the `as_value()` method and wrapping in the parser, but keep bare Types working in constructors for now. The parser is the main entry point for deserialization. For Python-level construction, add an `__init_subclass__` or `__post_init__` hook on types that auto-wraps bare Type args. This reduces the migration surface dramatically.

Add to parameterized type classes (or a base mixin):
```python
def __post_init__(self):
    for name, field_type in self.__params__:
        val = getattr(self, name)
        if field_type is Type and isinstance(val, Type) and not isinstance(val, Value):
            object.__setattr__(self, name, val.as_value())
```

This auto-wraps bare Types at construction time, so all existing code works unchanged. The `frozen=True` dataclass requires `object.__setattr__`.

**Sub-step 4h: Run full test suite**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "feat: unify type-kinded params as Value[TypeType] constants"
```

---

### Task 5: Remove `DimSizeOp.resolve_constant` monkey-patch

With type params as Values, the staging system sees them. `DimSizeOp` becomes naturally resolvable — its output is a function of type params that the staging loop can JIT-evaluate.

**Prerequisite:** Task 4 complete.

**Files:**
- Modify: `toy/dialects/__init__.py:70-80` (delete monkey-patch)
- Modify: `dgen/staging.py:123-142` (remove `_resolve_constant_ops`)
- Modify: `dgen/staging.py` (remove calls to `_resolve_constant_ops`)

**Step 1: Verify DimSizeOp resolves through staging**

Before removing anything, confirm that with type params as Values, the staging loop naturally resolves DimSizeOp. Write a test:

```python
def test_dim_size_resolves_through_staging():
    """DimSizeOp should resolve without resolve_constant, via staging."""
    # Temporarily remove resolve_constant and verify end-to-end still works
    ...
```

If staging doesn't naturally resolve DimSizeOp (because `type` isn't yet an honorary `__params__` field), this task is **blocked** and should be deferred. In that case, just note it as future work.

**Step 2: If staging resolves it, delete the monkey-patch**

Delete from `toy/dialects/__init__.py`:
```python
def _dim_size_resolve_constant(self: DimSizeOp) -> int | None:
    ...
DimSizeOp.resolve_constant = _dim_size_resolve_constant
```

Delete `_resolve_constant_ops` from `dgen/staging.py` and its two call sites (lines 253, 329).

**Step 3: Run tests**

Run: `python -m pytest . -q`

**Step 4: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "refactor: remove resolve_constant — staging handles DimSizeOp naturally"
```

---

### Task 6: Move `Tensor.__layout__` to `.dgen`

The `Tensor.__layout__` property computes `layout.Array(layout.Float64(), prod(shape_dims))`. Shape already has a `num_elements` method in `affine.dgen`. The blocker is that `.dgen` data field expressions can't call methods.

**Prerequisite:** Task 2 (unpack_shape moved to .dgen).

**Files:**
- Modify: `dgen/gen/python.py:70-88` (support method calls in layout expressions)
- Modify: `toy/dialects/toy.dgen:4-6` (add data field with method call)
- Modify: `toy/dialects/__init__.py:55-62` (delete monkey-patch)
- Regenerate: `toy/dialects/toy.py`
- Test: `test/test_gen_python.py`

**Step 1: Extend `_layout_expr` in code generator**

In `dgen/gen/python.py`, the `_layout_expr` function (line 70) resolves type refs to layout expressions. Currently for params it emits either `self.name.__layout__` (Type-kinded) or `self.name.__constant__.to_json()` (value-kinded).

Add support for method calls on params. When a TypeRef has the form `name.method()` (parsed as a CallExpr in the data field's type expression), emit `self.name.method()`.

This requires the `.dgen` parser to support method call syntax in data field type positions. The data field type is parsed as a `TypeRef`, but `Shape.num_elements()` isn't a TypeRef — it's an expression.

**Alternative approach:** Instead of extending data field syntax, use the existing `layout:` keyword with a custom expression. In `toy.dgen`:
```
type Tensor<shape: affine.Shape, dtype: Type = F64>:
    layout: Array<dtype, shape.num_elements()>
```

The `layout:` keyword currently only accepts a layout name (e.g., `layout Int`, `layout Array`). Extend it to accept parameterized expressions. The generator's handling at `python.py:260-272` already handles parametric layouts — it just needs to also handle method-call arguments in the param list.

**Step 2: Update .dgen parser for layout expressions with method calls**

In `dgen/gen/parser.py`, the `_parse_type_ref` function handles `Name<args>` syntax. For layout args that are method calls like `shape.num_elements()`, the existing `_parse_type_ref` will parse `shape.num_elements()` as a TypeRef with name `shape` — which is wrong.

Better: change the `layout:` keyword to accept an expression string that gets passed through to `_layout_expr`. Or define the layout using the data field syntax:
```
type Tensor<shape: affine.Shape, dtype: Type = F64>:
    data: Array<dtype, shape.num_elements()>
```

The data field's TypeRef for `Array<dtype, shape.num_elements()>` would have args:
- `TypeRef(name="dtype")`
- `TypeRef(name="shape.num_elements()")` — but this doesn't parse correctly

This needs the `.dgen` type ref parser to handle expressions, not just names. This is a more invasive change to the `.dgen` parser.

**Simplest approach:** Add a special-case in `_layout_expr` for known method patterns. When a param ref in a data field has no args and is value-kinded, check if the param's type has methods that return the needed value. Or just hard-code support for `name.method()` syntax in `_parse_type_ref`.

Actually — the `.dgen` parser's `_parse_type_ref` already calls `_parse_postfix` for sub-expressions when used in method bodies. The issue is only in data field parsing, which uses `_parse_type_ref`. Change the data field parser to use `_parse_expr` instead of `_parse_type_ref` for the args inside `<...>`, then map the resulting `Expr` nodes to layout expressions.

This is a significant refactor of the `.dgen` code generator. **If this proves too invasive, defer it** — the Tensor.__layout__ monkey-patch is the least harmful of the remaining patches.

**Step 3: If feasible, update toy.dgen and regenerate**

```
type Tensor<shape: affine.Shape, dtype: Type = F64>:
    data: Array<dtype, shape.num_elements()>

    method unpack_shape(self) -> list:
        return self.shape.__constant__.to_json()
```

Delete monkey-patch from `toy/dialects/__init__.py:55-62`.

**Step 4: Run tests**

Run: `python -m pytest . -q`

**Step 5: Format + commit**

```bash
ruff format && ruff check --fix
jj commit -m "refactor: move Tensor.__layout__ to toy.dgen data field"
```
