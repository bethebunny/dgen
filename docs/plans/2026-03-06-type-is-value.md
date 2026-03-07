# Type Is a Value — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Type` a subclass of `Value[TypeType]`, eliminating the dual bare-Type/Constant[TypeType] representation and all associated magic.

**Architecture:** Merge `value.py` into `type.py` to resolve the circular dependency, then make `Type` extend `Value`. Remove `__init_subclass__`/`__post_init__`/`as_value`/`_type_to_json` and all `isinstance(TypeType)` special cases. Update the generator to emit `TypeType` in `__params__` for type-kinded params. Regenerate all dialect files.

**Tech Stack:** Python, pytest, ruff, jj

**Design doc:** `docs/plans/2026-03-06-type-is-value-design.md`

**Baseline:** 356 passed, 2 xfailed. Run `python -m pytest . -q` after every change.

**Important:**
- Use `jj` not `git` for VCS (see CLAUDE.md)
- **Never hand-edit generated files.** Fix the generator or `.dgen` source instead.
- **No function-level imports** except for genuine circular dependencies.
- After regenerating dialect files, always verify output matches by running the generator and diffing.
- Regenerate commands:
  ```bash
  python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py
  python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.py
  python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.py
  python -m dgen.gen toy/dialects/toy.dgen -I affine=toy.dialects.affine > toy/dialects/toy.py
  ```

---

### Task 1: Merge value.py into type.py

`Type` needs to inherit from `Value`, but `value.py` imports from `type.py`. Merging resolves this circular dependency. This is a pure refactoring — no behavioral changes.

**Files:**
- Modify: `dgen/type.py` — absorb contents of `value.py`
- Delete: `dgen/value.py`
- Modify: all files that import from `dgen/value.py` or `dgen.value` — update import paths

**Step 1: Move Value and Constant into type.py**

Read `dgen/value.py` (72 lines). Move the `Value` class and `Constant` class into `dgen/type.py`, placing them between the `Type` class and `TypeType`. The file structure should be:

1. Layout imports
2. `Value` class (currently in value.py)
3. `Type(Value)` — **not yet**, just `Type` for now (we change the inheritance in Task 2)
4. `Constant(Value)` class (currently in value.py)
5. `TypeType(Type)`
6. `Memory`

Move the `Value` class ABOVE `Type` in the file (since `Type` will inherit from it in Task 2). Move `Constant` after `Type` but before `TypeType`.

Remove all imports that `value.py` had from `type.py` (they're now in the same file). Keep the `import dgen` for the `dgen.Block` forward reference in `Value.blocks`. Remove the `TYPE_CHECKING` import of `Layout` from value.py — it can use `Layout` directly since type.py already imports it.

Remove `value.py`'s `T = TypeVar("T", bound=Type)` since type.py already has this.

**Step 2: Update all imports**

Every file that does `from .value import ...` or `from dgen.value import ...` needs updating to import from `.type` or `dgen.type` instead.

Files to update (use grep to verify, these are the known sites):

- `dgen/__init__.py:5` — change `from .value import Constant, Value` to `from .type import Constant, Value`
- `dgen/op.py:9` — change `from .value import Constant, Value` to `from .type import Constant, Value`
- `dgen/block.py:9` — change `from .value import Value` to `from .type import Value`
- `dgen/staging.py:17` — change `from dgen.value import Constant` to `from dgen.type import Constant`
- `dgen/asm/formatting.py:16` — change `from ..value import Constant, Value` to `from ..type import Constant, Value`

Remove the `TYPE_CHECKING` guard around `from .value import Constant` in the old type.py — `Constant` is now in the same file.

**Step 3: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 4: Run linting**

Run: `ruff format && ruff check --fix`

**Step 5: Commit**

```bash
jj commit -m "refactor: merge value.py into type.py"
```

---

### Task 2: Make Type extend Value

The core change. `Type` inherits from `Value["TypeType"]`. Add lazy `.type` and `.__constant__` properties. Change `Value.ready` and `Constant.ready` from `ClassVar` to `@property`.

**Files:**
- Modify: `dgen/type.py`

**Step 1: Change Value.ready and Constant.ready to properties**

In the `Value` class, change:
```python
ready: ClassVar[bool] = False
```
to:
```python
@property
def ready(self) -> bool:
    return False
```

In the `Constant` class, change:
```python
ready: ClassVar[bool] = True
```
to:
```python
@property
def ready(self) -> bool:
    return True
```

Remove `ClassVar` from the typing imports in type.py if it's no longer used elsewhere. (Check first — `__params__` uses `ClassVar`.)

**Step 2: Make Type inherit from Value**

Change the class declaration from:
```python
class Type:
```
to:
```python
class Type(Value["TypeType"]):
```

Add `name: None = None` as a field on `Type` (types are never named SSA values).

Add these properties to `Type`:

```python
@cached_property
def type(self) -> TypeType:
    return TypeType(concrete=self)

@property
def ready(self) -> bool:
    return all(val.ready for _, val in self.parameters)

@cached_property
def __constant__(self) -> Memory[TypeType]:
    tt = self.type
    data: dict[str, object] = {"tag": self._asm_tag}
    for name, _ in self.__params__:
        data[name] = getattr(self, name).__constant__.to_json()
    return Memory.from_json(tt, data)
```

Add a helper property for the tag:

```python
@cached_property
def _asm_tag(self) -> str:
    cls = type(self)
    dialect = getattr(cls, "dialect", None)
    prefix = (
        f"{dialect.name}."
        if dialect is not None and dialect.name != "builtin"
        else ""
    )
    return f"{prefix}{getattr(cls, '_asm_name', type(self).__name__)}"
```

Add `from functools import cached_property` at the top of the file.

**Step 3: Fix TypeType**

`TypeType` inherits from `Type` which now inherits from `Value`. TypeType's `.type` would be `TypeType(concrete=TypeType(...))` — infinite recursion. Override `.type` on TypeType to break the cycle:

```python
@dataclass(frozen=True)
class TypeType(Type):
    concrete: Type
    __params__: ClassVar[Fields] = (("concrete", Type),)

    @property
    def __layout__(self) -> Layout:
        return self.concrete.type_layout

    @cached_property
    def type(self) -> TypeType:
        return self
```

Wait — `TypeType` is a frozen dataclass, and `Type` becoming a `Value` subclass means `Value.__init__` will want to set fields. Since `Value` is a dataclass with `name` and `type` fields, and `Type` is also a dataclass... this needs care.

Actually, `Type` subclasses (like `Index`, `F64`, etc.) are `@dataclass(frozen=True)`. `Value` is `@dataclass(eq=False, kw_only=True)`. When `Type` extends `Value`, the dataclass inheritance should work: `Type`'s dataclass fields include `Value`'s fields (`name`, `type`) plus its own. But `type` is now a `cached_property`, not a dataclass field, so we need to handle this.

The simplest approach: `Type` is NOT a dataclass itself. It inherits from `Value` (which is a dataclass) but doesn't use `@dataclass`. Its subclasses (`Index`, `F64`, etc.) ARE dataclasses. `Type` overrides `name` as a class attribute `None`, and provides `type` and `__constant__` as cached properties.

Since `Value` is `@dataclass(eq=False, kw_only=True)` with fields `name: str | None = None` and `type: T`, and `Type` provides `type` as a cached_property, we need `Type.__init__` to NOT require `type` as a constructor argument. Remove `type` from `Value`'s dataclass fields — make it a property on `Value` that raises `NotImplementedError`, overridden by `Constant` (which stores it) and `Type` (cached_property).

**Revised approach for Value:**

```python
@dataclass(eq=False, kw_only=True)
class Value:
    name: str | None = None

    @property
    def type(self) -> Type:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        return False

    @property
    def operands(self) -> list[Value]:
        return []

    @property
    def blocks(self) -> dict[str, dgen.Block]:
        return {}

    @property
    def __constant__(self) -> Memory:
        raise NotImplementedError
```

`Constant` stores `.type` and `.value` as dataclass fields:

```python
@dataclass(eq=False, kw_only=True)
class Constant(Value):
    type: Type
    value: Memory

    @property
    def ready(self) -> bool:
        return True

    @property
    def __constant__(self) -> Memory:
        return self.value
```

`Type` provides `.type` as a cached_property:

```python
class Type(Value):
    @cached_property
    def type(self) -> TypeType:
        return TypeType(concrete=self)
    # ...
```

This way, constructing `F64()` doesn't require passing `type=` — the `type` comes from the cached_property.

**Step 4: Run tests**

Run: `python -m pytest . -q`

Many tests will likely fail at this point because:
- `Op` inherits from `Value` and may expect `type` as a constructor arg
- Various places check `isinstance(x, Constant)` or access `.ready` as a class attribute

Fix failures iteratively. The key areas to watch:
- `Op` subclasses pass `type=...` as a constructor kwarg — this should still work since `Op` can declare `type` as a dataclass field that overrides the property
- `BlockArgument` in `block.py` — check its `type` field
- Tests that check `Value.ready` or `Constant.ready` as class attributes

**Step 5: Run linting**

Run: `ruff format && ruff check --fix`

**Step 6: Commit**

```bash
jj commit -m "feat: make Type extend Value — types are values"
```

---

### Task 3: Remove magic and special cases

Now that `Type` IS a `Value[TypeType]`, remove all the machinery that existed to bridge the two representations.

**Files:**
- Modify: `dgen/type.py` — remove `as_value`, `_type_to_json`, `__init_subclass__`/`__post_init__`, dead branches in `type_layout`
- Modify: `dgen/type.py` (Constant class) — remove `__layout__`, `__eq__`, `__hash__` special cases for TypeType
- Modify: `dgen/asm/formatting.py` — remove TypeType special case in `format_expr`
- Modify: `dgen/module.py` — remove `builtin.type("TypeType")(TypeType)` registration

**Step 1: Remove from type.py**

Delete these methods from `Type`:
- `as_value()` (lines 33-38)
- `_type_to_json()` (lines 40-57)
- `__init_subclass__()` (lines 59-79)

Simplify `type_layout` — remove the `isinstance(val, Type)` branch. After this change, all params are Values, so the unified path works:

```python
@property
def type_layout(self) -> Record:
    fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
    for _, val in self.parameters:
        fields.append((name, val.__constant__.type.__layout__))
    return Record(fields)
```

Wait — for type-kinded params, `val` is a `Type` instance. `val.__constant__` returns `Memory[TypeType]`. `val.__constant__.type` is `TypeType(concrete=val)`. `TypeType.__layout__` is `val.type_layout`. So `val.__constant__.type.__layout__` gives `val.type_layout`. This is correct — same result as the old `isinstance` branch.

**Step 2: Remove Constant special cases**

Delete `Constant.__layout__` property entirely (lines 47-56 in value section). Type-kinded params are now `Type` instances which have `.__layout__` directly (the data layout from the Type class hierarchy).

Delete the `isinstance(self.type, TypeType)` branches from `Constant.__eq__` and `Constant.__hash__`:

```python
def __eq__(self, other: object) -> bool:
    if not isinstance(other, Constant):
        return NotImplemented
    if self.type != other.type:
        return False
    return self.value == other.value

def __hash__(self) -> int:
    return hash((type(self), self.type, self.value))
```

**Step 3: Remove format_expr special case**

In `dgen/asm/formatting.py`, the block at lines 95-98:

```python
if isinstance(value, Constant) and not isinstance(value, Op):
    if isinstance(value.type, TypeType):
        return format_expr(value.type.concrete, tracker)
    return format_expr(value.__constant__.to_json(), tracker)
```

Remove the TypeType branch. `Type` instances are now `Value` instances and will be caught by the `isinstance(value, Type)` check later in the function (line 116). But we need to make sure `Type` is checked BEFORE `Value` since `Type` is now a subclass of `Value`. The existing order already has `Constant` before `Value` before `Type` — we need `Type` before the generic `Value` branch.

Reorder the checks in `format_expr`:
1. `Nil` check
2. `PackOp` check
3. `Constant and not Op` — format via `__constant__.to_json()` (no TypeType branch)
4. `Type` — format as type (move this BEFORE the `Value` check)
5. `Value` — format as `%name`
6. `Memory`, `list`, primitives...

Actually, since `Type` is now a `Value`, and `Type` comes after `Value` in the current order, Type instances would be caught by the `Value` branch and formatted as `%name` — wrong. The `Type` check MUST come before the generic `Value` check.

But also: a `Type` that is `.ready` is effectively a constant, and a `Type` that is not ready (like `Array<%0, 4>`) would need to be formatted as... what? Currently non-ready types don't appear as standalone values in the IR. For now, always format Type as a type literal.

Revised order:
```python
if isinstance(value, Nil):
    return "()"
if isinstance(value, PackOp):
    return "[" + ... + "]"
if isinstance(value, Type):
    # Types always format as type literals
    if getattr(type(value), "_asm_name", None) is not None:
        return type_asm(value, tracker)
    asm: str = getattr(value, "asm")
    return asm
if isinstance(value, Constant) and not isinstance(value, Op):
    return format_expr(value.__constant__.to_json(), tracker)
if isinstance(value, Value):
    ...
```

Note: `Nil` is a `Type`, so the `Nil` check must stay before the `Type` check.

**Step 4: Remove TypeType dialect registration**

In `dgen/module.py`, delete line 132:
```python
builtin.type("TypeType")(TypeType)
```

And remove `TypeType` from the import on line 21.

**Step 5: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 6: Run linting**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "refactor: remove TypeType magic — types are naturally values"
```

---

### Task 4: Update __params__ to use TypeType for type-kinded params

Type-kinded params should declare `TypeType` as the field type in `__params__`, not `Type`. This affects the generator and all generated files.

**Files:**
- Modify: `dgen/gen/python.py:97-109,259-261,267-271` — change generator output for type-kinded params
- Regenerate: all 4 dialect `.py` files
- Modify: `dgen/type.py` — update `TypeType.__params__` from `Type` to `TypeType`

**Step 1: Update the generator**

In `dgen/gen/python.py`, change `_resolve_type_ref`:
```python
def _resolve_type_ref(ref: TypeRef) -> str:
    if ref.name == "Type":
        return "TypeType"
    return ref.name
```

Change `_annotation_for_param`:
```python
def _annotation_for_param(param: ParamDecl) -> str:
    if param.variadic:
        inner = _resolve_type_ref(param.type)
        return f"list[Value[{inner}]]"
    if param.type.name == "Type":
        return "Type"
    return f"Value[{param.type.name}]"
```

The annotation stays `Type` (since `Type` IS `Value[TypeType]` now), but `__params__` entries use `TypeType`.

**Step 2: Regenerate all dialect files**

```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py
python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.py
python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/affine.py
python -m dgen.gen toy/dialects/toy.dgen -I affine=toy.dialects.affine > toy/dialects/toy.py
```

Verify the generated files import `TypeType` where needed. The generator's `from dgen import ...` line may need `TypeType` added — check if the generated `__params__` tuples reference `TypeType`.

**Step 3: Update TypeType.__params__**

In `dgen/type.py`, change TypeType's `__params__`:
```python
__params__: ClassVar[Fields] = (("concrete", TypeType),)
```

Wait — this is self-referential (TypeType not yet defined when the class body executes). Since `TypeType` is the last thing in the type hierarchy, this should work as long as the class is defined by the time `__params__` is accessed at runtime (not at class definition time). With `from __future__ import annotations` this works for annotations, but `__params__` is a runtime value. Use a string and resolve later, or just leave it as `Type` since TypeType IS a Type. Actually — `TypeType`'s concrete param is genuinely any `Type`, and `Type` is a `Value[TypeType]`, so `TypeType` is correct in __params__. But we need the name to be available. Since `TypeType` is defined in the same file and `__params__` is a class body statement... `TypeType` isn't available during its own class body. Leave it as `Type` — this is the one place where `Type` in `__params__` is correct, since TypeType's param genuinely accepts any Type.

**Step 4: Update parser isinstance checks**

In `dgen/asm/parser.py`, simplify `isinstance(raw_value, (Value, Type))` to `isinstance(raw_value, Value)` since `Type` is now a `Value`. Affected lines: 147, 172, 176.

**Step 5: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed

**Step 6: Run linting**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "refactor: use TypeType in __params__, simplify parser checks"
```

---

### Task 5: Update tests

Tests that reference `builtin.TypeType` or construct `Constant[TypeType]` manually need updating.

**Files:**
- Modify: `toy/test/test_layout.py` — update TypeType test imports and assertions
- Modify: `toy/test/test_type_roundtrip.py` — if any tests construct TypeType manually

**Step 1: Update test_layout.py**

Tests like `test_type_value_memory_non_parametric` currently do:
```python
metatype = TypeType(concrete=ty)
mem = Memory.from_json(metatype, {"tag": "builtin.Index"})
```

These should still work — `TypeType` still exists and `Memory.from_json` still works. But verify that `ty.__constant__` now gives the same Memory that was manually constructed. Add assertions:

```python
def test_type_is_value():
    """F64() is a Value[TypeType] with a valid __constant__."""
    ty = builtin.F64()
    assert isinstance(ty, Value)
    assert ty.ready
    assert ty.__constant__.to_json() == {"tag": "F64"}

def test_type_not_ready_when_param_unresolved():
    """A Type with an unresolved param has ready=False."""
    # This tests the concept — may need a mock unresolved Value
    ...
```

**Step 2: Run tests**

Run: `python -m pytest . -q`
Expected: 356 passed, 2 xfailed (plus any new tests)

**Step 3: Run linting**

Run: `ruff format && ruff check --fix`

**Step 4: Commit**

```bash
jj commit -m "test: update tests for Type-is-Value"
```

---

### Task 6: Clean up

Final cleanup pass.

**Files:**
- Modify: `dgen/type.py` — remove any unused imports, dead code
- Modify: `dgen/__init__.py` — ensure `TypeType` is exported if needed
- Modify: `dgen/module.py` — remove TypeType import if no longer needed
- Verify: all generated files are exactly generator output

**Step 1: Verify generated files match**

```bash
python -m dgen.gen dgen/dialects/builtin.dgen | diff - dgen/dialects/builtin.py
python -m dgen.gen dgen/dialects/llvm.dgen | diff - dgen/dialects/llvm.py
python -m dgen.gen toy/dialects/affine.dgen | diff - toy/dialects/affine.py
python -m dgen.gen toy/dialects/toy.dgen -I affine=toy.dialects.affine | diff - toy/dialects/toy.py
```

Expected: no diff for any file.

**Step 2: Run full validation**

```bash
python -m pytest . -q
ruff format
ruff check --fix
```

**Step 3: Commit**

```bash
jj commit -m "refactor: final cleanup for Type-is-Value"
```
