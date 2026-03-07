# Type Is a Value

## Problem

Types and values are separate hierarchies. `Type` describes layouts; `Value` is the base for SSA values. To make types first-class, we added `TypeType` and `__init_subclass__` magic that auto-wraps bare `Type` instances as `Constant[TypeType]`. This created a dual representation (bare `Type` vs `Constant[TypeType]`), dead code branches, `isinstance` special cases in `Constant.__eq__`/`__hash__`/`__layout__` and `format_expr`, and a `_type_to_json` method that duplicates `type_layout` logic.

## Core Insight

A type IS a value — specifically, a `Value[TypeType]`. This mirrors Python's `type`/`object` relationship: `type` is an `object`, `type(type) is type`.

## Design

### Type extends Value

`Type` inherits from `Value["TypeType"]`. Every type instance is a value of type `TypeType(concrete=self)`.

```python
class Type(Value["TypeType"]):
    __layout__: Layout
    __params__: ClassVar[Fields] = ()
    name: None

    @cached_property
    def type(self) -> TypeType:
        return TypeType(concrete=self)

    @property
    def ready(self) -> bool:
        return all(val.ready for _, val in self.parameters)

    @cached_property
    def __constant__(self) -> Memory[TypeType]:
        # Only valid when ready
        ...
```

The `.type` property is lazy — `TypeType(concrete=self)` is constructed on first access. This avoids the `type(type) is type` bootstrap problem. `.ready` checks whether all params are resolved. `.__constant__` builds the Memory buffer lazily, only when needed.

### Value.ready becomes a property

Currently `Value.ready` and `Constant.ready` are `ClassVar[bool]`. Since `Type.ready` depends on instance state (its params), `ready` becomes a `@property` throughout:

- `Value.ready` — returns `False`
- `Constant.ready` — returns `True`
- `Type.ready` — checks `all(val.ready for _, val in self.parameters)`
- `Op.ready` — already a property, checks `__params__` fields

### What gets removed

- `__init_subclass__` / `__post_init__` auto-wrapping
- `as_value()` — `Type` already IS a value
- `_type_to_json()` — Memory construction moves to `__constant__`
- `Constant.__layout__` special case for TypeType
- `Constant.__eq__` / `__hash__` special cases for TypeType
- `format_expr` special case for TypeType constants
- Dead `isinstance(val, Type)` branches in `type_layout` and `_type_to_json`

### __params__ changes

Type-kinded params change from `("element_type", Type)` to `("element_type", TypeType)`. The field annotation changes from `element_type: Type` to `element_type: Value[TypeType]`. Since `Type` IS `Value[TypeType]`, callers still pass `F64()` directly — no wrapping needed.

### __layout__ resolution

`Type.__layout__` (e.g., `F64.__layout__ = Float64()`) describes the runtime data layout. `TypeType.__layout__` (returns `concrete.type_layout`) describes the serialization layout of a type-as-value. These don't conflict — `self.element_type.__layout__` resolves to the Type's own class attribute, which is the data layout.

### Parser

`parse_type()` returns `Type` instances. Since `Type` is now `Value[TypeType]`, these are already values. The `isinstance(raw_value, (Value, Type))` checks simplify to `isinstance(raw_value, Value)`.

### Formatter

`format_expr` currently special-cases `Constant[TypeType]` to print the concrete type instead of Memory contents. In the new design, `Type` instances are recognized by the existing `isinstance(value, Type)` branch at the bottom of `format_expr`. The special case is removed.

### Constant — no changes

`Constant` stays as a dataclass. Regular constants (like `Index().constant(42)`) are unchanged. `ConstantOp(Op, Constant)` is unchanged.

## Scope

### In scope

- `Type` extends `Value[TypeType]` with lazy `.type` and `.__constant__`
- Remove `__init_subclass__`/`__post_init__`/`as_value`/`_type_to_json`
- Remove `isinstance(TypeType)` special cases in value.py and formatting.py
- Change `__params__` field type marker from `Type` to `TypeType`
- Update generator to emit `TypeType` in `__params__` and `Value[TypeType]` annotations
- Make `Value.ready` / `Constant.ready` properties instead of ClassVars
- Update parser to simplify `isinstance` checks

### Out of scope

- Changing `Constant` class structure
- `type` as honorary `__params__` on `Op`
- Memory equality fix for string-containing layouts
