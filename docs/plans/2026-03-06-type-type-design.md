# TypeType: Types as Values

## Problem

Phase 1 established `type_layout`, `type_to_json`, and `type_from_json` — but type values still can't flow through `Memory` uniformly. The `type_layout` property lives on `Type` but there's no registered metatype whose `__layout__` delegates to it. Type-kinded params are still bare `Type` instances, not `Constant` values like everything else.

## Core Insight

A type is a value of type `TypeType`. This is consistent with how `__layout__` works everywhere: `Index.__layout__` describes how Index instances are stored, so `TypeType.__layout__` describes how TypeType instances (i.e. type values) are stored.

## Design

### TypeType

A parameterized builtin type:

```python
@builtin.type("TypeType")
@dataclass(frozen=True)
class TypeType(Type):
    concrete: Type
    __params__ = (("concrete", Type),)

    @property
    def __layout__(self) -> Layout:
        return self.concrete.type_layout
```

This makes type values work with `Memory` uniformly:

```python
Memory(type=Index())                         # .__layout__ -> Int()
Memory(type=TypeType(concrete=Index()))       # .__layout__ -> Record([("tag", String())])
Memory(type=TypeType(concrete=List(Index()))) # .__layout__ -> Record([("tag", ..."), ...])
```

### Fix `__layout__` annotation

Change from `ClassVar[Layout]` to just `Layout` on `Type`. Parametric types already use `@property`, proving `ClassVar` is wrong.

## Design Decisions

- **`TypeType` is parameterized on `concrete`**: the layout varies per concrete type, so the metatype must know which type it describes. Same pattern as `Pointer(pointee=F64())`.
- **`__layout__` is instance-level, not ClassVar**: `Tensor` and `Array` already prove this.

## Scope

### In scope
- `TypeType` registered type in builtin.dgen and builtin.py
- `TypeType.__layout__` delegates to `concrete.type_layout`
- Fix `__layout__` annotation from `ClassVar` to instance-level
- Type values flow through `Memory` via `TypeType`
- Round-trip tests through Memory

### Out of scope (future)
- Type-kinded params stored as `Constant[TypeType]` (uniform with value params)
- `type: Type` as honorary `__params__` on `Op`
- SSA types flowing through the JIT
