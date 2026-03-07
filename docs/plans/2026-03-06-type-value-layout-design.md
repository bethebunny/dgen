# Memory Representation for Type Values

## Problem

Type *instances* (e.g. the float `3.14` of type `f64`) have memory layouts via `__layout__`. But type *values* themselves (e.g. "the type `f64`" or "the type `Tensor<[2,3], f64>`") are Python objects with no binary representation. They can't flow through the JIT.

To enable staging and the dependent type system, the JIT needs to evaluate functions that accept and return types. This requires type values to have a concrete memory layout.

## Design

### Type value layout

A type value's memory layout is always a `Record`:

```
Record([
    ("tag", TypeTag),           # dialect-qualified name
    *(name, <param layout>)     # for each param in __params__
])
```

Where `<param layout>` is:
- **Value params** (e.g. `rank: Index`): the param type's `__layout__` (e.g. `Int`)
- **Type params** (e.g. `element_type: Type`): the concrete type's `type_layout` (recursive, inlined)

### TypeTag

A new builtin type `TypeTag` with `storage: String`. This wraps the dialect-qualified name (e.g. `"builtin.Index"`, `"affine.Shape"`) and provides a semantic distinction from raw strings.

### Implementation

A `type_layout` property on the `Type` base class, derived from existing `__params__`:

```python
class Type:
    @property
    def type_layout(self) -> layout.Record:
        fields: list[tuple[str, layout.Layout]] = [("tag", TypeTag.__layout__)]
        for name, _ in self.__params__:
            val = getattr(self, name)
            if isinstance(val, Type):
                fields.append((name, val.type_layout))
            else:
                fields.append((name, val.__constant__.type.__layout__))
        return layout.Record(fields)
```

No new fields or attributes needed beyond the `TypeTag` type.

### Deserialization

The format is self-describing:

1. Read the tag (first 16 bytes, always `String` = `FatPointer<Byte>`) -> e.g. `"builtin.Array"`
2. Look up the type constructor from the dialect registry -> know its `__params__`
3. Walk params left to right, advancing the offset:
   - If the param is `Type`-kinded: recurse to step 1
   - Otherwise: read using the param type's `__layout__`

Recursion bottoms out at non-parametric types (just a tag, no params).

### Examples

| Type value | Layout | Size |
|---|---|---|
| `Index()` | `Record([("tag", String)])` | 16 bytes |
| `Shape(rank=2)` | `Record([("tag", String), ("rank", Int)])` | 24 bytes |
| `Array(element_type=F64(), n=3)` | `Record([("tag", String), ("element_type", Record([("tag", String)])), ("n", Int)])` | 40 bytes |

### Design decisions

- **Qualified string tag** (not integer ID or hash): survives serialization, self-describing, human-readable. The dialect-qualified name (e.g. `"builtin.F64"`) avoids collisions across dialects.
- **Inline params** (not pointers): simpler, more compact. The type-value layout is only needed for concrete types (all params known), so variable size is fine. If polymorphic JIT functions are needed later, we can add indirection.
- **No new field**: derived from `__params__`, which already declares parameter names and types.

## Scope

### In scope
- `TypeTag` builtin type with `String` storage
- `type_layout` property on `Type` base class
- `type_layout` in .dgen and code generation
- Deserialization: reading type values back from buffers using the self-describing format

### Out of scope (future)
- Making `type: Type` an honorary `__params__` on `Op`
- SSA types flowing through the JIT
