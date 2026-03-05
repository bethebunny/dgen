# Code Generator Simplification Design

**Goal:** Redesign `dgen/gen/python.py` to be ~130 lines with few to no special cases, following dgen principles: simple, predictable, composable.

**Current state:** ~445 lines with hardcoded exception lists (`_NAME_KEEP`, `_KNOWN_TRAITS`), three separate layout dicts, three layout resolution functions, fragile import tracking via substring matching.

---

## Naming

**Rule:** The Python class name for a type IS its `.dgen` CamelCase name. No suffix, no transformation, no exceptions.

- `type Index:` → `class Index(Type)`
- `type Nil:` → `class Nil(Type)`
- `type Shape<rank: Index>:` → `class Shape(Type)`
- `type Ptr:` → `class Ptr(Type)`

Ops keep the existing rule: `snake_case` → `CamelCaseOp`.

`.dgen` files with lowercase type names get renamed: `type index:` → `type Index:`, `type f64:` → `type F64:`.

Eliminates: `_NAME_KEEP`, `_type_class_name`, all suffix logic.

## Layout resolution

The generator hardcodes the full set of layout primitives and constructors. This is a fixed, small set — adding a new layout type is a structural change that warrants updating the generator.

```python
_LAYOUTS = {
    "Int": "layout.Int()",
    "Float64": "layout.Float64()",
    "Void": "layout.Void()",
    "Byte": "layout.Byte()",
    "String": "layout.String()",
    "Array": "layout.Array",
    "Pointer": "layout.Pointer",
    "FatPointer": "layout.FatPointer",
}
```

One recursive `_layout_expr` function resolves any data-field TypeRef:
- Parameter reference → `self.name.__layout__` (Type param) or `self.name.__constant__.to_json()` (value param)
- Name in `_LAYOUTS` → emit the expression, recurse into args for constructors
- Otherwise → error (invalid `.dgen`)

Layout imports: always `from dgen import layout`. No per-class import tracking. Just check whether any type has a data field.

Eliminates: `_TYPE_TO_LAYOUT`, `_COMPOUND_TO_LAYOUT`, `_LAYOUT_CONSTRUCTORS`, `_collect_layout_imports_from_type`, `_resolve_data_static`, `_resolve_data_parametric`, `_resolve_ref_parametric`, `_data_return_type`.

## `.dgen` syntax changes

1. **`layout` keyword** for primitive types: `layout Int` replaces `data: Index`
2. **CamelCase type names**: `type index:` → `type Index:`, `type f64:` → `type F64:`
3. Rename layout class `StringLayout` → `String` (in `dgen/layout.py`)

Example builtin.dgen after changes:
```dgen
type Index:
    layout Int

type F64:
    layout Float64

type Nil:
    layout Void

type String:
    layout String

type List<element_type: Type>:
    storage: FatPointer<element_type>
```

## Type expressions

One function resolves a concrete TypeRef to a Python construction expression:

```python
def _type_expr(ref: TypeRef, type_map: dict[str, TypeDecl]) -> str:
    td = type_map.get(ref.name)
    if not ref.args:
        if td is not None and td.params:
            raise ValueError(f"{ref.name} requires parameters")
        return f"{ref.name}()"
    if td is None:
        raise ValueError(f"unknown type {ref.name}")
    if len(ref.args) != len(td.params):
        raise ValueError(...)
    parts = []
    for arg, param in zip(ref.args, td.params):
        parts.append(f"{param.name}={param.type.name}().constant({arg.name})")
    return f"{ref.name}({', '.join(parts)})"
```

Used for return type defaults, parameter defaults, and any context where a TypeRef needs to become a Python expression. Callers check for `Type` (polymorphic) before calling.

Eliminates: `_type_default_expr`, `_type_has_no_required_params`, hardcoded `no_arg` set.

## Annotations

Staging determines annotation style:
- **Param** `foo: Index` → `foo: Value[Index]`
- **Param** `foo: list<String>` → `foo: list[Value[String]]`
- **Param** `foo: Type` → `foo: Type`
- **Operand** `foo: Type` → `foo: Value`
- **Operand** `foo: list<Type>` → `foo: list[Value]`

## Generator structure

Flat `generate()` function, no `_Generator` class. Lines appended to a list. ~130 lines total.

## Codebase rename

Mechanical find-and-replace dropping "Type" suffix from all generated type class names:

| Old | New |
|-----|-----|
| `IndexType` | `Index` |
| `F64Type` | `F64` |
| `ShapeType` | `Shape` |
| `MemRefType` | `MemRef` |
| `TensorType` | `Tensor` |
| `InferredShapeTensor` | `InferredShapeTensor` (no change) |
| `PtrType` | `Ptr` |
| `IntType` | `Int` |
| `FloatType` | `Float` |
| `VoidType` | `Void` |

All files importing these names update accordingly. Layout imports change from `from dgen.layout import Int, Void, ...` to `from dgen import layout` where needed to avoid name collisions.

## Followup (out of scope)

`Type` in `.dgen` currently serves as a polymorphic "any type" wildcard. Per `docs/dialect-files.md`, it should specifically refer to type values. This needs a dedicated pass to fix.
