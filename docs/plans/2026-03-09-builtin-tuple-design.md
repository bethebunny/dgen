# builtin.Tuple Design

## Goal

Add `Tuple` as a builtin type to replace remaining `list` special cases with a proper heterogeneous product type.

## .dgen definition

```
type Tuple<types: List<Type>>:
    layout Record
```

## Layout

`Tuple<[Index, String]>` → `Record([("0", Int()), ("1", String())])`.

The `__layout__` property iterates the types parameter, numbering fields `"0"`, `"1"`, etc., and resolves each type's layout.

## ASM — no special syntax

Uses existing list literal syntax for values:

```
%types : Tuple<[Type, Type]> = [Index, String]
%point : Tuple<[F64, F64]> = [1.0, 2.0]
%mixed : Tuple<[Index, String]> = [42, "hello"]
```

Types print in explicit form: `Tuple<[Index, String]>`.

`()` stays as Nil, unchanged.

## Code gen changes

- Add `"Record": "layout.Record"` to `_LAYOUTS` in `gen/python.py`
- Handle `layout Record` with a `List<Type>` parameter: generate a `__layout__` property that builds `Record([("0", layout0), ("1", layout1), ...])` from the types list

## Parser

Existing `_named_type` handles `Tuple<[Index, String]>` — the `[...]` parses via `value_expression` as a Python list, then `_coerce_param` wraps it. Need to ensure `_coerce_param` handles `List` (which has `__params__`) for a list of type values.

## What stays the same

- Nil unchanged
- `()` unchanged
- No tuple literal syntax
- No formatter changes

## Future

Tuple enables typing block arguments and call op inputs as single values, eventually removing the variadic `list` mechanism.
