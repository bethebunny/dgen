# ASM Literal Type Inference — Design Analysis

## Context

`_wrap_constant` in `dgen/asm/parser.py:322-329` infers type parameters from raw list literals by setting *every* param to `param_type().constant(len(raw))`. This only works for types whose params are all value-kinded scalars equal to the list length — today, just `Shape<rank: Index>`.

It breaks for:
- **Type-kinded params**: `StaticSpan<pointee: Type, n: Index>` — calls `TypeType().constant(3)`, nonsense
- **Multi-value params with different meanings**: hypothetical `Matrix<rows: Index, cols: Index>` — both get `len(raw)`
- **Params unrelated to list length**: any param whose semantics differ from "count of elements"

The xfail test at `test/test_layout.py:339` demonstrates the StaticSpan failure.

### Root cause

The **formatter** drops type information when emitting parameterized constants — it emits just the raw JSON literal (e.g. `[2, 3]`). The parser then must *reconstruct* the dropped type params from the literal alone. This is the fundamental design tension.

### Where `_wrap_constant` is called

1. **`_coerce_param`** — parsing type params in `_named_type`. E.g. `Tensor<[2, 3], F64>`: the `[2, 3]` occupies the `shape: Shape` slot, so `_wrap_constant(Shape, [2, 3])` must infer `rank=2`.
2. **`_coerce_operand`** — parsing op operands that are raw literals (less relevant; `ConstantOp` has an explicit pre-type).

### Future literal scenarios to consider

| Type | Literal | Inference needed |
|------|---------|-----------------|
| `Shape<rank: Index>` | `[2, 3]` | `rank = len(list)` — works today |
| `StaticSpan<pointee: Type, n: Index>` | `[10, 20, 30]` | `pointee = ???`, `n = len(list)` — broken |
| `Matrix<rows: Index, cols: Index>` | `[[1,2,3],[4,5,6]]` | `rows = len(outer)`, `cols = len(inner)` — wrong today (both get `len`) |
| `FixedString<n: Index>` | `"hello"` | `n = len(str)` — not a list, current code asserts list |
| `Dict<K: Type, V: Type>` | `{"a": 1}` | `K = String`, `V = Index` — completely un-inferrable from JSON |
| `BitVector<width: Index>` | `255` | `width = ???` — scalar, not a list at all |
| `Pair<A: Type, B: Type>` | `[x, y]` | `A`, `B` from element types — ambiguous |
| `Image<channels: Index, H: Index, W: Index>` | `[[[...]]]` | three value params from nested structure — fragile |

---

## Approach 1: No Inference — Require Explicit Type Annotations

**Rule**: Parameterized-type constants are always written `Type<params>(value)`. The formatter always emits this form. The parser never infers params from literals.

### Syntax

```
Tensor<Shape<2>([2, 3]), F64>           -- was: Tensor<[2, 3], F64>
Wrapper<StaticSpan<F64, 3>([10, 20, 30])>
Matrix<2, 3>([[1, 2, 3], [4, 5, 6]])
FixedString<5>("hello")
```

The `Type<params>(value)` form is parsed as: parse a type, then if `(` follows, parse a parenthesized constant value for that type.

### Implementation

**Formatter** (`formatting.py`): In `format_expr`, when formatting a `Constant` whose type has `__params__`, emit `type_asm(type)(format_expr(json))` instead of bare `format_expr(json)`.

**Parser** (`parser.py`): In `value_expression`, after parsing a named type via `_named_type`, if `(` follows, parse the parenthesized literal and return `type.constant(raw)`.

```python
# In value_expression, after _named_type returns:
type_val = _named_type(parser)
if parser.try_read("(") is not None:
    raw = value_expression(parser)
    parser.read(")")
    return type_val.constant(raw)
return type_val
```

**`_wrap_constant`**: Delete for parameterized types. Keep for non-parameterized types (e.g. bare `Index().constant(42)`).

### Trade-offs
- **Round-trip**: Correct by construction — no information lost
- **Simplicity**: Very simple parser/formatter changes
- **No special cases**: Uniform rule for all parameterized types
- **Extensibility**: Handles every future scenario in the table above
- **Downside**: More verbose ASM — `Tensor<Shape<2>([2, 3]), F64>` vs `Tensor<[2, 3], F64>`
- **Migration**: All existing IR strings with `Shape` literals change (test fixtures, testdata)

---

## Approach 2: Per-Type `from_literal` Hook

**Rule**: Types can define a `from_literal(cls, raw) -> Constant` classmethod. `_wrap_constant` dispatches to it.

```python
class Shape(Type):
    @classmethod
    def from_literal(cls, raw: object) -> Constant:
        assert isinstance(raw, list)
        return cls(rank=Index().constant(len(raw))).constant(raw)
```

### Trade-offs
- **Round-trip**: Works (formatter unchanged, hook reconstructs)
- **Violates "no special cases"**: Each type needs its own hook
- **Ambiguity**: For `StaticSpan`, inferring `pointee` from `[10, 20, 30]` — are those Index? F64? Byte?
- **Where do hooks live?**: Can't go in generated `.pyi` stubs. Needs new `.dgen` syntax or manual registration.
- **Extensibility**: Every new type needs a new hook

---

## Approach 3: Classify Params — Infer Value-Kinded, Require Annotation for Type-Kinded

**Rule**: Partition params into type-kinded and value-kinded (infrastructure for this exists in `dgen/gen/build.py:_is_type_kinded`). Value-kinded params are inferred from `len(raw)` as today. Type-kinded params must be explicit in ASM.

### Syntax options for the "partially explicit" form

A. **Hybrid**: `StaticSpan<F64>([10, 20, 30])` — supply type-kinded params explicitly, value-kinded inferred
B. **Full annotation when type-kinded present**: Fall back to Approach 1 only when type-kinded params exist

### Trade-offs
- **Preserves `Tensor<[2, 3], F64>`**: Shape has no type-kinded params, so inference still works
- **Two code paths**: Different behavior for types with/without type-kinded params
- **Doesn't fix multi-value-kinded**: `Matrix<rows, cols>` still broken (both get `len(raw)`)
- **Moderate complexity**: Need to partition params, format differently per category

---

## Approach 4: Structural Inference from Literal Shape

**Rule**: Instead of always using `len(raw)`, infer params from the *structure* of the literal: list length, nesting depth, element Python types, etc.

### Trade-offs
- **Fragile**: `[10, 20, 30]` — are elements int (Index)? Could be F64 with int-valued data
- **Not general**: Dicts, strings, scalars each need different structural analysis
- **Silent bugs**: Wrong inference produces valid but incorrect IR
- **Complexity**: Lots of heuristic code

---

## Recommendation: Approach 1 (No Inference)

**Approach 1 is the right design.** Rationale:

1. **Eliminates the problem entirely**: The formatter preserves all type information, so the parser has nothing to guess. No inference = no inference bugs.

2. **No special cases**: Single uniform rule — `Type<params>(value)` for all parameterized constants. Aligns with the project's "refuse to add special cases" principle.

3. **Simplifies the parser**: `_wrap_constant` for parameterized types can be deleted. The parser change is ~5 lines (check for `(` after a parsed type).

4. **Self-documenting ASM**: `StaticSpan<F64, 3>([10, 20, 30])` is more readable than a bare `[10, 20, 30]` where the reader must know the enclosing type to understand the value.

5. **Handles all future scenarios**: Matrix, FixedString, Dict, BitVector, Pair — all work with zero additional code.

6. **Round-trip correct by construction**: formatter emits exactly what parser expects.

The verbosity cost (`Shape<2>([2, 3])` vs `[2, 3]`) is real but acceptable — IR text is primarily machine-generated and machine-consumed. Clarity and correctness matter more than brevity.

### Optional: Parse-only backward compat for bare literals

We *could* keep `_wrap_constant` as a parse-only path for types with exclusively value-kinded params (like `Shape`), so old IR strings like `Tensor<[2, 3], F64>` still parse. The formatter would emit the new form. This eases migration but adds a second code path. I'd lean toward a clean break (no backward compat) since all IR strings are in test fixtures that can be regenerated.

---

## Implementation plan

### Files to modify

| File | Change |
|------|--------|
| `dgen/asm/parser.py` | Add `Type<params>(literal)` production to `value_expression`; simplify `_wrap_constant` |
| `dgen/asm/formatting.py` | `format_expr`: emit `type_asm(type)(json)` for parameterized `Constant` |
| `test/test_layout.py` | Remove `xfail` from `test_parse_type_with_static_span_param`, update expected IR |
| Various test files | Update IR string fixtures to use new format |

### Steps

1. **Parser**: Add typed-constant production to `value_expression` — after `_named_type` returns a `Type`, check for `(` and parse `Type<params>(literal)` → `Constant`.
2. **Formatter**: In `format_expr`, when value is a `Constant` with a parameterized type, emit `type_asm(type)(format_expr(json))`.
3. **Simplify `_wrap_constant`**: Remove the parameterized-type branch. Keep the non-parameterized path for bare scalars.
4. **Fix tests**: Run `pytest`, update all IR string fixtures to match new formatting.
5. **Remove xfail**: The StaticSpan test should now pass.

### Verification

```bash
pytest . -q                    # All tests pass
ruff format && ruff check --fix  # Clean
```

Specifically verify:
- `Tensor<Shape<2>([2, 3]), F64>` round-trips correctly
- `StaticSpan<F64, 3>([10, 20, 30])` round-trips correctly (previously xfail)
- All existing end-to-end tests pass with updated IR strings
