# DGEN Generator Gap Analysis & Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the `.dgen` parser/generator with the design in `docs/dialect-files.md`, eliminating `Type`-as-wildcard and `list` kludge, and cleaning up the generator code.

**Architecture:** The `.dgen` format specifies types, ops, and traits. The parser produces an AST, and the Python generator emits dialect class definitions. The current implementation has several semantic gaps vs. the design doc.

**Tech Stack:** Python, pytest, ruff, ty

---

## Gap Analysis

### 1. `Type` used as "anything" wildcard

**Design intent (dialect-files.md):** `Type` is a *type value* parameter -- it means "this parameter is itself a type." E.g. `Array<element_type: Type, n: Index>` means `element_type` is a type and `n` is an index value.

**Current reality:** `Type` is used pervasively as a generic wildcard meaning "any value at all":
- `op transpose(input: Type) -> Type` -- `input` is not a type value, it's a tensor
- `op gep(base: Type, index: Type) -> Ptr` -- base and index are runtime values, not types
- `op load(ptr: Type) -> Float` -- ptr is a runtime value
- `OpDecl.return_type` defaults to `TypeRef("Type")` (ast.py:78)
- `_resolve_type_ref` returns `"Type"` for the `Type` name (python.py:84-85)

In the design, op operands don't need type annotations at all (they're runtime values -- their types come from the IR). The `Type` annotation should only appear on *parameters* (compile-time), and it specifically means "this parameter is a type."

**What needs to change:**
- Op operands should be untyped (or have optional constraint annotations for validation, not generation)
- `Type` in parameter position keeps its meaning: "this param is a type value"
- Op return types need a different model: either concrete type, or "inferred" (no default)
- The `_resolve_type_ref` function should not conflate `Type` with "anything"

### 2. `list` is a kludge not in the design

**Design doc:** No `list<T>` syntax. The design has `List<element_type: Type>` as a proper parameterized type, and uses `List<Type>` in type positions (e.g. `Function<args: List<Type>, result: Type>`).

**Current reality:** `list<T>` is a lowercase pseudo-type baked into the parser and generator:
- `list<Type>` appears in operand positions: `op pack(values: list<Type>)`
- `list<String>` appears in param positions: `op phi<labels: list<String>>`
- `_resolve_type_ref` special-cases `"list"` (python.py:87)
- `_annotation_for_param` special-cases `"list"` (python.py:92-94)
- `_annotation_for_operand` special-cases `"list"` (python.py:101-102)

**What needs to change:**
- Variadic operands (`values: list<Type>`) need a proper variadic marker in the AST, not a fake `list` type
- Variadic params (`labels: list<String>`) similarly need a variadic marker
- The design's `List<Type>` is a *real* parameterized type (already defined in builtin.dgen), distinct from "this operand accepts multiple values"

### 3. Operand type annotations are semantically wrong

**Design doc (lines 76-90):** Op signatures use `$X` metavariables for type inference/constraints, and bare names for concrete-typed operands. Operands are runtime values whose types are determined by the IR, not statically.

**Current reality:** Operands are annotated with `Type` to mean "any type", or with a concrete type name. But the generator ignores most of this -- `_annotation_for_operand` always returns `"Value"` or `"list[Value]"` regardless of the type annotation. The type annotation on operands currently serves only one purpose: populating `__operands__` metadata tuples.

**What needs to change (incremental):**
- Short term: operand type annotations become optional constraint hints (for `__operands__` metadata), not Python type annotations
- Longer term: support `$X` metavariable constraints as described in the design

### 4. Traits are stubs

**Design doc (lines 34-39):** Traits can have members (`static signed: Boolean`, `static bitwidth: Index`), and ops can `has trait` to inherit behavior.

**Current reality:** Traits are empty classes with `pass`. No member support. Only `HasSingleBlock` is used, and only via a hardcoded check in the generator.

### 5. `has trait` / `requires` not supported

**Design doc (lines 43-46, 77-81, 113-115):** Types and ops can declare `has trait Foo` and `requires $X ~= Pattern` constraints.

**Current reality:** Not parsed, not generated. Trait association is implicit via blocks + `HasSingleBlock`.

### 6. Type methods not supported

**Design doc (lines 60-64):** Types can have methods (`method num_elements(self) -> Index`).

**Current reality:** Not parsed, not generated. These are part of the staging model and not needed for current code generation.

### 7. Only single data field per type

**Current reality:** `_parse_type_body` stores only one `DataField`. The design shows types with multiple fields (e.g. Shape has `dims`, Tensor has `data`).

**Current workaround:** The `layout` keyword handles simple cases; complex layouts are monkey-patched.

---

## Implementation Plan

Priority order: fix the semantic issues first (Type wildcard, list kludge), then incremental improvements.

### Task 1: Add `Variadic` marker to AST, replace `list` kludge in parser

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Test: `test/test_gen_parser.py`

**Step 1: Write failing tests**

Add tests in `test/test_gen_parser.py`:

```python
def test_parse_variadic_operand():
    """list<Type> in operand position parses as variadic."""
    result = parse("op pack(values: list<Type>) -> List\n")
    op = result.ops[0]
    assert op.operands[0].variadic is True
    assert op.operands[0].type.name == "Type"


def test_parse_variadic_param():
    """list<String> in param position parses as variadic."""
    result = parse("op phi<labels: list<String>>(values: list<Type>) -> Type\n")
    op = result.ops[0]
    assert op.params[0].variadic is True
    assert op.params[0].type.name == "String"
    assert op.operands[0].variadic is True
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_parser.py::test_parse_variadic_operand test/test_gen_parser.py::test_parse_variadic_param -v`

**Step 3: Add `variadic` field to AST nodes**

In `dgen/gen/ast.py`, add `variadic: bool = False` to both `ParamDecl` and `OperandDecl`.

**Step 4: Update parser to unwrap `list<T>` into variadic**

In `dgen/gen/parser.py`, modify `_parse_params` and `_parse_operands`: if the parsed `TypeRef` has name `"list"`, set `variadic=True` and use the inner type.

**Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_parser.py -v`

**Step 6: Commit**

```bash
jj new -m "feat: add variadic marker to AST, replace list kludge in parser"
```

### Task 2: Update generator to use `variadic` instead of `list` special-casing

**Files:**
- Modify: `dgen/gen/python.py`
- Modify: `test/test_gen_python.py`

**Step 1: Update existing tests**

Change `test_generate_list_operand` and `test_generate_list_param` to construct AST nodes with `variadic=True` instead of `TypeRef("list", ...)`.

**Step 2: Remove `list` special-casing from generator**

In `dgen/gen/python.py`:
- `_resolve_type_ref`: remove the `if ref.name == "list"` branch
- `_annotation_for_param`: replace `list` check with `variadic` check on param
- `_annotation_for_operand`: replace `list` check with `variadic` check on operand
- Update `_generate` op generation to use `variadic` field

**Step 3: Run all tests**

Run: `python -m pytest . -q`

**Step 4: Commit**

```bash
jj new -m "refactor: generator uses variadic field instead of list special-casing"
```

### Task 3: Make op operand type annotations optional (default to untyped)

This is the first step toward fixing the `Type`-as-wildcard issue. Currently operands say `input: Type` to mean "any runtime value." The design says operands are just runtime values.

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Modify: `.dgen` files (optional, can defer)
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_parse_untyped_operand():
    """Operands without type annotation default to no constraint."""
    result = parse("op transpose(input) -> Type\n")
    op = result.ops[0]
    assert op.operands[0].name == "input"
    assert op.operands[0].type is None
```

**Step 2: Make `OperandDecl.type` optional**

In `ast.py`, change `type: TypeRef` to `type: TypeRef | None = None`.

**Step 3: Update parser to allow untyped operands**

In `_parse_operands`, if no `:` in the operand part, create `OperandDecl(name=..., type=None)`.

**Step 4: Update generator for `None` operand types**

In `python.py`, when operand type is `None`, emit `Value` annotation and `("name", Type)` in `__operands__` (same as current `Type` behavior, but now it's explicit fallback rather than semantic confusion).

**Step 5: Run all tests**

Run: `python -m pytest . -q`

**Step 6: Commit**

```bash
jj new -m "feat: support untyped operands in .dgen (operands are runtime values)"
```

### Task 4: Distinguish `Type` (type value) from untyped (any value) in return types

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

Currently `OpDecl.return_type` defaults to `TypeRef("Type")`. An op with `-> Type` means "the caller specifies the return type" while an op with `-> Nil` means "always returns Nil." But `Type` is doing double duty as both "type value" and "generic/inferred."

**Step 1: Introduce a sentinel for "inferred" return type**

Change `OpDecl.return_type` default from `TypeRef("Type")` to `None`, meaning "return type must be specified by the caller at construction time" (i.e., `type: Type` field with no default).

**Step 2: Update parser**

When `-> Type` is explicit in the source, keep `TypeRef("Type")`. When no `->` clause, set `return_type = None`.

**Step 3: Update generator**

- `return_type is None` or `return_type.name == "Type"`: emit `type: Type` (no default) -- same behavior
- `return_type` is concrete: emit `type: Type = ConcreteType()` (same as current)

This is a refactor that makes the AST more honest without changing generated output.

**Step 4: Run all tests**

Run: `python -m pytest . -q`

**Step 5: Commit**

```bash
jj new -m "refactor: distinguish None (inferred) from Type (explicit type value) in return types"
```

### Task 5: Generator code cleanup -- deduplicate `_parse_params` / `_parse_operands`

**Files:**
- Modify: `dgen/gen/parser.py`

**Step 1: Merge the two functions**

`_parse_params` and `_parse_operands` are nearly identical (lines 172-211). Extract a shared `_parse_decls` that returns `(name, type, default, variadic)` tuples, then wrap in `ParamDecl` or `OperandDecl`.

**Step 2: Run all tests**

Run: `python -m pytest . -q`

**Step 3: Commit**

```bash
jj new -m "refactor: deduplicate _parse_params and _parse_operands"
```

### Task 6: Support multiple data fields on types

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_parse_multiple_data_fields():
    src = "type Foo:\n    x: Index\n    y: Float64\n"
    result = parse(src)
    assert len(result.types[0].data) == 2
```

**Step 2: Change `TypeDecl.data` from `DataField | None` to `list[DataField]`**

Update AST, parser, and generator. The layout expression for multiple fields would need a struct/tuple layout (may defer the layout part and just support parsing for now).

**Step 3: Run all tests, fix breakage**

**Step 4: Commit**

```bash
jj new -m "feat: support multiple data fields on types"
```

### Task 7: Support `has trait` declarations in type and op bodies

**Files:**
- Modify: `dgen/gen/ast.py` (add `traits: list[str]` to `TypeDecl` and `OpDecl`)
- Modify: `dgen/gen/parser.py` (parse `has trait Foo` in body)
- Modify: `dgen/gen/python.py` (generate trait inheritance)
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_parse_has_trait():
    src = "type Float64:\n    has trait FloatingPoint\n"
    result = parse(src)
    assert "FloatingPoint" in result.types[0].traits
```

**Step 2: Implement parsing and generation**

Parse `has trait Name` lines in type/op bodies. In the generator, add trait names to the class bases.

**Step 3: Migrate `HasSingleBlock` detection from hardcoded to `has trait`**

Currently the generator checks `if "HasSingleBlock" in trait_names` combined with `od.blocks`. Instead, `.dgen` files should say `has trait HasSingleBlock` in op bodies that need it, and the generator should just use the declared traits.

**Step 4: Run all tests**

**Step 5: Commit**

```bash
jj new -m "feat: support 'has trait' declarations in .dgen type and op bodies"
```

### Task 8: Update `.dgen` files to use new features

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Modify: `dgen/dialects/llvm.dgen`
- Modify: `toy/dialects/affine.dgen`
- Modify: `toy/dialects/toy.dgen`

**Step 1: Replace `Type` wildcard operand annotations**

Change operands like `input: Type` to untyped `input`, or to a meaningful constraint type where appropriate. Keep `Type` only where it genuinely means "this is a type value parameter."

Example changes:
```dgen
# Before
op transpose(input: Type) -> Type

# After
op transpose(input) -> Type
```

**Step 2: Regenerate Python and verify round-trips**

Run: `python -m pytest . -q`

**Step 3: Commit**

```bash
jj new -m "refactor: update .dgen files to remove Type-as-wildcard from operands"
```

---

## Deferred / Future Work (not in this plan)

- **`$X` metavariable constraints** (`requires $X ~= Tensor`) -- substantial design work needed
- **Trait members** (`static signed: Boolean`) -- needs staging model integration
- **Type methods** (`method num_elements`) -- needs the function mini-language
- **`requires` validation clauses** -- needs expression parsing
- **Layout for types with math expressions** (e.g. `math.prod(shape.dims)`) -- needs expression language
