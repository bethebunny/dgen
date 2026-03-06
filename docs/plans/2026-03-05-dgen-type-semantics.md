# DGEN Type Semantics Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Type` mean "type value" consistently in `.dgen` files. Add correct concrete types to operands and returns. Support bare `list` variadics, namespace imports, and `static` fields.

**Architecture:** `Type` always means "type value" — never "any value." Operands and returns have real types. When the type is concrete and known (e.g. `Float`, `Nil`, `Ptr`), write it. When the type is parameterized and the exact instantiation varies (e.g. `Tensor`), still write the base type name — the generator produces `type: Type` with no default for non-default-constructible types, same as for concrete types. This serves as documentation of what kind of value the op works with.

**Tech Stack:** Python, pytest, ruff, ty

---

## Analysis: Correct Types for Each Op

### Builtin ops

| Op | Operands | Return |
|---|---|---|
| `pack(values: list)` | variadic values | `-> List` (parameterized, no default) |
| `list_get<index: Index>(list: List)` | list: List | no return clause (element type varies) |
| `add_index(lhs: Index, rhs: Index)` | ✓ already correct | `-> Index` ✓ |
| `return(value = Nil)` | ✓ already correct | `-> Nil` ✓ |
| `function()` | none | `-> Function` (parameterized, no default) |

### LLVM ops (all concrete for this dialect)

| Op | Operands | Return |
|---|---|---|
| `alloca<elem_count: Index>()` | none | `-> Ptr` ✓ |
| `gep(base: Ptr, index: Int<64>)` | base: Ptr, index: Int<64> | `-> Ptr` ✓ |
| `load(ptr: Ptr)` | ptr: Ptr | `-> Float` ✓ |
| `store(value: Float, ptr: Ptr)` | value: Float, ptr: Ptr | `-> Nil` ✓ |
| `fadd(lhs: Float, rhs: Float)` | lhs: Float, rhs: Float | `-> Float` ✓ |
| `fmul(lhs: Float, rhs: Float)` | lhs: Float, rhs: Float | `-> Float` ✓ |
| `add(lhs: Int<64>, rhs: Int<64>)` | lhs: Int<64>, rhs: Int<64> | `-> Int<64>` ✓ |
| `mul(lhs: Int<64>, rhs: Int<64>)` | lhs: Int<64>, rhs: Int<64> | `-> Int<64>` ✓ |
| `icmp<pred: String>(lhs: Int<64>, rhs: Int<64>)` | both Int<64> | `-> Int<1>` ✓ |
| `br<dest: String>()` | none | `-> Nil` ✓ |
| `cond_br<true_dest: String, false_dest: String>(cond: Int<1>)` | cond: Int<1> | `-> Nil` ✓ |
| `label<label_name: String>()` | none | `-> Nil` ✓ |
| `phi<labels: list<String>>(values: list)` | variadic values | `-> Nil` ✓ (codegen infers actual type) |
| `fcmp<pred: String>(lhs: Float, rhs: Float)` | lhs: Float, rhs: Float | `-> Int<1>` ✓ |
| `zext(input: Int<1>)` | input: Int<1> | `-> Int<64>` ✓ |
| `call<callee: String>(args: list)` | variadic args | `-> Nil` ✓ |

### Affine ops

| Op | Operands | Return |
|---|---|---|
| `alloc(shape: Shape)` | shape: Shape ✓ | `-> MemRef` (parameterized, no default) |
| `dealloc(input: MemRef)` | input: MemRef | `-> Nil` ✓ |
| `load(memref: MemRef, indices: Index)` | memref: MemRef, indices: Index | `-> F64` ✓ |
| `store(value: F64, memref: MemRef, indices: Index)` | value: F64, memref: MemRef, indices: Index | `-> Nil` ✓ |
| `mul_f(lhs: F64, rhs: F64)` | lhs: F64, rhs: F64 | `-> F64` ✓ |
| `add_f(lhs: F64, rhs: F64)` | lhs: F64, rhs: F64 | `-> F64` ✓ |
| `print_memref(input: MemRef)` | input: MemRef | `-> Nil` ✓ |
| `for<lo: Index, hi: Index>()` | none | `-> Nil` ✓ |

### Toy ops

| Op | Operands | Return |
|---|---|---|
| `transpose(input: Tensor)` | input: Tensor | `-> Tensor` (parameterized, no default) |
| `reshape(input: Tensor)` | input: Tensor | `-> Tensor` |
| `mul(lhs: Tensor, rhs: Tensor)` | lhs: Tensor, rhs: Tensor | `-> Tensor` |
| `add(lhs: Tensor, rhs: Tensor)` | lhs: Tensor, rhs: Tensor | `-> Tensor` |
| `generic_call<callee: String>(args: list)` | variadic args | `-> Tensor` |
| `concat<axis: Index>(lhs: Tensor, rhs: Tensor)` | lhs: Tensor, rhs: Tensor | `-> Tensor` |
| `tile<count: Index>(input: Tensor)` | input: Tensor | `-> Tensor` |
| `nonzero_count(input: Tensor)` | input: Tensor | `-> Index` ✓ |
| `dim_size<axis: Index>(input: Tensor)` | input: Tensor | `-> Index` ✓ |
| `print(input: Tensor)` | input: Tensor | `-> Nil` ✓ |

---

## Implementation Tasks

### Task 1: Generator handles non-default-constructible return types gracefully

Currently `_type_expr` raises `ValueError` when a return type can't be default-constructed (e.g. `-> Tensor` where Tensor has required params). Change this to generate `type: Type` with no default, instead of erroring.

**Files:**
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_generate_op_parameterized_return_type():
    """Parameterized return type that can't be default-constructed generates no default."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Tensor",
                params=[ParamDecl(name="shape", type=TypeRef("Shape"))],
            )
        ],
        ops=[
            OpDecl(
                name="transpose",
                operands=[OperandDecl(name="input", type=TypeRef("Tensor"))],
                return_type=TypeRef("Tensor"),
            )
        ],
    )
    code = generate(f, dialect_name="test")
    assert "type: Type" in code
    assert "type: Type =" not in code  # no default, can't construct Tensor()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_gen_python.py::test_generate_op_parameterized_return_type -v`
Expected: FAIL with `ValueError: cannot default-construct Tensor`

**Step 3: Update generator**

In `_generate` (around line 240-244), catch ValueError from `_type_expr` and fall back to no-default:

```python
ret = od.return_type
if ret is None or ret.name == "Type":
    body.append("    type: Type")
else:
    try:
        body.append(f"    type: Type = {_type_expr(ret, type_map, known_names)}")
    except ValueError:
        body.append("    type: Type")
```

**Step 4: Run all tests**

Run: `python -m pytest . -q`

**Step 5: Commit**

```bash
jj new -m "fix: handle non-default-constructible return types gracefully in generator"
```

### Task 2: Support bare `list` for variadic operands

Currently variadic operands must spell `list<Type>`. Support bare `list` (no inner type).

**Files:**
- Modify: `dgen/gen/parser.py`
- Test: `test/test_gen_parser.py`

**Step 1: Write failing test**

```python
def test_parse_bare_list_operand():
    """Bare list (no inner type) in operand position parses as variadic."""
    result = parse("op pack(values: list)\n")
    op = result.ops[0]
    assert op.operands[0].variadic is True
    assert op.operands[0].type is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_gen_parser.py::test_parse_bare_list_operand -v`

**Step 3: Update `_parse_decl_parts`**

Change the variadic detection (around line 209) to also handle bare `list` (no args):

```python
if type_ref.name == "list":
    variadic = True
    if type_ref.args:
        type_ref = type_ref.args[0]
    else:
        type_ref = None
```

**Step 4: Run all tests**

Run: `python -m pytest . -q`

**Step 5: Commit**

```bash
jj new -m "feat: support bare list syntax for variadic operands"
```

### Task 3: Add correct types to all `.dgen` files

Add concrete operand types and return types based on the analysis above. Replace `-> Type` with correct return types. Replace untyped operands with correctly typed ones.

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Modify: `dgen/dialects/llvm.dgen`
- Modify: `toy/dialects/affine.dgen`
- Modify: `toy/dialects/toy.dgen`

**Step 1: Update `dgen/dialects/builtin.dgen`**

```dgen
trait HasSingleBlock

type Index:
    layout Int

type F64:
    layout Float64

type Nil:
    layout Void

type String:
    layout String

type Byte:
    layout Byte

type Array<element_type: Type, n: Index>:
    layout Array

type Pointer<pointee: Type>:
    layout Pointer

type FatPointer<pointee: Type>:
    layout FatPointer

type List<element_type: Type>:
    storage: FatPointer<element_type>

op pack(values: list) -> List
op list_get<index: Index>(list: List)
op add_index(lhs: Index, rhs: Index) -> Index
op return(value = Nil) -> Nil
op function():
    block body
    has trait HasSingleBlock
```

Changes: `pack` returns `-> List` (not `-> Type`), `list_get` has typed operand `list: List` and no return clause, `function` has no return clause (was `-> Type`), `pack` uses bare `list`.

**Step 2: Update `dgen/dialects/llvm.dgen`**

```dgen
from builtin import Index, Nil, F64, String, Pointer

type Ptr:
    data: Pointer<Nil>

type Int<bits: Index>:
    data: Index

type Float:
    data: F64

type Void:
    data: Nil

op alloca<elem_count: Index>() -> Ptr
op gep(base: Ptr, index: Int<64>) -> Ptr
op load(ptr: Ptr) -> Float
op store(value: Float, ptr: Ptr) -> Nil
op fadd(lhs: Float, rhs: Float) -> Float
op fmul(lhs: Float, rhs: Float) -> Float
op add(lhs: Int<64>, rhs: Int<64>) -> Int<64>
op mul(lhs: Int<64>, rhs: Int<64>) -> Int<64>
op icmp<pred: String>(lhs: Int<64>, rhs: Int<64>) -> Int<1>
op br<dest: String>() -> Nil
op cond_br<true_dest: String, false_dest: String>(cond: Int<1>) -> Nil
op label<label_name: String>() -> Nil
op phi<labels: list<String>>(values: list) -> Nil
op fcmp<pred: String>(lhs: Float, rhs: Float) -> Int<1>
op zext(input: Int<1>) -> Int<64>
op call<callee: String>(args: list) -> Nil
```

Changes: all operands now have correct concrete types, variadic operands use bare `list`.

**Step 3: Update `toy/dialects/affine.dgen`**

```dgen
from builtin import Index, Nil, F64, HasSingleBlock, Array, Pointer

type Shape<rank: Index>:
    dims: Array<Index, rank>

type MemRef<shape: Shape, dtype: Type = F64>:
    data: Pointer<Nil>

op alloc(shape: Shape) -> MemRef
op dealloc(input: MemRef) -> Nil
op load(memref: MemRef, indices: Index) -> F64
op store(value: F64, memref: MemRef, indices: Index) -> Nil
op mul_f(lhs: F64, rhs: F64) -> F64
op add_f(lhs: F64, rhs: F64) -> F64
op print_memref(input: MemRef) -> Nil
op for<lo: Index, hi: Index>() -> Nil:
    block body
    has trait HasSingleBlock
```

Changes: operands correctly typed (`MemRef`, `F64`, `Index`), `alloc` returns `-> MemRef` (parameterized, no default), `dealloc`/`print_memref` take `MemRef`.

**Step 4: Update `toy/dialects/toy.dgen`**

```dgen
from builtin import Index, Nil, F64, String
from affine import Shape

type Tensor<shape: Shape, dtype: Type = F64>:
    # TODO: layout requires math.prod of shape dims, not yet expressible in .dgen.
    # Temporary workaround: __layout__ monkey-patched in toy/dialects/__init__.py

type InferredShapeTensor<dtype: Type = F64>:
    data: Nil

op transpose(input: Tensor) -> Tensor
op reshape(input: Tensor) -> Tensor
op mul(lhs: Tensor, rhs: Tensor) -> Tensor
op add(lhs: Tensor, rhs: Tensor) -> Tensor
op generic_call<callee: String>(args: list) -> Tensor
op concat<axis: Index>(lhs: Tensor, rhs: Tensor) -> Tensor
op tile<count: Index>(input: Tensor) -> Tensor
op nonzero_count(input: Tensor) -> Index
op dim_size<axis: Index>(input: Tensor) -> Index
op print(input: Tensor) -> Nil
```

Changes: all operands typed as `Tensor`, return types are `-> Tensor` (parameterized, no default) where they were `-> Type`, variadic uses bare `list`.

**Step 5: Run all tests**

Run: `python -m pytest . -q`

**Step 6: Commit**

```bash
jj new -m "fix: add correct operand and return types to all .dgen files"
```

### Task 4: Namespace-qualified imports

Change from `from affine import Shape` to `import affine` with qualified references (`affine.Shape`). Per the design doc: "builtin types and ops are referenced without a prefix" (line 24), "other types and ops are imported by namespace, not directly" (line 25).

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Modify: `toy/dialects/toy.dgen`
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing parser test**

```python
def test_parse_namespace_import():
    result = parse("import affine\n")
    assert len(result.imports) == 1
    assert result.imports[0].module == "affine"
    assert result.imports[0].names == []


def test_parse_qualified_type_ref():
    result = parse("type Tensor<shape: affine.Shape>:\n    data: Nil\n")
    t = result.types[0]
    assert t.params[0].type.name == "affine.Shape"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_parser.py::test_parse_namespace_import -v`

**Step 3: Update parser**

In `_Parser.parse()`, add handling for bare `import` (no `from`):
```python
if line.startswith("import "):
    module = line.split()[1]
    result.imports.append(ImportDecl(module=module, names=[]))
```

`_parse_type_ref` already handles dotted names naturally — `affine.Shape` parses as `TypeRef(name="affine.Shape")` since there's no `<` in it.

**Step 4: Update generator**

In `_generate`, for imports with empty `names` (namespace import), generate:
```python
import toy.dialects.affine as affine
```

`_resolve_type_ref` already passes through names, so `affine.Shape` resolves to `"affine.Shape"`. The `known_names` set should add the namespace prefix. Update import processing to add `"affine"` as a namespace and trust qualified names.

For `_type_expr`, qualified names like `affine.MemRef` need to look up the type declaration. When the name contains `.`, split on the first `.` to get the namespace, and look up the short name in the type_map (which only contains types from the current file). For cross-module types, we can't default-construct anyway (we don't have the other file's AST), so fall back to no-default.

**Step 5: Update `toy/dialects/toy.dgen`**

```dgen
import affine

type Tensor<shape: affine.Shape, dtype: Type = F64>:
    # ...

type InferredShapeTensor<dtype: Type = F64>:
    data: Nil
```

Note: `from builtin import ...` stays — builtins are unqualified per the design doc.

**Step 6: Write generator test**

```python
def test_generate_namespace_import():
    f = DgenFile(
        imports=[ImportDecl(module="affine", names=[])],
    )
    code = generate(
        f,
        dialect_name="test",
        import_map={"affine": "toy.dialects.affine"},
    )
    assert "import toy.dialects.affine as affine" in code
```

**Step 7: Run all tests**

Run: `python -m pytest . -q`

**Step 8: Commit**

```bash
jj new -m "feat: support namespace-qualified imports (import module, module.Type)"
```

### Task 5: Static fields on trait declarations

Parse and generate `static name: Type [= default]` in trait bodies.

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_parse_trait_with_static_fields():
    src = "trait DType:\n    static signed: Boolean\n    static bitwidth: Index\n"
    result = parse(src)
    t = result.traits[0]
    assert t.name == "DType"
    assert len(t.statics) == 2
    assert t.statics[0].name == "signed"
    assert t.statics[0].type.name == "Boolean"
```

**Step 2: Add `StaticField` to AST, update `TraitDecl`**

```python
@dataclass
class StaticField:
    """A static field: static name: Type [= default]."""
    name: str
    type: TypeRef
    default: str | None = None

@dataclass
class TraitDecl:
    """A trait declaration with optional static fields."""
    name: str
    statics: list[StaticField] = field(default_factory=list)
```

**Step 3: Update parser — add `_parse_trait_body`**

Handle traits with colons (body follows) vs bare traits. Parse `static name: Type [= default]` lines.

**Step 4: Update generator**

```python
for trait in ast.traits:
    yield ""
    yield f"class {trait.name}:"
    if trait.statics:
        for sf in trait.statics:
            if sf.default:
                yield f"    {sf.name} = {sf.default}"
            else:
                yield f"    {sf.name}: {_resolve_type_ref(sf.type)}"
    else:
        yield "    pass"
    yield ""
```

**Step 5: Write generator test**

```python
def test_generate_trait_with_statics():
    from dgen.gen.ast import StaticField
    f = DgenFile(
        traits=[TraitDecl(name="DType", statics=[
            StaticField(name="signed", type=TypeRef("Boolean")),
            StaticField(name="bitwidth", type=TypeRef("Index")),
        ])]
    )
    code = generate(f, dialect_name="test")
    assert "class DType:" in code
    assert "signed: Boolean" in code
    assert "bitwidth: Index" in code
```

**Step 6: Run all tests**

Run: `python -m pytest . -q`

**Step 7: Commit**

```bash
jj new -m "feat: support static fields in trait declarations"
```

### Task 6: Static fields on type declarations

Parse and generate `static name: Type [= default]` in type bodies.

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_parser.py`, `test/test_gen_python.py`

**Step 1: Write failing test**

```python
def test_parse_type_with_static_fields():
    src = "type F64:\n    has trait FloatingPoint\n    static bitwidth: Index = 64\n"
    result = parse(src)
    t = result.types[0]
    assert t.traits == ["FloatingPoint"]
    assert len(t.statics) == 1
    assert t.statics[0].name == "bitwidth"
    assert t.statics[0].default == "64"
```

**Step 2: Add `statics` to `TypeDecl`**

Add `statics: list[StaticField] = field(default_factory=list)`.

**Step 3: Update parser**

Handle `static name: Type [= default]` in `_parse_type_body`. Update return tuple.

**Step 4: Update generator**

After layout/params, emit static fields:
```python
for sf in td.statics:
    if sf.default:
        body.append(f"    {sf.name} = {sf.default}")
    else:
        body.append(f"    {sf.name}: {_resolve_type_ref(sf.type)}")
```

**Step 5: Write generator test and run all tests**

**Step 6: Commit**

```bash
jj new -m "feat: support static fields in type declarations"
```

### Task 7: Document `Type` conventions

**Files:**
- Modify: `docs/dialect-files.md`

**Step 1: Add section after Conventions (line 26)**

```markdown
### Type semantics

`Type` always means **type value** — a value that is itself a type.

- **In parameters**: `element_type: Type` means this compile-time parameter holds a type.
- **In return position**: `-> Type` means the op returns a type value, not a runtime value.
- **Omitted return type**: The result type is supplied by the caller at construction time.
- **Concrete return type**: `-> Nil`, `-> Float` means the return type is always that type.
- **Parameterized return type**: `-> Tensor`, `-> MemRef` means the op returns that kind
  of value, but the caller must specify the exact parameterization.
- **Operand types**: Operands have types. Write `op add(lhs: Tensor, rhs: Tensor)`.
  Use the base type name even for parameterized types.
- **Variadic operands**: Bare `list` for variadic operands (`values: list`).
  For variadic params, use `list<T>` (`labels: list<String>`).
```

**Step 2: Commit**

```bash
jj new -m "docs: document Type semantics in dialect-files.md"
```

---

## Deferred / Future Work (not in this plan)

- **`$X` / `$Result` metavariable constraints** — for expressing relationships between operand and return types (e.g. `requires $X ~= Tensor`, `requires $Result.dtype == $X.dtype`)
- **Trait-as-constraint in parameters** (`dtype: DType` where `DType` is a trait) — once the constraint system exists
- **`requires` validation clauses** — needs expression parsing
- **Type methods** (`method num_elements(self) -> Index`) — needs the function mini-language
- **Layout for types with math expressions** (`math.prod(shape.dims)`) — needs expression language
