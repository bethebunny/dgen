# Code Generator Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign `dgen/gen/python.py` from ~445 lines with hardcoded exception lists to ~130 lines with no special cases, and rename all generated type classes to drop the "Type" suffix.

**Architecture:** Add `layout` keyword to parser for primitive types. Rewrite python.py as a flat `generate()` function with one `_LAYOUTS` dict, one `_layout_expr()`, one `_type_expr()`. Rename `StringLayout` → `layout.String`. Update all `.dgen` files to CamelCase names. Mechanically rename all type class references across the codebase.

**Tech Stack:** Python, pytest, ruff, jj

**Design doc:** `docs/plans/2026-03-05-codegen-simplification-design.md`

---

### Task 1: Add `layout` keyword to parser

Support `layout Int` as a type body line alongside `data: X` / `storage: X` fields.

**Files:**
- Modify: `dgen/gen/ast.py`
- Modify: `dgen/gen/parser.py`
- Test: `test/test_gen_python.py`

**Step 1: Add `layout` field to `TypeDecl` in ast.py**

In `dgen/gen/ast.py`, add an optional `layout` field to `TypeDecl`:

```python
@dataclass
class TypeDecl:
    """A type declaration with optional params and data fields."""

    name: str
    params: list[ParamDecl] = field(default_factory=list)
    data: DataField | None = None
    layout: str | None = None
```

**Step 2: Parse `layout` keyword in parser.py**

In `dgen/gen/parser.py`, update `_parse_type_body` to handle `layout X` lines:

```python
def _parse_type_body(self) -> tuple[DataField | None, str | None]:
    """Parse indented type body lines, return (data field, layout name)."""
    data = None
    layout = None
    while self.pos + 1 < len(self.lines):
        next_line = self.lines[self.pos + 1]
        if not next_line or not next_line[0].isspace():
            break
        self.pos += 1
        stripped = next_line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith("layout "):
            layout = stripped.split()[1]
            continue
        # Field declaration: name: TypeExpr
        if ":" in stripped:
            colon = stripped.index(":")
            field_name = stripped[:colon].strip()
            type_str = stripped[colon + 1 :].strip()
            data = DataField(name=field_name, type=_parse_type_ref(type_str))
    return data, layout
```

Update `_parse_type` to unpack the tuple:

```python
data, layout = self._parse_type_body() if has_body else (None, None)
return TypeDecl(name=name, params=params, data=data, layout=layout)
```

**Step 3: Write a test for the layout keyword**

Add to `test/test_gen_python.py`:

```python
def test_parse_layout_keyword():
    from dgen.gen.parser import parse
    f = parse("type Index:\n    layout Int\n")
    assert len(f.types) == 1
    assert f.types[0].name == "Index"
    assert f.types[0].layout == "Int"
    assert f.types[0].data is None
```

**Step 4: Run tests**

Run: `python -m pytest test/test_gen_python.py -q`
Expected: All pass (new test included)

**Step 5: Commit**

```
jj new -m "feat: add layout keyword to .dgen parser"
```

---

### Task 2: Rename `StringLayout` → `String` in layout.py

**Files:**
- Modify: `dgen/layout.py:175-189`

**Step 1: Rename the class and add backward-compat alias**

In `dgen/layout.py`, rename class `StringLayout` to `String` and add a temporary alias:

```python
class String(FatPointer):
    """FatPointer(Byte()) with str ↔ list[int] conversion in to_json/from_json."""
    # ... (same body)

# Temporary alias — remove after all consumers updated
StringLayout = String
```

**Step 2: Run tests**

Run: `python -m pytest . -q`
Expected: All 187 tests pass (alias keeps everything working)

**Step 3: Commit**

```
jj new -m "refactor: rename StringLayout to String in layout.py"
```

---

### Task 3: Update `.dgen` files

Rename lowercase type names to CamelCase and replace `data` fields with `layout` keyword for primitive types.

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Modify: `dgen/dialects/llvm.dgen`
- Modify: `toy/dialects/affine.dgen`
- Modify: `toy/dialects/toy.dgen`

**Step 1: Update `builtin.dgen`**

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

type List<element_type: Type>:
    storage: FatPointer<element_type>

op pack(values: list<Type>) -> List
op list_get<index: Index>(list: List) -> Type
op add_index(lhs: Index, rhs: Index) -> Index
op return(value: Type = Nil) -> Nil
op function() -> Function:
    block body
```

**Step 2: Update `llvm.dgen`**

Change import to match new CamelCase names:

```dgen
from builtin import Index, Nil, F64, String

type Ptr:
    layout Pointer

type Int<bits: Index>:
    layout Int

type Float:
    layout Float64

type Void:
    layout Void

op alloca<elem_count: Index>() -> Ptr
op gep(base: Type, index: Type) -> Ptr
op load(ptr: Type) -> Float
op store(value: Type, ptr: Type) -> Nil
op fadd(lhs: Type, rhs: Type) -> Float
op fmul(lhs: Type, rhs: Type) -> Float
op add(lhs: Type, rhs: Type) -> Int<64>
op mul(lhs: Type, rhs: Type) -> Int<64>
op icmp<pred: String>(lhs: Type, rhs: Type) -> Int<1>
op br<dest: String>() -> Nil
op cond_br<true_dest: String, false_dest: String>(cond: Type) -> Nil
op label<label_name: String>() -> Nil
op phi<labels: list<String>>(values: list<Type>) -> Nil
op fcmp<pred: String>(lhs: Type, rhs: Type) -> Int<1>
op zext(input: Type) -> Int<64>
op call<callee: String>(args: list<Type>) -> Nil
```

Note: `Ptr` layout changes from `data: Pointer<Nil>` to `layout Pointer` — this is because the `layout` keyword handles the full layout directly. The Pointer layout defaults to `Pointer(Void())` — handle this in the generator by hardcoding `"Pointer": "layout.Pointer(layout.Void())"` in `_LAYOUTS`.

**Step 3: Update `affine.dgen`**

```dgen
from builtin import Index, Nil, F64, HasSingleBlock

type Shape<rank: Index>:
    dims: Array<Index, rank>

type MemRef<shape: Shape, dtype: Type = F64>:
    data: Pointer<Nil>

op alloc(shape: Shape) -> Type
op dealloc(input: Type) -> Nil
op load(memref: Type, indices: Index) -> F64
op store(value: Type, memref: Type, indices: Index) -> Nil
op mul_f(lhs: Type, rhs: Type) -> F64
op add_f(lhs: Type, rhs: Type) -> F64
op print_memref(input: Type) -> Nil
op for<lo: Index, hi: Index>() -> Nil:
    block body
```

No changes needed — already CamelCase names, no `layout` keyword applicable (Shape has parametric layout, MemRef uses data field).

**Step 4: Update `toy.dgen`**

```dgen
from builtin import Index, Nil, F64, String
from affine import Shape

type Tensor<shape: Shape, dtype: Type = F64>:
    # Layout computed externally (needs math.prod)

type InferredShapeTensor<dtype: Type = F64>:
    data: Nil

op transpose(input: Type) -> Type
op reshape(input: Type) -> Type
op mul(lhs: Type, rhs: Type) -> Type
op add(lhs: Type, rhs: Type) -> Type
op generic_call<callee: String>(args: list<Type>) -> Type
op concat<axis: Index>(lhs: Type, rhs: Type) -> Type
op tile<count: Index>(input: Type) -> Type
op nonzero_count(input: Type) -> Index
op dim_size<axis: Index>(input: Type) -> Index
op print(input: Type) -> Nil
```

No changes needed — already CamelCase, no `layout` keyword applicable.

**Step 5: Commit**

```
jj new -m "refactor: update .dgen files to CamelCase names and layout keyword"
```

---

### Task 4: Rewrite `dgen/gen/python.py`

This is the core task. Replace the 445-line `_Generator` class with a ~130-line flat `generate()` function.

**Files:**
- Rewrite: `dgen/gen/python.py`

**Step 1: Write the new python.py**

```python
"""Python code generator for .dgen dialect specifications."""

from __future__ import annotations

from dgen.gen.ast import (
    DgenFile,
    OpDecl,
    OperandDecl,
    ParamDecl,
    TypeDecl,
    TypeRef,
)

# Layout primitives: name -> expression using `layout.` prefix.
# Leaf layouts are complete expressions; constructors take arguments.
_LAYOUTS: dict[str, str] = {
    "Int": "layout.Int()",
    "Float64": "layout.Float64()",
    "Void": "layout.Void()",
    "Byte": "layout.Byte()",
    "String": "layout.String()",
    "Pointer": "layout.Pointer",
    "Array": "layout.Array",
    "FatPointer": "layout.FatPointer",
}


def _type_class_name(name: str) -> str:
    """Python class name for a .dgen type. Identity — the name IS the class name."""
    return name


def _op_class_name(asm_name: str) -> str:
    """Derive Python class name from ASM op name."""
    parts = asm_name.split("_")
    return "".join(p.capitalize() for p in parts) + "Op"


def _type_expr(ref: TypeRef, type_map: dict[str, TypeDecl]) -> str:
    """Resolve a TypeRef to a Python construction expression.

    Used for return type defaults and anywhere a TypeRef becomes a Python expression.
    Caller must check for `Type` (polymorphic) before calling.
    """
    td = type_map.get(ref.name)
    if not ref.args:
        if td is not None and td.params:
            raise ValueError(f"{ref.name} requires parameters")
        return f"{ref.name}()"
    if td is None:
        raise ValueError(f"unknown parameterized type {ref.name}")
    if len(ref.args) != len(td.params):
        raise ValueError(
            f"{ref.name} expects {len(td.params)} args, got {len(ref.args)}"
        )
    parts = []
    for arg, param in zip(ref.args, td.params):
        parts.append(f"{param.name}={param.type.name}().constant({arg.name})")
    return f"{ref.name}({', '.join(parts)})"


def _layout_expr(ref: TypeRef, param_map: dict[str, ParamDecl]) -> str:
    """Resolve a data-field TypeRef to a layout expression.

    - Parameter reference with Type kind → self.name.__layout__
    - Parameter reference with value kind → self.name.__constant__.to_json()
    - Name in _LAYOUTS (leaf) → the expression directly
    - Name in _LAYOUTS (constructor) → constructor(recurse args)
    - Otherwise → error
    """
    if ref.name in param_map:
        p = param_map[ref.name]
        if p.type.name == "Type":
            return f"self.{ref.name}.__layout__"
        return f"self.{ref.name}.__constant__.to_json()"
    entry = _LAYOUTS.get(ref.name)
    if entry is None:
        raise ValueError(f"unknown layout type {ref.name!r}")
    if entry.endswith(")"):
        # Leaf layout (no args expected)
        return entry
    # Constructor — recurse into args
    args = ", ".join(_layout_expr(a, param_map) for a in ref.args)
    return f"{entry}({args})"


def _resolve_type_ref(ref: TypeRef) -> str:
    """Resolve a TypeRef to a Python class name for __params__/__operands__."""
    if ref.name == "Type":
        return "Type"
    if ref.name == "list":
        return _resolve_type_ref(ref.args[0]) if ref.args else "Type"
    return ref.name


def _annotation_for_param(param: ParamDecl) -> str:
    """Python type annotation for a compile-time parameter."""
    if param.type.name == "list":
        inner = _resolve_type_ref(param.type.args[0]) if param.type.args else "Type"
        return f"list[Value[{inner}]]"
    if param.type.name == "Type":
        return "Type"
    return f"Value[{param.type.name}]"


def _annotation_for_operand(operand: OperandDecl) -> str:
    """Python type annotation for a runtime operand."""
    if operand.type.name == "list":
        return "list[Value]"
    return "Value"


def generate(
    ast: DgenFile,
    dialect_name: str,
    import_map: dict[str, str] | None = None,
) -> str:
    """Generate Python source code from a DgenFile AST."""
    import_map = import_map or {}
    lines: list[str] = []
    type_map = {td.name: td for td in ast.types}
    needs_block = any(od.blocks for od in ast.ops)
    needs_layout = any(td.data or td.layout for td in ast.types)
    trait_names = {t.name for t in ast.traits}
    for imp in ast.imports:
        for name in imp.names:
            if name in trait_names:
                pass  # already known
            # Check imported types to populate type_map for cross-module defaults
            # (imported types aren't in ast.types, so _type_expr checks them)

    # Header
    lines.append(f"# GENERATED by dgen from {dialect_name}.dgen — do not edit.")
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("")

    # Framework imports
    dgen_names = ["Dialect", "Op", "Type", "Value"]
    if needs_block:
        dgen_names.insert(0, "Block")
    if needs_layout:
        dgen_names.append("layout")
    lines.append(f"from dgen import {', '.join(sorted(dgen_names))}")

    # Cross-dialect imports
    for imp in ast.imports:
        python_module = import_map.get(imp.module)
        if python_module:
            lines.append(f"from {python_module} import {', '.join(imp.names)}")

    lines.append("")
    lines.append(f'{dialect_name} = Dialect("{dialect_name}")')
    lines.append("")

    # Traits
    for trait in ast.traits:
        lines.append("")
        lines.append(f"class {trait.name}:")
        lines.append("    pass")
        lines.append("")

    # Types
    for td in ast.types:
        param_map = {p.name: p for p in td.params}
        is_parametric = td.data is not None and any(
            _ref_has_params(td.data.type, param_map) for _ in [None]
        )

        lines.append("")
        lines.append(f'@{dialect_name}.type("{td.name}")')
        lines.append("@dataclass(frozen=True)")
        lines.append(f"class {td.name}(Type):")

        body: list[str] = []

        # Layout from keyword
        if td.layout:
            entry = _LAYOUTS.get(td.layout)
            if entry is None:
                raise ValueError(f"unknown layout {td.layout!r}")
            if entry.endswith(")"):
                body.append(f"    __layout__ = {entry}")
            else:
                # Constructor without args (e.g., Pointer) → default to Void
                body.append(f"    __layout__ = {entry}(layout.Void())")

        # Static layout from data field
        elif td.data and not is_parametric:
            body.append(
                f"    __layout__ = {_layout_expr(td.data.type, param_map)}"
            )

        # Parameters
        for p in td.params:
            ann = _annotation_for_param(p)
            if p.default:
                body.append(f"    {p.name}: {ann} = {p.default}()")
            else:
                body.append(f"    {p.name}: {ann}")

        if td.params:
            parts = [f'("{p.name}", {_resolve_type_ref(p.type)})' for p in td.params]
            body.append(f"    __params__ = ({', '.join(parts)},)")

        # Parametric layout property from data field
        if td.data and is_parametric:
            body.append("")
            body.append("    @property")
            body.append(f"    def __layout__(self) -> layout.Layout:")
            body.append(
                f"        return {_layout_expr(td.data.type, param_map)}"
            )

        if not body:
            body.append("    pass")
        lines.extend(body)
        lines.append("")

    # Ops
    for od in ast.ops:
        has_trait = bool(od.blocks) and any(
            name in trait_names or any(name in imp.names for imp in ast.imports)
            for name in ["HasSingleBlock"]
        )

        lines.append("")
        lines.append(f'@{dialect_name}.op("{od.name}")')
        lines.append("@dataclass(eq=False, kw_only=True)")
        bases = "HasSingleBlock, Op" if has_trait else "Op"
        lines.append(f"class {_op_class_name(od.name)}({bases}):")

        body = []

        # Params
        for p in od.params:
            body.append(f"    {p.name}: {_annotation_for_param(p)}")

        # Operands
        for op in od.operands:
            ann = _annotation_for_operand(op)
            if op.default:
                body.append(f"    {op.name}: {ann} | {op.default} = {op.default}()")
            else:
                body.append(f"    {op.name}: {ann}")

        # Return type
        ret = od.return_type
        if ret.name == "Type":
            body.append("    type: Type")
        else:
            # Try to build a default expression. For imported types not in type_map,
            # simple no-arg types just become Name().
            expr = _type_expr(ret, type_map)
            body.append(f"    type: Type = {expr}")

        # Blocks
        for block_name in od.blocks:
            body.append(f"    {block_name}: Block")

        # __params__
        if od.params:
            parts = [f'("{p.name}", {_resolve_type_ref(p.type)})' for p in od.params]
            body.append(f"    __params__ = ({', '.join(parts)},)")

        # __operands__
        if od.operands:
            parts = [
                f'("{op.name}", {_resolve_type_ref(op.type)})' for op in od.operands
            ]
            body.append(f"    __operands__ = ({', '.join(parts)},)")

        # __blocks__
        if od.blocks:
            parts = [f'"{b}"' for b in od.blocks]
            body.append(f"    __blocks__ = ({', '.join(parts)},)")

        lines.extend(body)
        lines.append("")

    return "\n".join(lines) + "\n"


def _ref_has_params(ref: TypeRef, param_map: dict[str, ParamDecl]) -> bool:
    """Check if a TypeRef references any parameters."""
    if ref.name in param_map:
        return True
    return any(_ref_has_params(arg, param_map) for arg in ref.args)
```

**Step 2: Run the generator tests**

Run: `python -m pytest test/test_gen_python.py -q`
Expected: Several failures (tests still expect old names like `IndexType`, `ShapeType`, `MemRefType`). This is expected — we fix these in the next task.

**Step 3: Commit**

```
jj new -m "refactor: rewrite python.py as flat ~130-line generator"
```

---

### Task 5: Update generator tests

Update `test/test_gen_python.py` to expect the new naming (no "Type" suffix, `from dgen import layout` instead of individual layout imports, import names passed through as-is).

**Files:**
- Modify: `test/test_gen_python.py`

**Step 1: Update all assertions**

Key changes in test expectations:
- `"class IndexType(Type):"` → `"class Index(Type):"`  (but wait — the test creates `TypeDecl(name="index")`, which under old rules became `IndexType`. Under new rules, the class name IS the .dgen name. So `name="index"` → `class index(Type):`. But per the design, we're renaming .dgen types to CamelCase. The tests construct ASTs directly, so update the test AST names too.)
- Update test AST `TypeDecl(name="index")` → `TypeDecl(name="Index")`
- `"class ShapeType(Type):"` → `"class Shape(Type):"`  — update `TypeDecl(name="Shape")` stays the same
- `from dgen.layout import Int` → no longer in output; instead `from dgen import ... layout`
- `"__layout__ = Int()"` → `"__layout__ = layout.Int()"`
- `"class MemRefType(Type):"` → `"class MemRef(Type):"`
- `"Value[IndexType]"` → `"Value[Index]"` — but the param type in tests is `TypeRef("Index")`, so annotation becomes `Value[Index]`
- `"F64Type()"` → `"F64()"` — but this was a default, and it comes from `ParamDecl(default="F64")`, so the new code emits `F64()`
- Import test: `"from dgen.dialects.builtin import IndexType, Nil"` → `"from dgen.dialects.builtin import Index, Nil"` — but the new code passes import names through as-is from the .dgen file, so `ImportDecl(names=["Index", "Nil"])` → `from dgen.dialects.builtin import Index, Nil`

Full list of test updates needed — update each `assert` to match the new output format.

Also add a test for the `layout` keyword generating layout expressions.

**Step 2: Run tests**

Run: `python -m pytest test/test_gen_python.py -q`
Expected: All pass

**Step 3: Commit**

```
jj new -m "test: update generator tests for new naming and layout conventions"
```

---

### Task 6: Regenerate all dialect files

Run the generator to produce new `.py` files from the updated `.dgen` files.

**Files:**
- Regenerate: `dgen/dialects/builtin.py`
- Regenerate: `dgen/dialects/llvm.py`
- Regenerate: `toy/dialects/affine.py`
- Regenerate: `toy/dialects/toy.py`

**Step 1: Regenerate builtin.py**

```bash
python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py
```

Verify the output looks correct — class names should be `Index`, `F64`, `Nil`, `String`, `List`. Layout expressions should use `layout.Int()`, `layout.Void()`, etc.

**Step 2: Regenerate llvm.py**

```bash
python -m dgen.gen dgen/dialects/llvm.dgen -I builtin=dgen.dialects.builtin > dgen/dialects/llvm.py
```

**Step 3: Regenerate affine.py**

```bash
python -m dgen.gen toy/dialects/affine.dgen -I builtin=dgen.dialects.builtin > toy/dialects/affine.py
```

**Step 4: Regenerate toy.py**

```bash
python -m dgen.gen toy/dialects/toy.dgen -I builtin=dgen.dialects.builtin -I affine=toy.dialects.affine > toy/dialects/toy.py
```

**Step 5: Verify generated files**

Read each generated file and confirm:
- Class names have no "Type" suffix
- Layout uses `layout.Int()` not `Int()`
- Import line is `from dgen import Dialect, Op, Type, Value, layout` (sorted)
- Cross-dialect imports use bare names: `from dgen.dialects.builtin import Index, Nil, F64, String`

**Step 6: Commit**

```
jj new -m "refactor: regenerate all dialect files with new naming"
```

---

### Task 7: Mechanical rename across consumer files

Find-and-replace old type class names with new ones across all non-generated files.

**Files (non-exhaustive — search for each name):**
- Modify: `dgen/codegen.py`
- Modify: `dgen/module.py`
- Modify: `toy/dialects/__init__.py`
- Modify: `toy/passes/toy_to_affine.py`
- Modify: `toy/passes/affine_to_llvm.py`
- Modify: `toy/passes/shape_inference.py`
- Modify: `toy/passes/optimize.py`
- Modify: `toy/parser/lowering.py`
- Modify: `toy/cli.py`
- Modify: `toy/test/test_type_roundtrip.py`
- Modify: `toy/test/test_layout.py`
- Modify: `toy/test/test_toy_printer.py`
- Modify: `toy/test/test_end_to_end.py`
- Modify: `toy/test/test_affine_to_llvm.py`

**Step 1: Rename type classes**

Apply these replacements in each file (use search-and-replace):

| Old | New |
|-----|-----|
| `IndexType` | `Index` |
| `F64Type` | `F64` |
| `ShapeType` | `Shape` |
| `MemRefType` | `MemRef` |
| `TensorType` | `Tensor` |
| `PtrType` | `Ptr` |
| `IntType` | `Int` |
| `FloatType` | `Float` |
| `VoidType` | `Void` |

**Collision avoidance:** In files that also import from `dgen.layout`, change those imports from `from dgen.layout import Int, Void, ...` to `from dgen import layout` and prefix all layout references with `layout.`. This is needed in:
- `dgen/module.py` (imports `Void` from layout)
- `toy/dialects/__init__.py` (imports `Array, Float64, Layout` from layout)
- `toy/passes/toy_to_affine.py` (imports `Array` from layout)
- `toy/passes/affine_to_llvm.py` (imports `Array` from layout)
- `toy/test/test_layout.py` (imports many layout types)

**Important:** `builtin.IndexType` → `builtin.Index`, `builtin.F64Type` → `builtin.F64`, `llvm.IntType` → `llvm.Int`, etc. — these are attribute accesses, so the rename is straightforward.

**Step 2: Update import statements**

Each file's import lines need updating. For example:
- `from dgen.dialects.builtin import IndexType, Nil, String` → `from dgen.dialects.builtin import Index, Nil, String`
- `from toy.dialects.affine import MemRefType, ShapeType` → `from toy.dialects.affine import MemRef, Shape`
- `from toy.dialects.toy import TensorType` → `from toy.dialects.toy import Tensor`
- `from dgen.dialects.llvm import IntType` → `from dgen.dialects.llvm import Int as LlvmInt` (if collision) or just use `llvm.Int`

**Watch for `Int` collision in `dgen/codegen.py`:** This file imports `IntType` from llvm and uses it in an isinstance check. After rename, `from dgen.dialects.llvm import Int` shadows the builtin. Options:
- Use `llvm.Int` instead of importing directly (preferred)
- Use `from dgen.dialects.llvm import Int as LlvmInt`

**Step 3: Run tests to find remaining issues**

Run: `python -m pytest . -q`
Expected: Fix any remaining failures from missed renames.

**Step 4: Format and lint**

```bash
ruff format && ruff check --fix
```

**Step 5: Commit**

```
jj new -m "refactor: rename all type classes to drop Type suffix"
```

---

### Task 8: Update layout imports in non-generated files

Change remaining `from dgen.layout import X` to `from dgen import layout` and prefix usages.

**Files (same as collision-avoidance files from Task 7, plus any remaining):**
- `dgen/module.py`
- `toy/dialects/__init__.py`
- `toy/passes/toy_to_affine.py`
- `toy/passes/affine_to_llvm.py`
- `toy/test/test_layout.py`

**Note:** This may already be done as part of Task 7's collision avoidance. If so, skip this task.

**Step 1: Search for remaining old-style imports**

```bash
grep -rn "from dgen.layout import" --include="*.py"
```

Only generated files should have `from dgen import ... layout`. Non-generated files should use `from dgen import layout` or nothing.

Actually, generated files now emit `from dgen import ... layout` as part of the framework import line. Non-generated files that use layout types should use `from dgen import layout`.

**Step 2: Update each file**

For each file still importing from `dgen.layout`:
1. Change `from dgen.layout import X, Y` to `from dgen import layout` (if not already imported)
2. Replace `X(` with `layout.X(`, `Y(` with `layout.Y(`, etc.
3. Replace `isinstance(x, X)` with `isinstance(x, layout.X)`

**Step 3: Run tests**

Run: `python -m pytest . -q`

**Step 4: Format and lint**

```bash
ruff format && ruff check --fix
```

**Step 5: Commit**

```
jj new -m "refactor: use 'from dgen import layout' in all non-generated files"
```

---

### Task 9: Remove StringLayout alias and clean up

**Files:**
- Modify: `dgen/layout.py`

**Step 1: Remove the `StringLayout = String` alias**

Search for any remaining references to `StringLayout` across the codebase:
```bash
grep -rn "StringLayout" --include="*.py"
```

If none remain outside `layout.py`, remove the alias line.

**Step 2: Run full test suite**

Run: `python -m pytest . -q`
Expected: All tests pass

**Step 3: Format, lint, type-check**

```bash
ruff format && ruff check --fix && ty check
```

**Step 4: Commit**

```
jj new -m "cleanup: remove StringLayout backward-compat alias"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

```bash
python -m pytest . -q
```

Expected: All 187+ tests pass.

**Step 2: Verify line count of python.py**

```bash
wc -l dgen/gen/python.py
```

Expected: ~130-160 lines.

**Step 3: Verify no old names remain**

```bash
grep -rn "IndexType\|F64Type\|ShapeType\|MemRefType\|TensorType\|PtrType\|IntType\|FloatType\|VoidType\|StringLayout" --include="*.py" | grep -v "docs/"
```

Expected: No matches (except possibly in test expected-string contexts, which should also be updated).

**Step 4: Regenerate and diff to verify idempotency**

```bash
python -m dgen.gen dgen/dialects/builtin.dgen | diff - dgen/dialects/builtin.py
python -m dgen.gen dgen/dialects/llvm.dgen -I builtin=dgen.dialects.builtin | diff - dgen/dialects/llvm.py
python -m dgen.gen toy/dialects/affine.dgen -I builtin=dgen.dialects.builtin | diff - toy/dialects/affine.py
python -m dgen.gen toy/dialects/toy.dgen -I builtin=dgen.dialects.builtin -I affine=toy.dialects.affine | diff - toy/dialects/toy.py
```

Expected: No differences.

**Step 5: Format, lint, type-check one final time**

```bash
ruff format && ruff check --fix && ty check
```
