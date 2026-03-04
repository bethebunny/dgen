# Dialect File Codegen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a codegen tool that reads `.dgen` dialect specification files and generates Python dialect modules, then generate the 4 existing dialects from `.dgen` specs.

**Architecture:** A parser reads `.dgen` files into an AST; a Python code generator emits dialect modules from that AST. Dialects are always generated directly to a `dialect.py`, there is no such thing as hand-written or manual additions to dialects. Any language-specific helpers that are not generated must be written directly, and it's up to the individual compiler author how to package them.

**Tech Stack:** Python, pytest, ruff, dataclasses

---

## Gap Analysis

### What the spec envisions (docs/dialect-files.md)

- `.dgen` files as **language-independent** source of truth for dialect structure
- Generate definitions for **any compiler language** (Python, C++, Rust, etc.)
- Only specify types, ops, and traits — **no language-specific codegen passthroughs**
- Types have **known memory layouts**
- Methods use a **simple cross-language function language** (future)

### What exists today

- **0** `.dgen` files, **0** codegen tooling
- **4** hand-written Python dialect files: `builtin.py`, `llvm.py`, `affine.py`, `toy.py`
- Python-specific methods mixed with structural declarations
- "dgen dialect definition files + generation" listed as "not planned" in `docs/plan.md`

### What's in scope NOW

The structural subset covering the 4 existing dialects:
- Type declarations with parameters and layouts
- Op declarations with params, operands, return types, defaults
- Block declarations on ops
- Trait declarations
- Cross-dialect imports

### What's NOT in scope (future, per spec)

- `requires` constraints / type variables (`$X`)
- `has trait` / trait conformance
- `static` fields
- Cross-language method definitions
- Validation rules

---

## Design Decisions

### D1: No language-specific code in `.dgen` files

Per the spec: _"No language specific codegen passthroughs."_ Methods like `for_value`, `unpack_shape`, `resolve_constant`, `asm` are Python-specific and belong in companion Python files, not in `.dgen`.

### D2: Two-file pattern for dialects with Python-specific behavior

```
dialect.dgen           -> source of truth (language-independent structure)
_dialect_generated.py  -> fully generated from .dgen (never hand-edited)
dialect.py             -> public API (imports generated + adds Python behavior)
```

For simple dialects with no manual additions, generate directly to `dialect.py`.

### D3: `.dgen` name = ASM registration name

The name in the `.dgen` file is the name passed to `@dialect.type("name")` or `@dialect.op("name")`. This matches the current hand-written code exactly.

### D4: Python class name derivation (codegen concern)

The Python codegen derives class names from ASM names:
- Types: `CamelCase(asm_name) + "Type"` by default (e.g., `index` -> `IndexType`, `Shape` -> `ShapeType`)
- Ops: `CamelCase(asm_name) + "Op"` by default (e.g., `transpose` -> `TransposeOp`)
- Exceptions via override map for established names that don't follow the pattern: `Nil`, `String`, `List`, `InferredShapeTensor` keep their current names

### D5: Layout expressions are declarative

Layouts reference type parameters by name. The codegen resolves:
- Index param in layout -> `self.param.__constant__.unpack()[0]` (extract int)
- Type param in layout -> `self.param.__layout__` (extract layout)
- Static layouts -> class attribute
- Parametric layouts -> `@property`

### D6: LLVM dialect deferred

LLVM types have custom ASM formatting and are NOT dialect-registered (intentionally). This is fundamentally different from other dialects. LLVM dialect generation is deferred to a follow-up plan. The hand-written `llvm.py` remains as-is.

---

## `.dgen` Format Specification

### Grammar

```
file        = (import | trait | type | op)*
import      = "from" IDENT "import" name_list
trait       = "trait" NAME
type        = "type" NAME [ "<" param_list ">" ] ":" NEWLINE INDENT type_body DEDENT
            | "type" NAME [ "<" param_list ">" ]
op          = "op" IDENT [ "<" param_list ">" ] "(" operand_list ")" "->" return_type [":" NEWLINE INDENT op_body DEDENT]
param_list  = param ("," param)*
param       = IDENT ":" type_ref ["=" default]
operand_list = operand ("," operand)* | empty
operand     = IDENT ":" type_ref
type_body   = layout_decl
layout_decl = "layout" layout_expr
op_body     = block_decl*
block_decl  = "block" IDENT
return_type = type_ref ["=" default]
type_ref    = NAME ["<" type_args ">"] | "list" "<" type_ref ">" | "Type"
layout_expr = LAYOUT_NAME | LAYOUT_NAME "<" layout_args ">"
```

### Conventions (from spec)
- ops are `lower_snake_case`
- types and traits are `UpperCamelCase` (or lowercase for builtins like `index`, `f64`)
- builtin types referenced without prefix
- other types imported by namespace

### Example: affine dialect

```dgen
from builtin import Index, Nil, F64

trait HasSingleBlock

type Shape<rank: Index>:
    layout Array<INT, rank>

type MemRef<shape: Shape, dtype: Type = F64>:
    layout Pointer<VOID>

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

### Example: toy dialect

```dgen
from builtin import Index, Nil, F64, String
from affine import Shape

type Tensor<shape: Shape, dtype: Type = F64>:
    # Layout is complex (needs prod(shape)), provided by companion file

type InferredShapeTensor<dtype: Type = F64>:
    layout VOID

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

### Example: builtin dialect

```dgen
trait HasSingleBlock

type index:
    layout INT

type f64:
    layout FLOAT64

type Nil:
    layout VOID

type String:
    layout FatPointer<BYTE>

type List<element_type: Type>:
    layout FatPointer<element_type>

op pack(values: list<Type>) -> List
op list_get<index: Index>(list: List) -> Type
op add_index(lhs: Index, rhs: Index) -> Index
op return(value: Type = Nil) -> Nil
op function() -> Function:
    block body
```

---

## Implementation

### Phase 1: Codegen Tool — Parser

#### Task 1: Define AST types

**Files:**
- Create: `dgen/gen/__init__.py`
- Create: `dgen/gen/ast.py`
- Test: `test/test_gen_ast.py`

**Step 1: Write the AST dataclasses**

```python
# dgen/gen/ast.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ImportDecl:
    module: str
    names: list[str]

@dataclass
class TypeRef:
    """A reference to a type: Name, Name<args>, list<T>, or Type."""
    name: str
    args: list[TypeRef] = field(default_factory=list)

@dataclass
class ParamDecl:
    name: str
    type: TypeRef
    default: str | None = None

@dataclass
class OperandDecl:
    name: str
    type: TypeRef

@dataclass
class LayoutExpr:
    """A layout expression: INT, Array<INT, rank>, FatPointer<element_type>."""
    name: str
    args: list[str] = field(default_factory=list)  # primitive names or param refs

@dataclass
class TraitDecl:
    name: str

@dataclass
class TypeDecl:
    name: str
    params: list[ParamDecl] = field(default_factory=list)
    layout: LayoutExpr | None = None

@dataclass
class OpDecl:
    name: str
    params: list[ParamDecl] = field(default_factory=list)
    operands: list[OperandDecl] = field(default_factory=list)
    return_type: TypeRef = field(default_factory=lambda: TypeRef("Type"))
    return_default: str | None = None
    blocks: list[str] = field(default_factory=list)

@dataclass
class DgenFile:
    imports: list[ImportDecl] = field(default_factory=list)
    traits: list[TraitDecl] = field(default_factory=list)
    types: list[TypeDecl] = field(default_factory=list)
    ops: list[OpDecl] = field(default_factory=list)
```

**Step 2: Write a smoke test**

```python
# test/test_gen_ast.py
from dgen.gen.ast import DgenFile, TypeDecl, OpDecl, LayoutExpr, TypeRef

def test_ast_construction():
    f = DgenFile(
        types=[TypeDecl(name="index", layout=LayoutExpr("INT"))],
        ops=[OpDecl(name="return", return_type=TypeRef("Nil"))],
    )
    assert len(f.types) == 1
    assert f.types[0].name == "index"
```

**Step 3: Run test**

Run: `pytest test/test_gen_ast.py -v`
Expected: PASS

**Step 4: Commit**

```
jj new -m "gen: add AST types for .dgen file format"
```

---

#### Task 2: Build the `.dgen` parser

**Files:**
- Create: `dgen/gen/parser.py`
- Test: `test/test_gen_parser.py`

**Step 1: Write parser tests**

```python
# test/test_gen_parser.py
from dgen.gen.parser import parse

def test_parse_import():
    result = parse("from builtin import Index, Nil\n")
    assert len(result.imports) == 1
    assert result.imports[0].module == "builtin"
    assert result.imports[0].names == ["Index", "Nil"]

def test_parse_trait():
    result = parse("trait HasSingleBlock\n")
    assert len(result.traits) == 1
    assert result.traits[0].name == "HasSingleBlock"

def test_parse_simple_type():
    result = parse("type index:\n    layout INT\n")
    assert len(result.types) == 1
    t = result.types[0]
    assert t.name == "index"
    assert t.layout is not None
    assert t.layout.name == "INT"

def test_parse_parameterized_type():
    result = parse("type Shape<rank: Index>:\n    layout Array<INT, rank>\n")
    t = result.types[0]
    assert t.name == "Shape"
    assert len(t.params) == 1
    assert t.params[0].name == "rank"
    assert t.params[0].type.name == "Index"
    assert t.layout is not None
    assert t.layout.name == "Array"
    assert t.layout.args == ["INT", "rank"]

def test_parse_type_with_default_param():
    result = parse("type Tensor<shape: Shape, dtype: Type = F64>:\n    layout VOID\n")
    t = result.types[0]
    assert len(t.params) == 2
    assert t.params[1].name == "dtype"
    assert t.params[1].default == "F64"

def test_parse_simple_op():
    result = parse("op transpose(input: Type) -> Type\n")
    op = result.ops[0]
    assert op.name == "transpose"
    assert len(op.operands) == 1
    assert op.operands[0].name == "input"
    assert op.return_type.name == "Type"

def test_parse_op_with_params():
    result = parse("op concat<axis: Index>(lhs: Type, rhs: Type) -> Type\n")
    op = result.ops[0]
    assert len(op.params) == 1
    assert op.params[0].name == "axis"
    assert len(op.operands) == 2

def test_parse_op_with_block():
    src = "op for<lo: Index, hi: Index>() -> Nil:\n    block body\n"
    op = parse(src).ops[0]
    assert op.blocks == ["body"]
    assert op.return_type.name == "Nil"

def test_parse_op_with_default_operand():
    result = parse("op return(value: Type = Nil) -> Nil\n")
    op = result.ops[0]
    assert op.operands[0].name == "value"

def test_parse_list_operand():
    result = parse("op pack(values: list<Type>) -> List\n")
    op = result.ops[0]
    assert op.operands[0].type.name == "list"
    assert op.operands[0].type.args[0].name == "Type"

def test_parse_op_with_list_param():
    result = parse("op phi<labels: list<String>>(values: list<Type>) -> Type\n")
    op = result.ops[0]
    assert op.params[0].type.name == "list"
    assert op.params[0].type.args[0].name == "String"

def test_parse_no_operands():
    result = parse("op function() -> Function:\n    block body\n")
    op = result.ops[0]
    assert op.operands == []
    assert op.blocks == ["body"]

def test_parse_bodyless_type():
    result = parse("trait HasSingleBlock\ntype Nil:\n    layout VOID\n")
    assert len(result.traits) == 1
    assert len(result.types) == 1

def test_parse_full_file():
    src = """\
from builtin import Index, Nil

trait HasSingleBlock

type Shape<rank: Index>:
    layout Array<INT, rank>

op alloc(shape: Shape) -> Type
op for<lo: Index, hi: Index>() -> Nil:
    block body
"""
    result = parse(src)
    assert len(result.imports) == 1
    assert len(result.traits) == 1
    assert len(result.types) == 1
    assert len(result.ops) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest test/test_gen_parser.py -v`
Expected: FAIL (parser not implemented)

**Step 3: Implement the parser**

The parser is line-oriented with indentation tracking. It processes:
1. `from ... import ...` lines
2. `trait Name` lines
3. `type Name...` lines (with optional indented body)
4. `op name...` lines (with optional indented body)

```python
# dgen/gen/parser.py
"""Parser for .dgen dialect specification files."""

from __future__ import annotations

from dgen.gen.ast import (
    DgenFile,
    ImportDecl,
    LayoutExpr,
    OpDecl,
    OperandDecl,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)


def parse(source: str) -> DgenFile:
    """Parse a .dgen source string into a DgenFile AST."""
    return _Parser(source).parse()


class _Parser:
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.pos = 0

    def parse(self) -> DgenFile:
        result = DgenFile()
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            if not line or line.startswith("#"):
                self.pos += 1
                continue
            if line.startswith("from "):
                result.imports.append(self._parse_import(line))
            elif line.startswith("trait "):
                result.traits.append(self._parse_trait(line))
            elif line.startswith("type "):
                result.types.append(self._parse_type(line))
            elif line.startswith("op "):
                result.ops.append(self._parse_op(line))
            else:
                raise SyntaxError(f"unexpected line: {line!r}")
            self.pos += 1
        return result

    def _parse_import(self, line: str) -> ImportDecl:
        # from module import Name1, Name2
        parts = line.split()
        assert parts[0] == "from" and parts[2] == "import"
        module = parts[1]
        names = [n.strip().rstrip(",") for n in parts[3:]]
        return ImportDecl(module=module, names=names)

    def _parse_trait(self, line: str) -> TraitDecl:
        name = line.split()[1]
        return TraitDecl(name=name)

    def _parse_type(self, line: str) -> TypeDecl:
        # Remove "type " prefix
        rest = line[5:]
        params: list[ParamDecl] = []
        layout: LayoutExpr | None = None

        # Parse name and optional params
        if "<" in rest.split(":")[0]:
            name, param_str = rest.split("<", 1)
            name = name.strip()
            param_str = param_str.split(">")[0]
            params = _parse_params(param_str)
            rest = rest[rest.index(">") + 1 :]
        else:
            name = rest.split(":")[0].strip()

        # Check for body (ends with ':')
        if line.rstrip().endswith(":"):
            layout = self._parse_type_body()

        return TypeDecl(name=name, params=params, layout=layout)

    def _parse_type_body(self) -> LayoutExpr | None:
        """Parse indented type body lines."""
        layout = None
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("layout "):
                layout = _parse_layout(stripped[7:])
        return layout

    def _parse_op(self, line: str) -> OpDecl:
        # Remove "op " prefix
        rest = line[3:]
        params: list[ParamDecl] = []
        blocks: list[str] = []

        # Parse name
        name_end = min(
            (rest.index(c) for c in "<(:" if c in rest),
            default=len(rest),
        )
        name = rest[:name_end].strip()
        rest = rest[name_end:]

        # Parse optional params <...>
        if rest.startswith("<"):
            close = _find_matching(rest, "<", ">")
            param_str = rest[1:close]
            params = _parse_params(param_str)
            rest = rest[close + 1 :]

        # Parse operands (...)
        operands: list[OperandDecl] = []
        if rest.startswith("("):
            close = rest.index(")")
            operand_str = rest[1:close].strip()
            if operand_str:
                operands = _parse_operands(operand_str)
            rest = rest[close + 1 :]

        # Parse return type -> Type
        return_type = TypeRef("Type")
        return_default: str | None = None
        if "->" in rest:
            ret_str = rest[rest.index("->") + 2 :].strip().rstrip(":")
            return_type = _parse_type_ref(ret_str.strip())

        # Parse optional body (blocks)
        if line.rstrip().endswith(":"):
            blocks = self._parse_op_body()

        return OpDecl(
            name=name,
            params=params,
            operands=operands,
            return_type=return_type,
            return_default=return_default,
            blocks=blocks,
        )

    def _parse_op_body(self) -> list[str]:
        """Parse indented op body lines."""
        blocks: list[str] = []
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("block "):
                blocks.append(stripped.split()[1])
        return blocks


def _parse_params(s: str) -> list[ParamDecl]:
    """Parse a comma-separated parameter list."""
    params: list[ParamDecl] = []
    for part in _split_commas(s):
        part = part.strip()
        if "=" in part:
            decl, default = part.rsplit("=", 1)
            name, type_str = decl.split(":", 1)
            params.append(
                ParamDecl(
                    name=name.strip(),
                    type=_parse_type_ref(type_str.strip()),
                    default=default.strip(),
                )
            )
        else:
            name, type_str = part.split(":", 1)
            params.append(
                ParamDecl(name=name.strip(), type=_parse_type_ref(type_str.strip()))
            )
    return params


def _parse_operands(s: str) -> list[OperandDecl]:
    """Parse a comma-separated operand list."""
    operands: list[OperandDecl] = []
    for part in _split_commas(s):
        part = part.strip()
        # Handle default: "value: Type = Nil"
        if "=" in part:
            part = part[: part.index("=")].strip()
        name, type_str = part.split(":", 1)
        operands.append(
            OperandDecl(name=name.strip(), type=_parse_type_ref(type_str.strip()))
        )
    return operands


def _parse_type_ref(s: str) -> TypeRef:
    """Parse a type reference: Name, Name<args>, list<T>, or Type."""
    s = s.strip()
    if "<" in s:
        name = s[: s.index("<")]
        args_str = s[s.index("<") + 1 : s.rindex(">")]
        args = [_parse_type_ref(a.strip()) for a in _split_commas(args_str)]
        return TypeRef(name=name, args=args)
    return TypeRef(name=s)


def _parse_layout(s: str) -> LayoutExpr:
    """Parse a layout expression: INT, Array<INT, rank>, FatPointer<BYTE>."""
    s = s.strip()
    if "<" in s:
        name = s[: s.index("<")]
        args_str = s[s.index("<") + 1 : s.rindex(">")]
        args = [a.strip() for a in args_str.split(",")]
        return LayoutExpr(name=name, args=args)
    return LayoutExpr(name=s)


def _split_commas(s: str) -> list[str]:
    """Split on commas respecting <> nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _find_matching(s: str, open_ch: str, close_ch: str) -> int:
    """Find the matching close bracket."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise SyntaxError(f"unmatched {open_ch}")
```

**Step 4: Run tests**

Run: `pytest test/test_gen_parser.py -v`
Expected: PASS

**Step 5: Commit**

```
jj new -m "gen: add .dgen file parser"
```

---

### Phase 2: Codegen Tool — Python Generator

#### Task 3: Build the Python code generator

**Files:**
- Create: `dgen/gen/python.py`
- Test: `test/test_gen_python.py`

**Step 1: Write generator tests**

Test that the generator produces correct Python source from AST. Tests compare against the existing hand-written files line-by-line where possible.

```python
# test/test_gen_python.py
from dgen.gen.ast import (
    DgenFile, ImportDecl, TypeDecl, OpDecl, OperandDecl,
    ParamDecl, LayoutExpr, TraitDecl, TypeRef,
)
from dgen.gen.python import generate


def test_generate_header():
    f = DgenFile()
    code = generate(f, dialect_name="test")
    assert "# GENERATED" in code
    assert "from dgen import" in code
    assert 'Dialect("test")' in code


def test_generate_simple_type():
    f = DgenFile(types=[TypeDecl(name="index", layout=LayoutExpr("INT"))])
    code = generate(f, dialect_name="test")
    assert '@test.type("index")' in code
    assert "class IndexType(Type):" in code
    assert "__layout__ = INT" in code
    assert "@dataclass(frozen=True)" in code


def test_generate_parameterized_type():
    f = DgenFile(types=[TypeDecl(
        name="Shape",
        params=[ParamDecl(name="rank", type=TypeRef("Index"))],
        layout=LayoutExpr("Array", ["INT", "rank"]),
    )])
    code = generate(f, dialect_name="test")
    assert "class ShapeType(Type):" in code
    assert "rank: Value[IndexType]" in code
    assert '__params__ = (("rank", IndexType),)' in code
    assert "def __layout__(self)" in code
    assert "Array(INT," in code


def test_generate_type_default_param():
    f = DgenFile(types=[TypeDecl(
        name="Tensor",
        params=[
            ParamDecl(name="shape", type=TypeRef("Shape")),
            ParamDecl(name="dtype", type=TypeRef("Type"), default="F64"),
        ],
    )])
    code = generate(f, dialect_name="test")
    assert "dtype: Type = F64Type()" in code or "dtype: Type = builtin.F64Type()" in code


def test_generate_simple_op():
    f = DgenFile(ops=[OpDecl(
        name="transpose",
        operands=[OperandDecl(name="input", type=TypeRef("Type"))],
        return_type=TypeRef("Type"),
    )])
    code = generate(f, dialect_name="test")
    assert '@test.op("transpose")' in code
    assert "class TransposeOp(Op):" in code
    assert "input: Value" in code
    assert "type: Type" in code
    assert '__operands__ = (("input", Type),)' in code


def test_generate_op_with_params():
    f = DgenFile(ops=[OpDecl(
        name="concat",
        params=[ParamDecl(name="axis", type=TypeRef("Index"))],
        operands=[
            OperandDecl(name="lhs", type=TypeRef("Type")),
            OperandDecl(name="rhs", type=TypeRef("Type")),
        ],
        return_type=TypeRef("Type"),
    )])
    code = generate(f, dialect_name="test")
    assert "axis: Value[IndexType]" in code
    assert '__params__ = (("axis", IndexType),)' in code
    assert '__operands__ = (("lhs", Type), ("rhs", Type))' in code


def test_generate_op_with_block():
    f = DgenFile(ops=[OpDecl(
        name="for",
        params=[
            ParamDecl(name="lo", type=TypeRef("Index")),
            ParamDecl(name="hi", type=TypeRef("Index")),
        ],
        return_type=TypeRef("Nil"),
        blocks=["body"],
    )])
    code = generate(f, dialect_name="test")
    assert "body: Block" in code
    assert '__blocks__ = ("body",)' in code


def test_generate_op_return_default():
    f = DgenFile(ops=[OpDecl(
        name="store",
        operands=[
            OperandDecl(name="value", type=TypeRef("Type")),
            OperandDecl(name="ptr", type=TypeRef("Type")),
        ],
        return_type=TypeRef("Nil"),
    )])
    code = generate(f, dialect_name="test")
    assert "type: Type = Nil()" in code


def test_generate_trait():
    f = DgenFile(traits=[TraitDecl(name="HasSingleBlock")])
    code = generate(f, dialect_name="test")
    assert "class HasSingleBlock:" in code


def test_generate_op_with_trait_and_block():
    f = DgenFile(
        traits=[TraitDecl(name="HasSingleBlock")],
        ops=[OpDecl(name="function", return_type=TypeRef("Function"), blocks=["body"])],
    )
    code = generate(f, dialect_name="test")
    assert "class FunctionOp(HasSingleBlock, Op):" in code


def test_generate_list_operand():
    f = DgenFile(ops=[OpDecl(
        name="pack",
        operands=[OperandDecl(name="values", type=TypeRef("list", [TypeRef("Type")]))],
        return_type=TypeRef("List"),
    )])
    code = generate(f, dialect_name="test")
    assert "values: list[Value" in code


def test_generate_imports():
    f = DgenFile(imports=[ImportDecl(module="builtin", names=["Index", "Nil"])])
    code = generate(f, dialect_name="test")
    # Should produce appropriate Python imports
    assert "IndexType" in code or "Index" in code
```

**Step 2: Run tests to verify they fail**

Run: `pytest test/test_gen_python.py -v`
Expected: FAIL

**Step 3: Implement the generator**

The generator takes a `DgenFile` AST and produces Python source code. Key responsibilities:
- Emit imports (dgen framework + cross-dialect)
- Emit trait classes
- Emit type dataclasses with registration decorators, params, layouts
- Emit op dataclasses with registration decorators, params, operands, blocks
- Handle name mapping (ASM name -> Python class name)
- Handle layout code generation (static vs property)

Key implementation details for the Python generator:

```python
# dgen/gen/python.py  (outline — full implementation in task)

# Name mapping: ASM name -> Python class name
_TYPE_NAME_OVERRIDES: dict[str, str] = {
    "Nil": "Nil",
    "String": "String",
    "List": "List",
    "InferredShapeTensor": "InferredShapeTensor",
}

def _type_class_name(asm_name: str) -> str:
    """Derive Python class name from ASM registration name."""
    if asm_name in _TYPE_NAME_OVERRIDES:
        return _TYPE_NAME_OVERRIDES[asm_name]
    # CamelCase + Type suffix
    camel = "".join(word.capitalize() for word in asm_name.replace("_", " ").split())
    if not camel[-1].isalpha():
        # Handle names like "f64" -> "F64"
        camel = asm_name.upper() if len(asm_name) <= 4 else camel
    return camel + "Type" if not camel.endswith("Type") else camel

def _op_class_name(asm_name: str) -> str:
    camel = "".join(word.capitalize() for word in asm_name.split("_"))
    return camel + "Op"

# Layout generation:
# - Static (no param refs): emit __layout__ = LAYOUT(args)
# - Parametric (has param refs): emit @property def __layout__(self)
#   - Index params: self.param.__constant__.unpack()[0]
#   - Type params: self.param.__layout__

# Return type defaults:
# - "Type" -> no default (generic)
# - Concrete type with no required params -> default instance
# - Concrete parameterized type -> no default (caller provides)
```

**Step 4: Run tests**

Run: `pytest test/test_gen_python.py -v`
Expected: PASS

**Step 5: Commit**

```
jj new -m "gen: add Python code generator for .dgen files"
```

---

#### Task 4: Add CLI entry point for codegen

**Files:**
- Create: `dgen/gen/__main__.py`

**Step 1: Implement CLI**

```python
# dgen/gen/__main__.py
"""CLI: python -m dgen.gen path/to/dialect.dgen"""
import sys
from pathlib import Path

from dgen.gen.parser import parse
from dgen.gen.python import generate


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m dgen.gen <file.dgen>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    source = path.read_text()
    ast = parse(source)
    dialect_name = path.stem  # e.g., "affine" from "affine.dgen"
    code = generate(ast, dialect_name=dialect_name)
    print(code)


if __name__ == "__main__":
    main()
```

**Step 2: Test manually**

Run: `echo "type index:\n    layout INT" | python -m dgen.gen /dev/stdin`
Expected: prints generated Python

**Step 3: Commit**

```
jj new -m "gen: add CLI entry point (python -m dgen.gen)"
```

---

### Phase 3: Generate Affine Dialect (proof of concept)

The affine dialect is the best proof of concept: it has parameterized types, ops with params/operands/blocks, default return types, and trait usage — but no custom methods that need companion files (except `shape_constant` helper and `ShapeType.for_value`/`ShapeType.__layout__`).

#### Task 5: Write affine.dgen

**Files:**
- Create: `toy/dialects/affine.dgen`

```dgen
from builtin import Index, Nil, F64

trait HasSingleBlock

type Shape<rank: Index>:
    layout Array<INT, rank>

type MemRef<shape: Shape, dtype: Type = F64>:
    layout Pointer<VOID>

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

**Step 1: Commit the .dgen file**

```
jj new -m "gen: add affine.dgen dialect spec"
```

---

#### Task 6: Generate affine and validate

**Step 1: Generate to _affine_generated.py**

Run: `python -m dgen.gen toy/dialects/affine.dgen > toy/dialects/_affine_generated.py`

**Step 2: Write the companion affine.py**

The companion `affine.py` imports everything from the generated file and adds Python-specific behavior:

```python
# toy/dialects/affine.py
"""Affine dialect types and operations."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dgen import Constant
from dgen.dialects.builtin import IndexType
from dgen.layout import INT, Array, Layout

# Import all generated types and ops
from toy.dialects._affine_generated import *  # noqa: F403
from toy.dialects._affine_generated import ShapeType, affine  # noqa: F401


# --- Python-specific methods on generated types ---

@classmethod  # type: ignore[misc]
def _shape_for_value(cls: type[ShapeType], value: object) -> ShapeType:
    if isinstance(value, Constant):
        assert isinstance(value.type, IndexType)
        return cls(rank=value.type.constant(value.__constant__.unpack()[0]))
    assert isinstance(value, list)
    return cls(rank=IndexType().constant(len(value)))

ShapeType.for_value = _shape_for_value  # type: ignore[assignment]


@property  # type: ignore[misc]
def _shape_layout(self: ShapeType) -> Layout:
    assert self.rank.ready
    return Array(INT, self.rank.__constant__.unpack()[0])

ShapeType.__layout__ = _shape_layout  # type: ignore[assignment, misc]


# --- Helper functions ---

def shape_constant(dims: Sequence[int]) -> Constant:
    """Create a Constant[ShapeType] from a list of dims."""
    rank = IndexType().constant(len(dims))
    return ShapeType(rank=rank).constant(dims)
```

**Step 3: Run all tests**

Run: `pytest . -q`
Expected: 217 passed

**Step 4: Format and lint**

Run: `ruff format && ruff check --fix`
Expected: clean

**Step 5: Commit**

```
jj new -m "gen: generate affine dialect from .dgen spec"
```

---

### Phase 4: Generate Toy Dialect

#### Task 7: Write toy.dgen and generate

**Files:**
- Create: `toy/dialects/toy.dgen`
- Create: `toy/dialects/_toy_generated.py` (generated)
- Modify: `toy/dialects/toy.py` -> companion file

**Step 1: Write toy.dgen**

```dgen
from builtin import Index, Nil, F64, String
from affine import Shape

type Tensor<shape: Shape, dtype: Type = F64>:
    # Layout computed in companion (needs math.prod)

type InferredShapeTensor<dtype: Type = F64>:
    layout VOID

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

**Step 2: Generate**

Run: `python -m dgen.gen toy/dialects/toy.dgen > toy/dialects/_toy_generated.py`

**Step 3: Write companion toy.py**

```python
# toy/dialects/toy.py
"""Toy dialect IR types and operations."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from dgen import Type
from dgen.dialects import builtin
from dgen.layout import FLOAT64, Array, Layout
from dgen.type import Memory

# Import all generated types and ops
from toy.dialects._toy_generated import *  # noqa: F403
from toy.dialects._toy_generated import (  # noqa: F401
    DimSizeOp,
    TensorType,
    toy,
)
from toy.dialects.affine import ShapeType


# --- TensorType methods ---

def _tensor_unpack_shape(self: TensorType) -> list[int]:
    return list(self.shape.__constant__.unpack())

TensorType.unpack_shape = _tensor_unpack_shape  # type: ignore[assignment]


@property  # type: ignore[misc]
def _tensor_layout(self: TensorType) -> Layout:
    assert self.shape.ready
    shape: Memory[ShapeType] = self.shape.__constant__
    return Array(FLOAT64, prod(shape.unpack()))

TensorType.__layout__ = _tensor_layout  # type: ignore[assignment, misc]


# --- DimSizeOp methods ---

def _dim_size_resolve_constant(self: DimSizeOp) -> int | None:
    shape = getattr(self.input.type, "shape", None)
    if shape is not None and getattr(shape, "ready", False):
        return shape.__constant__.unpack()[self.axis.__constant__.unpack()[0]]
    return None

DimSizeOp.resolve_constant = _dim_size_resolve_constant  # type: ignore[assignment]


# --- FunctionType (not dialect-registered) ---

@dataclass
class FunctionType(builtin.Function):
    """Toy function signature with explicit input types."""
    inputs: list[Type]
```

**Step 4: Run tests**

Run: `pytest . -q`
Expected: 217 passed

**Step 5: Format, lint, commit**

```
ruff format && ruff check --fix
jj new -m "gen: generate toy dialect from .dgen spec"
```

---

### Phase 5: Generate Builtin Dialect

The builtin dialect is the most complex: `ConstantOp`, `Function`, `Module`, and helper functions can't be generated. Uses two-file pattern.

#### Task 8: Write builtin.dgen and generate

**Files:**
- Create: `dgen/dialects/builtin.dgen`
- Create: `dgen/dialects/_builtin_generated.py` (generated)
- Modify: `dgen/dialects/builtin.py` -> companion file

**Step 1: Write builtin.dgen**

```dgen
trait HasSingleBlock

type index:
    layout INT

type f64:
    layout FLOAT64

type Nil:
    layout VOID

type String:
    layout FatPointer<BYTE>

type List<element_type: Type>:
    layout FatPointer<element_type>

op pack(values: list<Type>) -> List
op list_get<index: Index>(list: List) -> Type
op add_index(lhs: Index, rhs: Index) -> Index
op return(value: Type = Nil) -> Nil
op function() -> Function:
    block body
```

**Step 2: Handle the codegen challenges**

Key issues for the builtin dialect:
- **Self-referential imports:** The generated file can't `from dgen.dialects.builtin import IndexType` because IT IS builtin. The codegen needs a `self_dialect=True` flag to skip cross-dialect imports.
- **`Function` type:** Not dialect-registered, defined in companion. Referenced as return type of `function` op. The codegen must accept it as an import or a known name.
- **`HasSingleBlock` trait:** Used by `function` op. Defined in this file AND potentially imported by other dialects. The codegen emits the trait class.
- **Op return defaults with no-arg types:** `Nil()`, `IndexType()` can be constructed without args. `List` cannot (needs `element_type`). The codegen must know which types are constructable.

**Step 3: Generate**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/_builtin_generated.py`

**Step 4: Write companion builtin.py**

```python
# dgen/dialects/builtin.py
"""Builtin structure types shared across all dialects."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dgen import Block, Constant, Dialect, Op, Type, Value, asm
from dgen.asm.formatting import SlotTracker, _is_sugar_op, format_expr, op_asm
from dgen.type import Memory

# Import all generated types and ops
from dgen.dialects._builtin_generated import *  # noqa: F403
from dgen.dialects._builtin_generated import (  # noqa: F401
    FunctionOp,
    HasSingleBlock,
    IndexType,
    List,
    Nil,
    String,
    builtin,
)


# --- Function type (not dialect-registered) ---

@dataclass
class Function(Type):
    """A function signature."""
    from dgen.layout import VOID
    __layout__ = VOID
    result: Type


# --- ConstantOp (custom __init__, multiple inheritance) ---

@builtin.op("constant")
@dataclass(eq=False, kw_only=True, init=False)
class ConstantOp(Op, Constant):
    value: Memory
    type: Type
    __operands__ = (("value", Type),)

    def __init__(self, *, value: object, type: Type, name: str | None = None) -> None:
        self.name = name
        self.type = type
        self.value = (
            value if isinstance(value, Memory) else Memory.from_value(type, value)
        )

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# --- String/List methods ---

@classmethod  # type: ignore[misc]
def _string_for_value(cls: type[String], value: object) -> String:
    assert isinstance(value, str)
    return cls()

String.for_value = _string_for_value  # type: ignore[assignment]


@classmethod  # type: ignore[misc]
def _list_for_value(cls: type[List], value: object) -> List:
    assert isinstance(value, list)
    return cls(element_type=IndexType())

List.for_value = _list_for_value  # type: ignore[assignment]


# --- FunctionOp ASM formatting ---

@property  # type: ignore[misc]
def _func_asm(self: FunctionOp) -> Iterable[str]:
    tracker = SlotTracker()
    for arg in self.body.args:
        tracker.track_name(arg)
    tracker.register(self.body.ops)
    name = tracker.track_name(self)
    arg_parts = []
    for a in self.body.args:
        n = tracker.track_name(a)
        if a.type is not None:
            arg_parts.append(f"%{n}: {format_expr(a.type, tracker)}")
        else:
            arg_parts.append(f"%{n}")
    args = ", ".join(arg_parts)
    yield f"%{name} = function ({args}) -> {format_expr(self.type.result, tracker)}:"
    for op in self.body.ops:
        if _is_sugar_op(op):
            continue
        yield from asm.indent(op_asm(op, tracker))

FunctionOp.asm = _func_asm  # type: ignore[assignment]


# --- Helper functions ---

def string_constant(s: str) -> Constant[String]:
    return String.for_value(s).constant(s)

def string_value(v: Value[String]) -> str:
    result = v.__constant__.to_python()
    assert isinstance(result, str)
    return result


# --- Module ---

def _walk_all_ops(op: Op) -> Iterable[Op]:
    yield op
    for _, block in op.blocks:
        for child in block.ops:
            yield from _walk_all_ops(child)

def _collect_dialects(func: FunctionOp, dialects: set[Dialect]) -> None:
    def _check_type(t: object) -> None:
        if t is None:
            return
        d = getattr(t, "dialect", None)
        if d is not None and d.name != "builtin":
            dialects.add(d)

    for op in _walk_all_ops(func):
        if op.dialect.name != "builtin":
            dialects.add(op.dialect)
        _check_type(getattr(op, "type", None))
    for arg in func.body.args:
        _check_type(arg.type)
    _check_type(func.type.result)


@dataclass
class Module:
    functions: list[FunctionOp]

    @property
    def asm(self) -> Iterable[str]:
        dialects: set[Dialect] = set()
        for func in self.functions:
            _collect_dialects(func, dialects)
        for d in sorted(dialects, key=lambda d: d.name):
            yield f"import {d.name}"
        if dialects:
            yield ""
        for function in self.functions:
            yield from function.asm
            yield ""
```

**Step 5: Rename FuncOp -> FunctionOp across codebase**

The generated code will name the op class `FunctionOp` (from `CamelCase("function") + "Op"`). The existing code uses `FuncOp`. Rename in ~15 sites:

Files to update:
- `dgen/codegen.py` (~2 references)
- `dgen/asm/parser.py` (~1 reference)
- `dgen/staging.py` (~1 reference)
- `toy/parser/lowering.py` (~2 references)
- `toy/passes/shape_inference.py` (~2 references)
- `toy/passes/optimize.py` (~1 reference)
- `toy/passes/toy_to_affine.py` (~2 references)
- `toy/passes/affine_to_llvm.py` (~1 reference)
- `toy/test/test_toy_printer.py` (~3 references)
- `toy/test/test_type_roundtrip.py` (~1 reference)

**Step 6: Run all tests**

Run: `pytest . -q`
Expected: 217 passed

**Step 7: Format, lint, commit**

```
ruff format && ruff check --fix
jj new -m "gen: generate builtin dialect from .dgen spec"
```

---

### Phase 6: Cross-Dialect Import Resolution

The Python codegen needs to resolve imports like `from builtin import Index` to the correct Python import path.

#### Task 9: Implement cross-dialect import resolution

When the codegen sees `from builtin import Index`, it needs to:
1. Know that `builtin` is a dialect
2. Know the Python module path for `builtin` (e.g., `dgen.dialects.builtin`)
3. Know the Python class name for `Index` (e.g., `IndexType`)

**Approach:** The `generate()` function accepts an `import_resolver` callback or a simple mapping:

```python
def generate(
    ast: DgenFile,
    dialect_name: str,
    import_map: dict[str, str] | None = None,
) -> str:
    # import_map: {"builtin": "dgen.dialects.builtin", "affine": "toy.dialects.affine"}
```

The import resolution emits:
```python
from dgen.dialects.builtin import IndexType, Nil
from toy.dialects.affine import ShapeType
```

For non-dialect imports (like `from math import prod`), emit verbatim.

**Step 1: Add import resolution tests to test_gen_python.py**

**Step 2: Implement in python.py**

**Step 3: Run tests, commit**

```
jj new -m "gen: add cross-dialect import resolution"
```

---

### Phase 7: LLVM Dialect (future)

The LLVM dialect has unregistered types with custom ASM formatting. This doesn't fit the standard codegen pattern. Options to explore:

1. **Generate ops only**, keep types hand-written
2. **Add custom ASM override** to `.dgen` format (small spec extension)
3. **Keep `llvm.py` fully hand-written** until the format evolves

This is deferred. The LLVM dialect is small (22 ops, 4 types) and stable. Generating it adds complexity without proportional benefit.

---

## Verification

After each task:
1. `pytest . -q` — all 217 tests pass
2. `ruff format && ruff check --fix`
3. `python -m toy.cli toy/test/testdata/constant.toy` — end-to-end pipeline works

## File Summary

| File | Change |
|------|--------|
| `dgen/gen/__init__.py` | NEW (empty) |
| `dgen/gen/ast.py` | NEW — AST types |
| `dgen/gen/parser.py` | NEW — .dgen parser |
| `dgen/gen/python.py` | NEW — Python code generator |
| `dgen/gen/__main__.py` | NEW — CLI entry point |
| `toy/dialects/affine.dgen` | NEW — affine dialect spec |
| `toy/dialects/_affine_generated.py` | NEW — generated |
| `toy/dialects/affine.py` | REWRITTEN — companion (imports generated + methods) |
| `toy/dialects/toy.dgen` | NEW — toy dialect spec |
| `toy/dialects/_toy_generated.py` | NEW — generated |
| `toy/dialects/toy.py` | REWRITTEN — companion |
| `dgen/dialects/builtin.dgen` | NEW — builtin dialect spec |
| `dgen/dialects/_builtin_generated.py` | NEW — generated |
| `dgen/dialects/builtin.py` | REWRITTEN — companion |
| `dgen/dialects/llvm.py` | UNCHANGED (deferred) |
| ~10 files | `FuncOp` -> `FunctionOp` rename |
| `test/test_gen_ast.py` | NEW — AST tests |
| `test/test_gen_parser.py` | NEW — parser tests |
| `test/test_gen_python.py` | NEW — codegen tests |
