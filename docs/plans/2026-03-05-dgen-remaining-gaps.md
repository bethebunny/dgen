# Closing All .dgen Gaps — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the three remaining features from `docs/dialect-files.md`: multi-field struct layout (`layout.Record`), `$X` metavariable constraints with `requires` clauses, and type methods with a minimal function mini-language.

**Architecture:** All three features follow the same pipeline: parse `.dgen` syntax into AST nodes, then generate Python code from those nodes. Feature 1 (Record layout) is a standalone addition to `dgen/layout.py` plus generator updates. Feature 2 (constraints) adds AST + parser + generator metadata. Feature 3 (methods) adds an expression/statement AST, parser, and Python codegen. Features 1 and 2 are independent; Feature 3 benefits from Feature 1.

**Tech Stack:** Python, pytest, ruff, ty

**Baseline:** 305 tests passing, ~1s.

---

## Feature 1: Multi-Field Struct Layout (`layout.Record`)

### Task 1: Add `layout.Record` class

**Files:**
- Modify: `dgen/layout.py` (add after `String` class, line ~193)
- Test: `test/test_layout_record.py` (create)

**Step 1: Write failing tests**

Create `test/test_layout_record.py`:

```python
"""Tests for layout.Record — fixed struct of named fields."""

from dgen.layout import Record, Int, Float64, Byte, Void


def test_record_byte_size():
    r = Record([("x", Int()), ("y", Float64())])
    assert r.byte_size == Int().byte_size + Float64().byte_size


def test_record_round_trip_dict():
    r = Record([("x", Int()), ("y", Float64())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"x": 42, "y": 3.14}, origins)
    result = r.to_json(buf, 0)
    assert result == {"x": 42, "y": 3.14}


def test_record_single_field():
    r = Record([("val", Int())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"val": 99}, origins)
    assert r.to_json(buf, 0) == {"val": 99}


def test_record_with_void():
    r = Record([("x", Int()), ("tag", Void())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"x": 7, "tag": None}, origins)
    result = r.to_json(buf, 0)
    assert result == {"x": 7, "tag": None}


def test_record_nested():
    inner = Record([("a", Int()), ("b", Int())])
    outer = Record([("first", inner), ("second", Float64())])
    buf = bytearray(outer.byte_size)
    origins: list[bytearray] = []
    outer.from_json(buf, 0, {"first": {"a": 1, "b": 2}, "second": 9.0}, origins)
    result = outer.to_json(buf, 0)
    assert result == {"first": {"a": 1, "b": 2}, "second": 9.0}
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_layout_record.py -v`
Expected: FAIL with `ImportError: cannot import name 'Record'`

**Step 3: Implement `Record`**

Add to `dgen/layout.py` after the `String` class:

```python
class Record(Layout):
    """Fixed struct of named fields, laid out sequentially."""

    def __init__(self, fields: list[tuple[str, Layout]]) -> None:
        self.fields = fields
        self._offsets: list[int] = []
        offset = 0
        for _, lay in fields:
            self._offsets.append(offset)
            offset += lay.byte_size
        # Build a struct format from concatenated field formats
        fmt = "".join(lay.struct.format for _, lay in fields)
        self.struct = Struct(fmt) if fmt else Struct("0s")

    def to_json(self, buf: bytes | bytearray, offset: int) -> dict[str, object]:
        result: dict[str, object] = {}
        for (name, lay), field_offset in zip(self.fields, self._offsets):
            result[name] = lay.to_json(buf, offset + field_offset)
        return result

    def from_json(
        self, buf: bytearray, offset: int, value: object, origins: list[bytearray]
    ) -> None:
        assert isinstance(value, dict)
        for (name, lay), field_offset in zip(self.fields, self._offsets):
            lay.from_json(buf, offset + field_offset, value[name], origins)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_layout_record.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: add layout.Record for multi-field struct layouts"
```

### Task 2: Generator emits `Record` for multi-field types

**Files:**
- Modify: `dgen/gen/python.py:165-166` (the `is_parametric` line and layout generation)
- Test: `test/test_gen_python.py`

**Step 1: Write failing test**

Add to `test/test_gen_python.py`:

```python
def test_generate_type_multi_field_record():
    """Multiple data fields generate a Record layout."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Point",
                data=[
                    DataField(name="x", type=TypeRef("Index")),
                    DataField(name="y", type=TypeRef("F64")),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class Point(Type):" in code
    assert "layout.Record" in code
    assert '"x"' in code
    assert '"y"' in code
    assert "Index.__layout__" in code
    assert "F64.__layout__" in code


def test_generate_type_multi_field_parametric_record():
    """Multi-field type with parametric fields generates Record property."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Pair",
                params=[ParamDecl(name="t", type=TypeRef("Type"))],
                data=[
                    DataField(name="first", type=TypeRef("t")),
                    DataField(name="second", type=TypeRef("Index")),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "def __layout__(self)" in code
    assert "layout.Record" in code
    assert "self.t.__layout__" in code
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_python.py::test_generate_type_multi_field_record test/test_gen_python.py::test_generate_type_multi_field_parametric_record -v`
Expected: FAIL — currently uses only first field

**Step 3: Update generator**

In `dgen/gen/python.py`, update the type layout generation. The current code at lines 165-222 handles single-field types. Replace the multi-field path:

1. Change the `is_parametric` check (line 166) to consider all data fields:

```python
is_parametric = bool(td.data) and any(
    _ref_has_params(df.type, param_map) for df in td.data
)
```

2. Update the static layout path (line 191-192) — when `len(td.data) > 1`, emit `Record`:

```python
elif td.data and not is_parametric:
    if len(td.data) == 1:
        body.append(f"    __layout__ = {_layout_expr(td.data[0].type, param_map)}")
    else:
        fields = ", ".join(
            f'("{df.name}", {_layout_expr(df.type, param_map)})'
            for df in td.data
        )
        body.append(f"    __layout__ = layout.Record([{fields}])")
```

3. Update the parametric layout property path (line 218-222) — same Record pattern but in the property:

```python
elif td.data and is_parametric:
    body.append("")
    body.append("    @property")
    body.append("    def __layout__(self) -> layout.Layout:")
    if len(td.data) == 1:
        body.append(f"        return {_layout_expr(td.data[0].type, param_map)}")
    else:
        fields = ", ".join(
            f'("{df.name}", {_layout_expr(df.type, param_map)})'
            for df in td.data
        )
        body.append(f"        return layout.Record([{fields}])")
```

**Step 4: Update the existing multi-field test**

The existing `test_generate_type_multiple_data_fields` asserts `__layout__ = Index.__layout__` (first field only). Update it to expect `layout.Record`:

```python
def test_generate_type_multiple_data_fields():
    """Multiple data fields generate a Record layout."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Pair",
                data=[
                    DataField(name="x", type=TypeRef("Index")),
                    DataField(name="y", type=TypeRef("F64")),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class Pair(Type):" in code
    assert "layout.Record" in code
```

**Step 5: Run all tests**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: generator emits layout.Record for multi-field types"
```

---

## Feature 2: `$X` Metavariables and `requires` Constraints

### Task 3: Add constraint AST nodes

**Files:**
- Modify: `dgen/gen/ast.py`
- Test: `test/test_gen_ast.py` (create — simple smoke test)

**Step 1: Write test**

Create `test/test_gen_ast.py`:

```python
"""Tests for .dgen AST dataclasses."""

from dgen.gen.ast import Constraint, OpDecl, TypeRef


def test_constraint_match():
    c = Constraint(kind="match", lhs="$X", pattern="Tensor")
    assert c.kind == "match"
    assert c.lhs == "$X"
    assert c.pattern == "Tensor"


def test_constraint_eq():
    c = Constraint(kind="eq", lhs="$X", rhs="$Result")
    assert c.kind == "eq"


def test_constraint_expr():
    c = Constraint(kind="expr", expr="axis < $X.rank")
    assert c.kind == "expr"


def test_op_with_constraints():
    op = OpDecl(
        name="tile",
        return_type=TypeRef("Type"),
        constraints=[
            Constraint(kind="match", lhs="$X", pattern="Tensor"),
        ],
    )
    assert len(op.constraints) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_ast.py -v`
Expected: FAIL with `ImportError: cannot import name 'Constraint'`

**Step 3: Add `Constraint` to AST**

In `dgen/gen/ast.py`, add after `StaticField`:

```python
@dataclass
class Constraint:
    """A requires clause on an op.

    Kinds:
    - "match": requires $Var ~= TypePattern  (lhs, pattern)
    - "eq":    requires $Var == $Var          (lhs, rhs)
    - "expr":  requires <expression>          (expr)
    """

    kind: str  # "match", "eq", or "expr"
    lhs: str | None = None
    pattern: str | None = None
    rhs: str | None = None
    expr: str | None = None
```

Add `constraints: list[Constraint] = field(default_factory=list)` to `OpDecl`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_ast.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: add Constraint dataclass to .dgen AST"
```

### Task 4: Parse `$X` metavariables and `requires` clauses

**Files:**
- Modify: `dgen/gen/parser.py`
- Test: `test/test_gen_parser.py`

**Step 1: Write failing tests**

Add to `test/test_gen_parser.py`:

```python
def test_parse_metavar_operand():
    """$X in operand position is a metavariable."""
    result = parse("op tile(x: $X) -> $Result\n")
    op = result.ops[0]
    assert op.operands[0].name == "x"
    assert op.operands[0].type.name == "$X"
    assert op.return_type.name == "$Result"


def test_parse_requires_match():
    src = "op tile(x: $X) -> $Result:\n    requires $X ~= Tensor\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "match"
    assert c.lhs == "$X"
    assert c.pattern == "Tensor"


def test_parse_requires_eq():
    src = "op sqrt(x: $X) -> $Result:\n    requires $X == $Result\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "eq"
    assert c.lhs == "$X"
    assert c.rhs == "$Result"


def test_parse_requires_expr():
    src = "op tile<axis: Index>(x: $X) -> $Result:\n    requires axis < $X.rank\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "expr"
    assert c.expr == "axis < $X.rank"


def test_parse_multiple_requires():
    src = """\
op tile<axis: Index>(x: $X) -> $Result:
    requires $X ~= Tensor
    requires $Result ~= Tensor
    requires $X.dtype == $Result.dtype
"""
    op = parse(src).ops[0]
    assert len(op.constraints) == 3
    assert op.constraints[0].kind == "match"
    assert op.constraints[1].kind == "match"
    assert op.constraints[2].kind == "expr"


def test_parse_requires_with_block():
    src = "op foo() -> Nil:\n    block body\n    requires $X ~= Tensor\n"
    op = parse(src).ops[0]
    assert op.blocks == ["body"]
    assert len(op.constraints) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_parser.py::test_parse_metavar_operand test/test_gen_parser.py::test_parse_requires_match -v`
Expected: FAIL — `$X` might parse fine (it's just a name), but `requires` is unrecognized in op body

**Step 3: Update parser**

The `$X` syntax should already parse as a `TypeRef(name="$X")` since `_parse_type_ref` treats it as a name. Verify this, and if not, allow `$` in type ref names.

In `_parse_op_body` (parser.py:212-228), add handling for `requires`:

```python
from dgen.gen.ast import Constraint

def _parse_op_body(self) -> tuple[list[str], list[str], list[Constraint]]:
    """Parse indented op body lines, return (block names, traits, constraints)."""
    blocks: list[str] = []
    traits: list[str] = []
    constraints: list[Constraint] = []
    while self.pos + 1 < len(self.lines):
        next_line = self.lines[self.pos + 1]
        if not next_line or not next_line[0].isspace():
            break
        self.pos += 1
        stripped = next_line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith("block "):
            blocks.append(stripped.split()[1])
        elif stripped.startswith("has trait "):
            traits.append(stripped.split()[2])
        elif stripped.startswith("requires "):
            constraints.append(_parse_constraint(stripped))
    return blocks, traits, constraints
```

Add the `_parse_constraint` function:

```python
def _parse_constraint(line: str) -> Constraint:
    """Parse a 'requires ...' line into a Constraint."""
    rest = line[9:]  # strip "requires "
    if " ~= " in rest:
        lhs, pattern = rest.split(" ~= ", 1)
        return Constraint(kind="match", lhs=lhs.strip(), pattern=pattern.strip())
    if " == " in rest:
        lhs, rhs = rest.split(" == ", 1)
        return Constraint(kind="eq", lhs=lhs.strip(), rhs=rhs.strip())
    return Constraint(kind="expr", expr=rest.strip())
```

Update `_parse_op` to pass constraints through:

```python
blocks: list[str] = []
traits: list[str] = []
constraints: list[Constraint] = []
if has_body:
    blocks, traits, constraints = self._parse_op_body()
return OpDecl(
    name=name,
    params=params,
    operands=operands,
    return_type=return_type,
    blocks=blocks,
    traits=traits,
    constraints=constraints,
)
```

Add `Constraint` to the import from `dgen.gen.ast` at the top of parser.py.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_parser.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: parse \$X metavariables and requires constraints in .dgen"
```

### Task 5: Generator emits `__constraints__` metadata

**Files:**
- Modify: `dgen/gen/python.py:277-286` (after `__operands__` / `__blocks__`)
- Test: `test/test_gen_python.py`

**Step 1: Write failing test**

Add to `test/test_gen_python.py`:

```python
from dgen.gen.ast import Constraint


def test_generate_op_constraints():
    """Constraints are emitted as __constraints__ metadata."""
    f = DgenFile(
        ops=[
            OpDecl(
                name="tile",
                operands=[OperandDecl(name="x", type=TypeRef("$X"))],
                return_type=TypeRef("$Result"),
                constraints=[
                    Constraint(kind="match", lhs="$X", pattern="Tensor"),
                    Constraint(kind="match", lhs="$Result", pattern="Tensor"),
                    Constraint(kind="eq", lhs="$X.dtype", rhs="$Result.dtype"),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "__constraints__" in code
    assert '"$X ~= Tensor"' in code
    assert '"$Result ~= Tensor"' in code
    assert '"$X.dtype == $Result.dtype"' in code


def test_generate_op_no_constraints():
    """Ops without constraints have no __constraints__."""
    f = DgenFile(
        ops=[
            OpDecl(
                name="nop",
                return_type=TypeRef("Nil"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "__constraints__" not in code
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_python.py::test_generate_op_constraints -v`
Expected: FAIL — no `__constraints__` in output

**Step 3: Update generator**

In `dgen/gen/python.py`, after the `__blocks__` emission (around line 286), add:

```python
if od.constraints:
    parts = []
    for c in od.constraints:
        if c.kind == "match":
            parts.append(f'"{c.lhs} ~= {c.pattern}"')
        elif c.kind == "eq":
            parts.append(f'"{c.lhs} == {c.rhs}"')
        else:
            parts.append(f'"{c.expr}"')
    body.append(f"    __constraints__ = ({', '.join(parts)},)")
```

Add `Constraint` to the import if not already imported (check — `python.py` only imports from `ast` selectively). Since we reference `od.constraints` which is `list[Constraint]`, we don't need to import the class into python.py — we just iterate it.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_python.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: generator emits __constraints__ metadata for requires clauses"
```

### Task 6: Add constraints to `.dgen` files

**Files:**
- Modify: `toy/dialects/toy.dgen`
- Modify: `dgen/dialects/builtin.dgen` (if any ops need constraints)

This task adds `requires` clauses to ops that have type relationships described in `docs/dialect-files.md`. For now, these are documentation/metadata only — they don't generate validation code.

**Step 1: Update `toy/dialects/toy.dgen`**

The design doc (line 91-99) shows `tile` with full constraints. Add constraints to toy ops where meaningful:

```dgen
from builtin import Index, Nil, F64, String
import affine

type Tensor<shape: affine.Shape, dtype: Type = F64>:
    # TODO: layout requires math.prod of shape dims, not yet expressible in .dgen.
    # Temporary workaround: __layout__ monkey-patched in toy/dialects/__init__.py

type InferredShapeTensor<dtype: Type = F64>:
    data: Nil

op transpose(input: Tensor) -> Tensor
op reshape(input: Tensor) -> Tensor
op mul(lhs: Tensor, rhs: Tensor) -> Tensor
op add(lhs: Tensor, rhs: Tensor) -> Tensor
op generic_call<callee: String>(args: list) -> Tensor
op concat<axis: Index>(lhs: Tensor, rhs: Tensor) -> Tensor:
    requires axis < lhs.shape.rank
op tile<count: Index>(input: Tensor) -> Tensor
op nonzero_count(input: Tensor) -> Index
op dim_size<axis: Index>(input: Tensor) -> Index:
    requires axis < input.shape.rank
op print(input: Tensor) -> Nil
```

**Step 2: Run all tests**

Run: `python -m pytest . -q`

**Step 3: Lint**

Run: `ruff format && ruff check --fix`

**Step 4: Commit**

```bash
jj commit -m "feat: add requires constraints to toy.dgen ops"
```

---

## Feature 3: Type Methods and Function Mini-Language

### Task 7: Add expression and statement AST nodes

**Files:**
- Modify: `dgen/gen/ast.py`
- Test: `test/test_gen_ast.py`

**Step 1: Write tests**

Add to `test/test_gen_ast.py`:

```python
from dgen.gen.ast import (
    Expr,
    NameExpr,
    AttrExpr,
    BinOpExpr,
    CallExpr,
    LiteralExpr,
    Assignment,
    ReturnStmt,
    ForStmt,
    IfStmt,
    MethodDecl,
    TypeDecl,
    TypeRef,
)


def test_expr_name():
    e = NameExpr(name="x")
    assert e.name == "x"


def test_expr_attr():
    e = AttrExpr(value=NameExpr(name="self"), attr="dims")
    assert e.attr == "dims"


def test_expr_binop():
    e = BinOpExpr(op="*", left=NameExpr(name="a"), right=NameExpr(name="b"))
    assert e.op == "*"


def test_expr_call():
    e = CallExpr(func=NameExpr(name="foo"), args=[NameExpr(name="x")])
    assert len(e.args) == 1


def test_expr_literal():
    e = LiteralExpr(value=42)
    assert e.value == 42


def test_assignment():
    s = Assignment(
        name="count",
        type=TypeRef("Index"),
        value=LiteralExpr(value=1),
    )
    assert s.name == "count"


def test_return_stmt():
    s = ReturnStmt(value=NameExpr(name="count"))
    assert isinstance(s.value, NameExpr)


def test_for_stmt():
    s = ForStmt(
        var="dim",
        iter=AttrExpr(value=NameExpr(name="self"), attr="dims"),
        body=[],
    )
    assert s.var == "dim"


def test_if_stmt():
    s = IfStmt(
        condition=BinOpExpr(
            op="==", left=NameExpr(name="x"), right=LiteralExpr(value=0)
        ),
        then_body=[ReturnStmt(value=LiteralExpr(value=0))],
        else_body=[],
    )
    assert s.condition.op == "=="


def test_method_decl():
    m = MethodDecl(
        name="num_elements",
        params=[],
        return_type=TypeRef("Index"),
        body=[ReturnStmt(value=LiteralExpr(value=1))],
    )
    assert m.name == "num_elements"


def test_type_with_methods():
    t = TypeDecl(
        name="Shape",
        methods=[
            MethodDecl(
                name="num_elements",
                params=[],
                return_type=TypeRef("Index"),
                body=[ReturnStmt(value=LiteralExpr(value=1))],
            )
        ],
    )
    assert len(t.methods) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_ast.py -v`
Expected: FAIL — none of these classes exist yet

**Step 3: Add expression/statement/method AST nodes**

In `dgen/gen/ast.py`, add:

```python
# --- Expression AST (mini-language) ---

@dataclass
class Expr:
    """Base class marker for expressions."""


@dataclass
class NameExpr(Expr):
    """A name reference: x, count, self."""
    name: str


@dataclass
class LiteralExpr(Expr):
    """An integer or float literal."""
    value: int | float


@dataclass
class AttrExpr(Expr):
    """Attribute access: value.attr."""
    value: Expr
    attr: str


@dataclass
class BinOpExpr(Expr):
    """Binary operation: left op right."""
    op: str  # +, -, *, //, <, ==, !=
    left: Expr
    right: Expr


@dataclass
class CallExpr(Expr):
    """Function/method call: func(args)."""
    func: Expr
    args: list[Expr]


# --- Statement AST ---

@dataclass
class Assignment:
    """name[: Type] = value."""
    name: str
    value: Expr
    type: TypeRef | None = None


@dataclass
class ReturnStmt:
    """return value."""
    value: Expr


@dataclass
class ForStmt:
    """for var in iter: body."""
    var: str
    iter: Expr
    body: list[Assignment | ReturnStmt | ForStmt | IfStmt]


@dataclass
class IfStmt:
    """if condition: then_body [else: else_body]."""
    condition: Expr
    then_body: list[Assignment | ReturnStmt | ForStmt | IfStmt]
    else_body: list[Assignment | ReturnStmt | ForStmt | IfStmt] = field(
        default_factory=list
    )
```

Type alias for statement union:

```python
Statement = Assignment | ReturnStmt | ForStmt | IfStmt
```

Add `MethodDecl`:

```python
@dataclass
class MethodDecl:
    """A method on a type: method name(self[, args]) -> ReturnType: body."""
    name: str
    params: list[ParamDecl]
    return_type: TypeRef
    body: list[Statement]
```

Add `methods: list[MethodDecl] = field(default_factory=list)` to `TypeDecl`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_ast.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: add expression, statement, and method AST nodes for mini-language"
```

### Task 8: Parse method declarations and mini-language body

**Files:**
- Modify: `dgen/gen/parser.py`
- Test: `test/test_gen_parser.py`

**Step 1: Write failing tests**

Add to `test/test_gen_parser.py`:

```python
def test_parse_method_simple_return():
    src = """\
type Foo:
    method size(self) -> Index:
        return 42
"""
    t = parse(src).types[0]
    assert len(t.methods) == 1
    m = t.methods[0]
    assert m.name == "size"
    assert m.return_type.name == "Index"
    assert len(m.body) == 1
    from dgen.gen.ast import ReturnStmt, LiteralExpr
    assert isinstance(m.body[0], ReturnStmt)
    assert isinstance(m.body[0].value, LiteralExpr)
    assert m.body[0].value.value == 42


def test_parse_method_with_assignment():
    src = """\
type Shape:
    method num_elements(self) -> Index:
        count: Index = 1
        return count
"""
    t = parse(src).types[0]
    m = t.methods[0]
    from dgen.gen.ast import Assignment, ReturnStmt, NameExpr, LiteralExpr
    assert len(m.body) == 2
    assert isinstance(m.body[0], Assignment)
    assert m.body[0].name == "count"
    assert m.body[0].type.name == "Index"
    assert isinstance(m.body[0].value, LiteralExpr)
    assert isinstance(m.body[1], ReturnStmt)
    assert isinstance(m.body[1].value, NameExpr)


def test_parse_method_for_loop():
    src = """\
type Shape<rank: Index>:
    dims: Array<Index, rank>

    method num_elements(self) -> Index:
        count: Index = 1
        for dim in self.dims:
            count = count * dim
        return count
"""
    t = parse(src).types[0]
    m = t.methods[0]
    from dgen.gen.ast import Assignment, ForStmt, ReturnStmt, BinOpExpr
    assert len(m.body) == 3
    assert isinstance(m.body[0], Assignment)
    assert isinstance(m.body[1], ForStmt)
    assert m.body[1].var == "dim"
    assert len(m.body[1].body) == 1
    assert isinstance(m.body[1].body[0], Assignment)
    assert isinstance(m.body[1].body[0].value, BinOpExpr)
    assert isinstance(m.body[2], ReturnStmt)


def test_parse_method_attr_access():
    src = """\
type Foo:
    method bar(self) -> Index:
        return self.x.y
"""
    t = parse(src).types[0]
    m = t.methods[0]
    from dgen.gen.ast import ReturnStmt, AttrExpr
    assert isinstance(m.body[0], ReturnStmt)
    assert isinstance(m.body[0].value, AttrExpr)
    assert m.body[0].value.attr == "y"


def test_parse_method_call():
    src = """\
type Foo:
    method bar(self) -> Index:
        return prod(self.dims)
"""
    t = parse(src).types[0]
    m = t.methods[0]
    from dgen.gen.ast import ReturnStmt, CallExpr
    assert isinstance(m.body[0], ReturnStmt)
    assert isinstance(m.body[0].value, CallExpr)
    assert len(m.body[0].value.args) == 1


def test_parse_method_if():
    src = """\
type Foo:
    method bar(self) -> Index:
        if self.x == 0:
            return 0
        return 1
"""
    t = parse(src).types[0]
    m = t.methods[0]
    from dgen.gen.ast import IfStmt, ReturnStmt
    assert len(m.body) == 2
    assert isinstance(m.body[0], IfStmt)
    assert len(m.body[0].then_body) == 1
    assert isinstance(m.body[1], ReturnStmt)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_parser.py::test_parse_method_simple_return -v`
Expected: FAIL — `method` keyword unrecognized in type body

**Step 3: Implement method and expression parsing**

Add an expression parser and method parser to `dgen/gen/parser.py`. The expression parser is a simple recursive descent for the mini-language:

```python
from dgen.gen.ast import (
    # ... existing imports ...
    Assignment,
    AttrExpr,
    BinOpExpr,
    CallExpr,
    Constraint,
    Expr,
    ForStmt,
    IfStmt,
    LiteralExpr,
    MethodDecl,
    NameExpr,
    ReturnStmt,
    Statement,
)


def _parse_expr(s: str) -> Expr:
    """Parse a mini-language expression string."""
    s = s.strip()
    # Try binary operators (lowest precedence first)
    for op in ("==", "!=", "<", ">", "<=", ">="):
        idx = _find_binop(s, op)
        if idx >= 0:
            left = _parse_expr(s[:idx])
            right = _parse_expr(s[idx + len(op):])
            return BinOpExpr(op=op, left=left, right=right)
    for op in ("+", "-"):
        idx = _find_binop(s, op)
        if idx >= 0:
            left = _parse_expr(s[:idx])
            right = _parse_expr(s[idx + len(op):])
            return BinOpExpr(op=op, left=left, right=right)
    for op in ("*", "//"):
        idx = _find_binop(s, op)
        if idx >= 0:
            left = _parse_expr(s[:idx])
            right = _parse_expr(s[idx + len(op):])
            return BinOpExpr(op=op, left=left, right=right)
    return _parse_postfix(s)


def _find_binop(s: str, op: str) -> int:
    """Find rightmost occurrence of op outside parentheses, return index or -1."""
    depth = 0
    # Search right-to-left for left-associativity
    i = len(s) - len(op)
    while i >= 0:
        ch = s[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif depth == 0 and s[i:i + len(op)] == op:
            # Don't match == inside !=, or // as part of something else
            if op == "=" and i > 0 and s[i - 1] in "!<>=":
                i -= 1
                continue
            if op in ("+", "-") and i == 0:
                i -= 1
                continue
            return i
        i -= 1
    return -1


def _parse_postfix(s: str) -> Expr:
    """Parse name, name.attr, name.attr.attr, name(args)."""
    s = s.strip()
    # Parenthesized expression
    if s.startswith("(") and s.endswith(")"):
        return _parse_expr(s[1:-1])
    # Literal integer
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return LiteralExpr(value=int(s))
    # Literal float
    try:
        return LiteralExpr(value=float(s))
    except ValueError:
        pass
    # Function call: name(args) or expr.method(args)
    if s.endswith(")") and "(" in s:
        paren = _find_matching_reverse(s, len(s) - 1, ")", "(")
        func_str = s[:paren].strip()
        args_str = s[paren + 1:-1].strip()
        func = _parse_postfix(func_str)
        args = [_parse_expr(a) for a in _split_commas(args_str)] if args_str else []
        return CallExpr(func=func, args=args)
    # Attribute access: a.b.c
    if "." in s:
        last_dot = s.rindex(".")
        value = _parse_postfix(s[:last_dot])
        attr = s[last_dot + 1:]
        return AttrExpr(value=value, attr=attr)
    # Simple name
    return NameExpr(name=s)


def _find_matching_reverse(s: str, start: int, close_ch: str, open_ch: str) -> int:
    """Find matching open bracket scanning backwards from start."""
    depth = 0
    for i in range(start, -1, -1):
        if s[i] == close_ch:
            depth += 1
        elif s[i] == open_ch:
            depth -= 1
            if depth == 0:
                return i
    raise SyntaxError(f"unmatched {close_ch} in {s!r}")
```

Add method body parsing:

```python
def _parse_method_body(
    lines: list[str], pos: int, base_indent: int
) -> tuple[list[Statement], int]:
    """Parse indented method body statements. Returns (statements, new_pos)."""
    stmts: list[Statement] = []
    while pos < len(lines):
        line = lines[pos]
        if not line or not line[0].isspace():
            break
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            pos += 1
            continue

        if stripped.startswith("return "):
            expr = _parse_expr(stripped[7:])
            stmts.append(ReturnStmt(value=expr))
            pos += 1
        elif stripped.startswith("for "):
            # for var in expr:
            rest = stripped[4:]
            in_idx = rest.index(" in ")
            var = rest[:in_idx].strip()
            iter_str = rest[in_idx + 4:].rstrip(":")
            iter_expr = _parse_expr(iter_str)
            pos += 1
            body, pos = _parse_method_body(lines, pos, indent)
            stmts.append(ForStmt(var=var, iter=iter_expr, body=body))
        elif stripped.startswith("if "):
            # if condition:
            cond_str = stripped[3:].rstrip(":")
            cond = _parse_expr(cond_str)
            pos += 1
            then_body, pos = _parse_method_body(lines, pos, indent)
            else_body: list[Statement] = []
            if pos < len(lines):
                next_stripped = lines[pos].strip()
                if next_stripped == "else:":
                    pos += 1
                    else_body, pos = _parse_method_body(lines, pos, indent)
            stmts.append(IfStmt(condition=cond, then_body=then_body, else_body=else_body))
        elif "=" in stripped and not stripped.startswith("="):
            # Assignment: name[: Type] = value  or  name = value
            eq_idx = stripped.index("=")
            # Make sure it's not == or !=
            if stripped[eq_idx:eq_idx + 2] != "==" and (eq_idx == 0 or stripped[eq_idx - 1] != "!"):
                lhs = stripped[:eq_idx].strip()
                rhs = stripped[eq_idx + 1:].strip()
                type_ref: TypeRef | None = None
                if ":" in lhs:
                    name, type_str = lhs.split(":", 1)
                    name = name.strip()
                    type_ref = _parse_type_ref(type_str.strip())
                else:
                    name = lhs
                stmts.append(Assignment(name=name, value=_parse_expr(rhs), type=type_ref))
                pos += 1
            else:
                raise SyntaxError(f"unexpected statement: {stripped!r}")
        else:
            raise SyntaxError(f"unexpected statement: {stripped!r}")
    return stmts, pos
```

In `_parse_type_body`, add handling for `method`:

```python
if stripped.startswith("method "):
    # method name(self[, args]) -> ReturnType:
    rest = stripped[7:].rstrip(":")
    # Parse name
    paren = rest.index("(")
    method_name = rest[:paren].strip()
    close_paren = rest.index(")")
    params_str = rest[paren + 1:close_paren]
    # Skip 'self' param
    param_names = [p.strip() for p in params_str.split(",") if p.strip() != "self"]
    params = _parse_params(", ".join(param_names)) if param_names else []
    # Return type
    ret_str = rest[close_paren + 1:].strip()
    ret_type = _parse_type_ref(ret_str.split("->")[1].strip()) if "->" in ret_str else TypeRef("Nil")
    self.pos += 1
    # Determine method body indent
    method_indent = len(next_line) - len(next_line.lstrip())
    body, new_pos = _parse_method_body(self.lines, self.pos + 1, method_indent)
    self.pos = new_pos - 1  # -1 because outer loop does pos += 1
    methods.append(MethodDecl(name=method_name, params=params, return_type=ret_type, body=body))
    continue
```

Add `methods` list to the `_parse_type_body` return and to the `_parse_type` call site. Update the return type of `_parse_type_body` to include `list[MethodDecl]`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_parser.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: parse type methods and mini-language expressions in .dgen"
```

### Task 9: Generator emits Python methods from MethodDecl

**Files:**
- Modify: `dgen/gen/python.py`
- Test: `test/test_gen_python.py`

**Step 1: Write failing tests**

Add to `test/test_gen_python.py`:

```python
from dgen.gen.ast import (
    Assignment,
    AttrExpr,
    BinOpExpr,
    CallExpr,
    ForStmt,
    LiteralExpr,
    MethodDecl,
    NameExpr,
    ReturnStmt,
)


def test_generate_method_simple():
    """Simple method generates a Python method."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Foo",
                layout="Void",
                methods=[
                    MethodDecl(
                        name="size",
                        params=[],
                        return_type=TypeRef("Index"),
                        body=[ReturnStmt(value=LiteralExpr(value=42))],
                    )
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "def size(self):" in code
    assert "return 42" in code


def test_generate_method_for_loop():
    """Method with for loop generates correct Python."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Shape",
                params=[ParamDecl(name="rank", type=TypeRef("Index"))],
                data=[
                    DataField(
                        name="dims",
                        type=TypeRef("Array", [TypeRef("Index"), TypeRef("rank")]),
                    )
                ],
                methods=[
                    MethodDecl(
                        name="num_elements",
                        params=[],
                        return_type=TypeRef("Index"),
                        body=[
                            Assignment(
                                name="count",
                                type=TypeRef("Index"),
                                value=LiteralExpr(value=1),
                            ),
                            ForStmt(
                                var="dim",
                                iter=AttrExpr(
                                    value=NameExpr(name="self"), attr="dims"
                                ),
                                body=[
                                    Assignment(
                                        name="count",
                                        value=BinOpExpr(
                                            op="*",
                                            left=NameExpr(name="count"),
                                            right=NameExpr(name="dim"),
                                        ),
                                    )
                                ],
                            ),
                            ReturnStmt(value=NameExpr(name="count")),
                        ],
                    )
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "def num_elements(self):" in code
    assert "count = 1" in code
    assert "for dim in self.dims:" in code
    assert "count = count * dim" in code
    assert "return count" in code


def test_generate_method_call():
    """Method with function call."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Foo",
                layout="Void",
                methods=[
                    MethodDecl(
                        name="total",
                        params=[],
                        return_type=TypeRef("Index"),
                        body=[
                            ReturnStmt(
                                value=CallExpr(
                                    func=NameExpr(name="prod"),
                                    args=[
                                        AttrExpr(
                                            value=NameExpr(name="self"), attr="dims"
                                        )
                                    ],
                                )
                            )
                        ],
                    )
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "return prod(self.dims)" in code
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_gen_python.py::test_generate_method_simple -v`
Expected: FAIL — no method in generated output

**Step 3: Add expression and statement codegen to generator**

In `dgen/gen/python.py`, add helper functions:

```python
def _emit_expr(expr: Expr) -> str:
    """Convert an Expr AST node to a Python expression string."""
    if isinstance(expr, NameExpr):
        return expr.name
    if isinstance(expr, LiteralExpr):
        return repr(expr.value)
    if isinstance(expr, AttrExpr):
        return f"{_emit_expr(expr.value)}.{expr.attr}"
    if isinstance(expr, BinOpExpr):
        return f"{_emit_expr(expr.left)} {expr.op} {_emit_expr(expr.right)}"
    if isinstance(expr, CallExpr):
        args = ", ".join(_emit_expr(a) for a in expr.args)
        return f"{_emit_expr(expr.func)}({args})"
    raise ValueError(f"unknown expr type: {type(expr)}")


def _emit_stmts(stmts: list[Statement], indent: str) -> Iterator[str]:
    """Emit Python statements at the given indent level."""
    for stmt in stmts:
        if isinstance(stmt, ReturnStmt):
            yield f"{indent}return {_emit_expr(stmt.value)}"
        elif isinstance(stmt, Assignment):
            yield f"{indent}{stmt.name} = {_emit_expr(stmt.value)}"
        elif isinstance(stmt, ForStmt):
            yield f"{indent}for {stmt.var} in {_emit_expr(stmt.iter)}:"
            yield from _emit_stmts(stmt.body, indent + "    ")
        elif isinstance(stmt, IfStmt):
            yield f"{indent}if {_emit_expr(stmt.condition)}:"
            yield from _emit_stmts(stmt.then_body, indent + "    ")
            if stmt.else_body:
                yield f"{indent}else:"
                yield from _emit_stmts(stmt.else_body, indent + "    ")
```

Then in the type generation section of `_generate`, after static fields and before the `if not body` check, add method generation:

```python
for method in td.methods:
    body.append("")
    body.append(f"    def {method.name}(self):")
    body.extend(_emit_stmts(method.body, "        "))
```

Add the necessary imports from ast at the top of python.py:

```python
from dgen.gen.ast import (
    Assignment,
    AttrExpr,
    BinOpExpr,
    CallExpr,
    DgenFile,
    Expr,
    ForStmt,
    IfStmt,
    LiteralExpr,
    MethodDecl,
    NameExpr,
    OperandDecl,
    ParamDecl,
    ReturnStmt,
    Statement,
    TypeDecl,
    TypeRef,
)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_gen_python.py -v`

**Step 5: Run full suite**

Run: `python -m pytest . -q`

**Step 6: Lint**

Run: `ruff format && ruff check --fix`

**Step 7: Commit**

```bash
jj commit -m "feat: generator emits Python methods from MethodDecl AST"
```

### Task 10: Add `num_elements` method to `affine.dgen`, remove monkey-patch

**Files:**
- Modify: `toy/dialects/affine.dgen`
- Modify: `toy/dialects/__init__.py` (remove Tensor.__layout__ monkey-patch once method works)

**Step 1: Add method to affine.dgen**

Update `toy/dialects/affine.dgen`:

```dgen
from builtin import Index, Nil, F64, HasSingleBlock, Array, Pointer

type Shape<rank: Index>:
    dims: Array<Index, rank>

    method num_elements(self) -> Index:
        count: Index = 1
        for dim in self.dims:
            count = count * dim
        return count

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

**Step 2: Regenerate `toy/dialects/affine.py` and verify**

Run: `python -m dgen.gen toy/dialects/affine.dgen --import builtin=dgen.dialects.builtin > /tmp/affine_gen.py && diff /tmp/affine_gen.py toy/dialects/affine.py`

If the generated file has the `num_elements` method, update `toy/dialects/affine.py`.

**Step 3: Run all tests**

Run: `python -m pytest . -q`

The `num_elements` method on Shape is not currently used by the pipeline directly (the monkey-patched `Tensor.__layout__` uses `prod(shape.to_json())` instead), so existing tests should still pass. The method is available for future use.

**Step 4: Lint**

Run: `ruff format && ruff check --fix`

**Step 5: Commit**

```bash
jj commit -m "feat: add num_elements method to Shape in affine.dgen"
```

### Task 11: Final verification and cleanup

**Files:**
- Test: all

**Step 1: Run full test suite**

Run: `python -m pytest . -q`

**Step 2: Run type checker**

Run: `ty check`

Review any new type errors from the added code.

**Step 3: Run linter**

Run: `ruff format && ruff check --fix`

**Step 4: Verify .dgen round-trips**

Run each generator to verify outputs are valid Python:

```bash
python -m dgen.gen dgen/dialects/builtin.dgen 2>&1 | python -c "import sys; compile(sys.stdin.read(), '<stdin>', 'exec'); print('OK')"
python -m dgen.gen dgen/dialects/llvm.dgen --import builtin=dgen.dialects.builtin 2>&1 | python -c "import sys; compile(sys.stdin.read(), '<stdin>', 'exec'); print('OK')"
python -m dgen.gen toy/dialects/affine.dgen --import builtin=dgen.dialects.builtin 2>&1 | python -c "import sys; compile(sys.stdin.read(), '<stdin>', 'exec'); print('OK')"
python -m dgen.gen toy/dialects/toy.dgen --import builtin=dgen.dialects.builtin --import affine=toy.dialects.affine 2>&1 | python -c "import sys; compile(sys.stdin.read(), '<stdin>', 'exec'); print('OK')"
```

**Step 5: Commit any fixups**

```bash
jj commit -m "chore: final cleanup for .dgen gap closure"
```

---

## Summary

| Task | Feature | What |
|------|---------|------|
| 1 | Record layout | `layout.Record` class |
| 2 | Record layout | Generator emits `Record` for multi-field types |
| 3 | Constraints | `Constraint` AST node |
| 4 | Constraints | Parse `$X` and `requires` |
| 5 | Constraints | Generator emits `__constraints__` |
| 6 | Constraints | Add constraints to `.dgen` files |
| 7 | Methods | Expression/statement/method AST |
| 8 | Methods | Parse methods and mini-language |
| 9 | Methods | Generator emits Python methods |
| 10 | Methods | Add `num_elements` to `affine.dgen` |
| 11 | Cleanup | Full verification pass |

## Deferred / Future Work (not in this plan)

- **Runtime constraint validation** — generating actual Python validation code from `__constraints__`, rather than storing them as metadata strings
- **`else` in for loops** — Python `for/else` pattern, not needed by the mini-language
- **Method params beyond self** — the mini-language only uses `self` methods for now
- **Tensor.__layout__ as a method** — requires expression-level `prod()` which needs a builtin function registry in the mini-language
- **Removing remaining monkey-patches** — `Shape.for_value`, `Tensor.unpack_shape`, `DimSizeOp.resolve_constant` use Python-specific APIs outside the mini-language scope
