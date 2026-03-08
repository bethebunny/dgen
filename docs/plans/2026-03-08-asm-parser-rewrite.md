# ASM Parser Rewrite — Grammar Classes + Direct-to-IR

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic `IRParser` class with composable grammar classes that parse directly to IR types, eliminating the intermediate Python-value phase.

**Architecture:** `ASMParser` provides low-level tokenizing + `read(G)/try_read(G)` protocol. Grammar classes (`TypeExpression`, `OpExpression`, etc.) implement `read(cls, parser)` classmethods that return IR objects directly. Type-directed parsing replaces `_wrap_constant`/`_expand_list_sugar`/`pending_ops`.

**Tech Stack:** Python, re, dataclasses. No new dependencies.

---

### Task 1: ASMParser core tokenizer

Build the low-level parser with `read`/`try_read` protocol, alongside the existing `IRParser`.

**Files:**
- Modify: `dgen/asm/parser.py` (add new class, keep old code)
- Create: `test/test_asm_parser.py`

**Step 1: Write failing tests for ASMParser**

```python
# test/test_asm_parser.py
from dgen.asm.parser import ASMParser


def test_read_string_literal():
    p = ASMParser("= hello")
    assert p.read("=") == "="
    assert p.read_token(_IDENT) == "hello"


def test_try_read_string_success():
    p = ASMParser("= x")
    assert p.try_read("=") == "="


def test_try_read_string_failure():
    p = ASMParser("+ x")
    assert p.try_read("=") is None
    # Position unchanged
    assert p.try_read("+") == "+"


def test_done():
    p = ASMParser("")
    assert p.done
    p2 = ASMParser("x")
    assert not p2.done
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_asm_parser.py -q`
Expected: ImportError — ASMParser not defined

**Step 3: Implement ASMParser**

Add to `dgen/asm/parser.py` (below existing code):

```python
class Namespace:
    """Op and type registries, populated by imports."""

    def __init__(self) -> None:
        self.ops: dict[str, type[Op]] = {}
        self.types: dict[str, type[Type]] = {}
        # Implicit: from builtin import *
        builtin_dialect = Dialect.get("builtin")
        self.ops.update(builtin_dialect.ops)
        self.types.update(builtin_dialect.types)

    def import_dialect(self, dialect_name: str) -> None:
        for _pfx in ("toy.dialects", "dgen.dialects"):
            try:
                importlib.import_module(f"{_pfx}.{dialect_name}")
                break
            except ModuleNotFoundError:
                continue
        d = Dialect.get(dialect_name)
        for op_name, cls in d.ops.items():
            self.ops[f"{dialect_name}.{op_name}"] = cls
        for tname, tcls in d.types.items():
            self.types[f"{dialect_name}.{tname}"] = tcls


class ASMParser:
    """Low-level tokenizer with grammar-class dispatch."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.namespace = Namespace()
        self.name_table: dict[str, Value] = {}

    @property
    def done(self) -> bool:
        self._skip_whitespace_and_newlines()
        return self.pos >= len(self.text)

    def read(self, grammar: type | str) -> object:
        """Read a grammar element or literal string. Raises on failure."""
        if isinstance(grammar, str):
            return self._read_punct(grammar)
        return grammar.read(self)

    def try_read(self, grammar: type | str) -> object | None:
        """Try to read, backtracking on failure."""
        saved = self.pos
        try:
            return self.read(grammar)
        except (RuntimeError, AssertionError):
            self.pos = saved
            return None

    # -- Low-level token methods (carried over from IRParser) --

    def peek(self) -> str: ...
    def read_token(self, regex) -> str | None: ...
    def expect_token(self, regex, name) -> str: ...
    # etc. — same as current IRParser methods
```

The low-level methods (`peek`, `skip_whitespace`, `read_token`, `expect_token`, `parse_punct`, etc.) are carried directly from `IRParser` with minimal renaming.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_asm_parser.py -q`
Expected: PASS

**Step 5: Commit**

`jj commit -m "add ASMParser core tokenizer with read/try_read protocol"`

---

### Task 2: Primitive grammar classes — SSAName, QualifiedName

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
from dgen.asm.parser import ASMParser, SSAName, QualifiedName


def test_ssa_name():
    p = ASMParser("%foo")
    assert p.read(SSAName) == "foo"


def test_qualified_name():
    p = ASMParser("toy.transpose")
    assert p.read(QualifiedName) == "toy.transpose"


def test_qualified_name_simple():
    p = ASMParser("Index")
    assert p.read(QualifiedName) == "Index"
```

**Step 2: Run to verify failure**

**Step 3: Implement**

```python
class SSAName:
    """Parse %name, return the name string."""
    @classmethod
    def read(cls, parser: ASMParser) -> str:
        token = parser.expect_token(_SSA, "SSA name")
        return token[1:]


class QualifiedName:
    """Parse a possibly-qualified name: ident or ident.ident."""
    @classmethod
    def read(cls, parser: ASMParser) -> str:
        return parser.expect_token(_QUALIFIED, "qualified name")
```

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add SSAName and QualifiedName grammar classes"`

---

### Task 3: ImportStatement grammar

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
def test_import_statement():
    p = ASMParser("import toy\n")
    p.read(ImportStatement)
    assert "toy.transpose" in p.namespace.ops


def test_import_builtin_noop():
    p = ASMParser("from builtin import Index, F64\n")
    p.read(ImportStatement)
    # builtin is already imported, this is a no-op
    assert "Index" in p.namespace.types


def test_try_read_import_fails_on_non_import():
    p = ASMParser("%x : Index = 42\n")
    assert p.try_read(ImportStatement) is None
    assert p.pos == 0  # backtracked
```

**Step 2: Run to verify failure**

**Step 3: Implement**

```python
class ImportStatement:
    """Parse import headers. Side-effects namespace."""

    @classmethod
    def read(cls, parser: ASMParser) -> ImportStatement:
        parser._skip_whitespace()
        word = parser.expect_token(_IDENT, "import keyword")

        if word == "from":
            mod_name = parser.expect_token(_IDENT, "module name")
            if mod_name != "builtin":
                raise RuntimeError(f"Expected 'builtin' after 'from', got '{mod_name}'")
            kw = parser.expect_token(_IDENT, "'import'")
            if kw != "import":
                raise RuntimeError(f"Expected 'import', got '{kw}'")
            parser._skip_line()
            return cls()

        if word == "import":
            dialect_name = parser.expect_token(_IDENT, "dialect name")
            parser._skip_line()
            parser.namespace.import_dialect(dialect_name)
            return cls()

        raise RuntimeError(f"Expected 'import' or 'from', got '{word}'")
```

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add ImportStatement grammar class"`

---

### Task 4: TypeExpression — type-directed type parsing

This is the core of the direct-to-IR approach. Returns `Type | Value[TypeType]`.

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
from dgen.dialects.builtin import Index, F64, Nil, Array


def test_parse_simple_type():
    p = ASMParser("Index")
    result = p.read(TypeExpression)
    assert isinstance(result, Index)


def test_parse_nil():
    p = ASMParser("Nil")
    result = p.read(TypeExpression)
    assert isinstance(result, Nil)


def test_parse_parameterized_type():
    p = ASMParser("Array<Index, 4>")
    result = p.read(TypeExpression)
    assert isinstance(result, Array)


def test_parse_ssa_type_ref():
    p = ASMParser("%t")
    from dgen.type import TypeType, Value
    # Pre-populate name table with a TypeType value
    from dgen.block import BlockArgument
    arg = BlockArgument(name="t", type=TypeType(concrete=Index()))
    p.name_table["t"] = arg
    result = p.read(TypeExpression)
    assert result is arg
```

**Step 2: Run to verify failure**

**Step 3: Implement**

`TypeExpression.read` dispatches on first token:
- `%` → SSA reference (must be TypeType-typed)
- `{` → dict literal (for TypeType constants)
- identifier → namespace lookup → if `__params__`, parse `<param, param, ...>` recursively

For parameterized types, each `__params__` field has a declared type. The parser reads each parameter as:
- If param type is a `Type` subclass → recurse into `TypeExpression`
- If param type is `Index`/`F64`/etc (scalar) → parse a literal and wrap as `Constant`

This is the type-directed parsing: `Array<Index, 4>` knows field 0 expects `TypeType` (→ recurse) and field 1 expects `Index` (→ parse int, wrap as `Index().constant(4)`).

```python
class TypeExpression:
    """Parse a type expression, returning Type | Value[TypeType] directly."""

    @classmethod
    def read(cls, parser: ASMParser) -> Type | Value:
        parser._skip_whitespace()
        c = parser.peek()

        if c == "%":
            name = parser.read(SSAName)
            val = parser.name_table.get(name)
            if val is None:
                raise RuntimeError(f"Unknown SSA name: %{name}")
            return val

        if c == "{":
            return cls._read_dict_type(parser)

        name = parser.read(QualifiedName)
        type_cls = parser.namespace.types.get(name)
        if type_cls is None:
            raise RuntimeError(f"Unknown type: {name}")

        if not type_cls.__params__:
            return type_cls()

        # All fields have defaults and no '<' follows → use defaults
        if cls._all_defaults(type_cls) and parser.peek() != "<":
            return type_cls()

        parser.read("<")
        kwargs = cls._read_params(parser, type_cls)
        parser.read(">")
        return type_cls(**kwargs)

    @classmethod
    def _read_params(cls, parser: ASMParser, type_cls: type[Type]) -> dict[str, object]:
        """Type-directed parameter parsing."""
        kwargs: dict[str, object] = {}
        for i, (field_name, field_type) in enumerate(type_cls.__params__):
            if i > 0:
                parser.read(",")
            if issubclass(field_type, TypeType):
                kwargs[field_name] = cls.read(parser)
            else:
                # Scalar param: parse literal, wrap as Constant
                kwargs[field_name] = _read_typed_value(parser, field_type)
        return kwargs
```

`_read_typed_value(parser, field_type)` is a helper that reads a literal matching the expected type and wraps it as a `Constant`. Shared between `TypeExpression` and `OpExpression`.

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add TypeExpression grammar class with type-directed parsing"`

---

### Task 5: Type-directed value reading — `_read_typed_value`

Shared helper for reading a value when the expected type is known. Replaces `_wrap_constant`.

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
def test_read_typed_int():
    p = ASMParser("42")
    from dgen.dialects.builtin import Index
    result = _read_typed_value(p, Index)
    assert isinstance(result, Constant)
    assert result.__constant__.to_json() == 42


def test_read_typed_float():
    p = ASMParser("3.14")
    from dgen.dialects.builtin import F64
    result = _read_typed_value(p, F64)
    assert isinstance(result, Constant)
    assert result.__constant__.to_json() == 3.14


def test_read_typed_ssa_ref():
    """SSA reference passes through regardless of expected type."""
    p = ASMParser("%x")
    from dgen.block import BlockArgument
    from dgen.dialects.builtin import F64
    arg = BlockArgument(name="x", type=F64())
    p.name_table["x"] = arg
    result = _read_typed_value(p, F64)
    assert result is arg


def test_read_typed_list():
    """[1, 2, 3] for a List<Index> field → PackOp of ConstantOps."""
    p = ASMParser("[1, 2, 3]")
    from dgen.dialects.builtin import Index
    result = _read_typed_value(p, Index)
    assert isinstance(result, list)  # list of Constant values
```

**Step 2: Run to verify failure**

**Step 3: Implement**

```python
def _read_typed_value(parser: ASMParser, expected_type: type[Type]) -> Value | list[Value]:
    """Read a value of the expected type, wrapping literals as Constants."""
    parser._skip_whitespace()
    c = parser.peek()

    if c == "%":
        name = parser.read(SSAName)
        val = parser.name_table.get(name)
        if val is None:
            val = Value(name=name, type=builtin.Nil())  # forward ref
            parser.name_table[name] = val
        return val

    if c == "[":
        return _read_typed_list(parser, expected_type)

    if c == "(":
        parser._expect("()")
        return builtin.Nil()

    if c == '"':
        s = parser._parse_string_literal()
        return expected_type().constant(s)

    if c == "{":
        d = _read_dict(parser)
        return expected_type().constant(d)

    if c in "-0123456789":
        token = parser.expect_token(_NUMBER, "number")
        val = float(token) if "." in token else int(token)
        return expected_type().constant(val)

    # Must be a type name — read as TypeExpression
    return TypeExpression.read(parser)
```

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add _read_typed_value for type-directed literal parsing"`

---

### Task 6: OpExpression — type-directed op parsing

Returns a partial that `OpStatement` finishes with name + type.

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
def test_parse_op_expression():
    p = ASMParser("toy.transpose(%a)")
    # Set up namespace and name table
    import toy.dialects.toy
    p.namespace.import_dialect("toy")
    from dgen.block import BlockArgument
    from toy.dialects.toy import InferredShapeTensor
    arg = BlockArgument(name="a", type=InferredShapeTensor(element_type=F64()))
    p.name_table["a"] = arg

    partial = p.read(OpExpression)
    op = partial.build(name="0", type=InferredShapeTensor(element_type=F64()))
    assert op.dialect.name == "toy"
    assert op.name == "0"
```

**Step 2: Run to verify failure**

**Step 3: Implement**

`OpExpression.read`:
1. Parse qualified name → look up op class in `namespace.ops`
2. If `cls.__params__` → parse `<param, ...>` type-directed
3. Parse `(operand, ...)` type-directed using `cls.__operands__`
4. If `cls.__blocks__` → parse block args + indented blocks
5. Return a `PartialOp(cls, kwargs)` with a `.build(name, type)` method

```python
@dataclass
class PartialOp:
    cls: type[Op]
    kwargs: dict[str, object]

    def build(self, *, name: str | None, type: Value[TypeType] | None) -> Op:
        if name is not None:
            self.kwargs["name"] = name
        if type is not None:
            self.kwargs["type"] = type
        return self.cls(**self.kwargs)


class OpExpression:
    @classmethod
    def read(cls, parser: ASMParser) -> PartialOp:
        name = parser.read(QualifiedName)
        op_cls = parser.namespace.ops.get(name)
        if op_cls is None:
            raise RuntimeError(f"Unknown op: {name}")

        kwargs: dict[str, object] = {}

        # Params in <...>
        if op_cls.__params__:
            parser.read("<")
            for i, (f_name, f_type) in enumerate(op_cls.__params__):
                if i > 0:
                    parser.read(",")
                if issubclass(f_type, TypeType):
                    kwargs[f_name] = TypeExpression.read(parser)
                else:
                    kwargs[f_name] = _read_typed_value(parser, f_type)
            parser.read(">")

        # Operands in (...)
        parser.read("(")
        for i, (f_name, f_type) in enumerate(op_cls.__operands__):
            if i > 0:
                parser.read(",")
            kwargs[f_name] = _read_typed_value(parser, f_type)
        parser.read(")")

        # Blocks
        if op_cls.__blocks__:
            # Parse block args + indented bodies
            ...  # See Task 7

        return PartialOp(op_cls, kwargs)
```

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add OpExpression grammar class with type-directed parsing"`

---

### Task 7: BlockArgs and IndentedBlock

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

Test parsing block args `(%x: F64, %y: Index)` and indented blocks of ops.

Use an IR snippet with a function to test end-to-end:

```python
def test_parse_block_args():
    p = ASMParser("(%x: F64, %y: Index)")
    args = p.read(BlockArgs)
    assert len(args) == 2
    assert args[0].name == "x"
    assert isinstance(args[0].type, F64)
```

**Step 2: Run to verify failure**

**Step 3: Implement**

```python
class BlockArgs:
    """Parse (%name: Type, ...) or empty ()."""
    @classmethod
    def read(cls, parser: ASMParser) -> list[BlockArgument]:
        parser.read("(")
        args: list[BlockArgument] = []
        if parser.try_read(")") is not None:
            return args
        args.append(cls._read_one(parser))
        while parser.try_read(",") is not None:
            args.append(cls._read_one(parser))
        parser.read(")")
        return args

    @classmethod
    def _read_one(cls, parser: ASMParser) -> BlockArgument:
        name = parser.read(SSAName)
        parser.read(":")
        type_ = TypeExpression.read(parser)
        arg = BlockArgument(name=name, type=type_)
        parser.name_table[name] = arg
        return arg
```

`IndentedBlock` carries over the indent-tracking logic from `IRParser._parse_block` / `parse_indented_block`, but calls `OpStatement.read` instead of `parse_op`.

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add BlockArgs and IndentedBlock grammar classes"`

---

### Task 8: OpStatement — assembles the full op

**Files:**
- Modify: `dgen/asm/parser.py`
- Modify: `test/test_asm_parser.py`

**Step 1: Write failing tests**

```python
def test_parse_op_statement():
    p = ASMParser("%0 : Index = 42")
    op = p.read(OpStatement)
    assert op.name == "0"
    assert isinstance(op, ConstantOp)


def test_parse_op_statement_with_op():
    p = ASMParser("import toy\n\n%0 : toy.InferredShapeTensor<F64> = toy.transpose(%a)")
    p.read(ImportStatement)
    from dgen.block import BlockArgument
    from toy.dialects.toy import InferredShapeTensor
    arg = BlockArgument(name="a", type=InferredShapeTensor(element_type=F64()))
    p.name_table["a"] = arg
    op = p.read(OpStatement)
    assert op.name == "0"
```

**Step 2: Run to verify failure**

**Step 3: Implement**

```python
class OpStatement:
    """Parse %name [: Type] = expr and return the assembled Op."""

    @classmethod
    def read(cls, parser: ASMParser) -> Op:
        name = parser.read(SSAName)
        pre_type = None
        if parser.try_read(":") is not None:
            pre_type = TypeExpression.read(parser)
        parser.read("=")

        # Implicit constant: starts with literal syntax
        parser._skip_whitespace()
        c = parser.peek()
        if c in "{[-0123456789":
            if pre_type is None:
                raise RuntimeError(f"constant %{name} missing type annotation")
            value = _read_constant_value(parser, pre_type)
            op = ConstantOp(name=name, value=value, type=pre_type)
            parser.name_table[name] = op
            return op

        partial = parser.read(OpExpression)
        op = partial.build(name=name, type=pre_type)
        parser.name_table[name] = op
        return op
```

`_read_constant_value` reads the RHS of an implicit constant (`[1.0, 2.0]`, `42`, `{"tag": ...}`) as a raw Python value (since `ConstantOp.value` stores raw Python, not IR).

**Step 4: Run tests, verify pass**
**Step 5: Commit**

`jj commit -m "add OpStatement grammar class"`

---

### Task 9: Wire up parse_module + parse_value

Replace the old `parse_module` implementation with the new grammar classes.

**Files:**
- Modify: `dgen/asm/parser.py`

**Step 1: Replace parse_module**

```python
def parse_module(text: str) -> Module:
    parser = ASMParser(text)
    while parser.try_read(ImportStatement) is not None:
        pass  # ImportStatement side-effects namespace
    functions: list[builtin.FunctionOp] = []
    while not parser.done:
        op = parser.read(OpStatement)
        assert isinstance(op, builtin.FunctionOp)
        functions.append(op)
    return Module(functions=functions)


def parse_value(text: str, type: Type) -> Value:
    """Parse a typed value from ASM text. Used by Memory.from_asm."""
    parser = ASMParser(text)
    return _read_typed_value(parser, type.__class__)
```

**Step 2: Run FULL test suite**

Run: `python -m pytest . -q`
Expected: All 404 tests pass (this is the critical validation — existing round-trip tests cover all syntax)

**Step 3: Fix any failures**

If tests fail, debug against the specific IR patterns that broke. The round-trip tests are very specific about exact output matching, so this may require iterating.

**Step 4: Commit**

`jj commit -m "wire up parse_module to use new grammar classes"`

---

### Task 10: Update external callers + remove old code

**Files:**
- Modify: `dgen/type.py` (Memory.from_asm)
- Modify: `toy/cli.py` (_parse_arg)
- Modify: `test/test_type_values.py` (update imports)
- Modify: `toy/test/test_type_roundtrip.py` (update imports)
- Modify: `dgen/asm/parser.py` (remove IRParser, parse_expr, helpers)

**Step 1: Update Memory.from_asm**

```python
# dgen/type.py line 245-251
@classmethod
def from_asm(cls, type: Type, text: str) -> Memory:
    """Create Memory from a Type and an ASM literal string."""
    from dgen.asm.parser import parse_value
    val = parse_value(text, type)
    return val.__constant__  # or cls.from_value(type, val)
```

**Step 2: Update cli.py**

Replace `parse_expr(IRParser(arg))` with `ast.literal_eval(arg)`:

```python
import ast

def _parse_arg(arg: str) -> object:
    """Parse a string arg to a Python value."""
    return ast.literal_eval(arg)
```

**Step 3: Update test imports**

- `test/test_type_values.py`: Remove `IRParser, parse_expr` imports. Replace `IRParser(text)` + `parse_expr(parser)` usage with equivalent new API calls.
- `toy/test/test_type_roundtrip.py`: Replace `_parse_type` helper to use new API.

**Step 4: Remove old code from parser.py**

Delete: `IRParser` class, `parse_expr`, `_resolve_or_create`, `_wrap_constant`, `_expand_list_sugar`, `_parse_fields_from_exprs`, `parse_op_fields`.

**Step 5: Run full test suite**

Run: `python -m pytest . -q`
Expected: All 404 tests pass

**Step 6: Run formatting and type checker**

Run: `ruff format && ruff check --fix && ty check`

**Step 7: Commit**

`jj commit -m "remove old IRParser, update callers to new grammar-class parser"`
