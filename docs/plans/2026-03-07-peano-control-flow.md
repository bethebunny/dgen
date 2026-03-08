# Control Flow, Call, and Traits Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build if/else, call, equal_index, subtract_index ops and marker traits, demonstrated via a recursive Peano arithmetic test.

**Architecture:** Add new ops to builtin.dgen and regenerate. Extend the ASM parser/formatter to handle multi-block ops. Add trait registration to Dialect. Extend codegen and staging for new ops and multi-function modules. Test everything via the recursive peano test.

**Tech Stack:** Python, llvmlite, pytest

---

### Task 1: Add `equal_index` and `subtract_index` to builtin

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Regenerate: `dgen/dialects/builtin.py`
- Modify: `dgen/codegen.py:198-211` (add LLVM emission)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

Add to `test/test_peano.py`:

```python
def test_equal_and_subtract_index():
    """equal_index and subtract_index ops round-trip and lower to LLVM."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %eq : Index = equal_index(%n, 0)
        |     %sub : Index = subtract_index(%n, 1)
        |     %result : Index = add_index(%eq, %sub)
        |     %_ : Nil = return(%result)
    """)
    from dgen.asm.parser import parse_module
    module = parse_module(ir)
    # round-trip
    asm_lines = list(module.asm)
    assert "equal_index" in "\n".join(asm_lines)
    assert "subtract_index" in "\n".join(asm_lines)
```

Note: the `0` and `1` operands are integer literals that become ConstantOps via the parser (same as existing `add_index` tests).

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_peano.py::test_equal_and_subtract_index -xvs`
Expected: FAIL with "Unknown op: equal_index"

**Step 3: Add ops to builtin.dgen**

Add after `op add_index(lhs: Index, rhs: Index) -> Index` (line 35):

```
op equal_index(lhs: Index, rhs: Index) -> Index
op subtract_index(lhs: Index, rhs: Index) -> Index
```

**Step 4: Regenerate builtin.py**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

**Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_peano.py::test_equal_and_subtract_index -xvs`
Expected: PASS

**Step 6: Add LLVM codegen for equal_index and subtract_index**

In `dgen/codegen.py`, inside `_emit_func`, after the `elif isinstance(op, llvm.IcmpOp)` block (~line 211), add handling. But these are builtin ops, not llvm ops. Add them after the existing isinstance checks, before the `elif isinstance(op, builtin.ReturnOp)` block:

```python
elif isinstance(op, builtin.EqualIndexOp):
    lines.append(
        f"  %{name}_raw = icmp eq i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
    )
    lines.append(f"  %{name} = zext i1 %{name}_raw to i64")
elif isinstance(op, builtin.SubtractIndexOp):
    lines.append(
        f"  %{name} = sub i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
    )
```

**Step 7: Write a JIT execution test**

Add to `test/test_peano.py`:

```python
def test_equal_subtract_jit():
    """equal_index and subtract_index execute correctly via JIT."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %sub : Index = subtract_index(%n, 1)
        |     %_ : Nil = return(%sub)
    """)
    module = parse_module(ir)
    from dgen import codegen
    exe = codegen.compile(module)
    assert exe.run(5) == 4
```

**Step 8: Run all tests**

Run: `python -m pytest test/test_peano.py -xvs`
Expected: all pass

**Step 9: Commit**

```bash
jj commit -m "add equal_index and subtract_index builtin ops with codegen"
```

---

### Task 2: Multi-block op parsing and formatting

**Files:**
- Modify: `dgen/asm/parser.py:234-253` (parse_op_fields block handling)
- Modify: `dgen/asm/formatting.py:142-191` (op_asm block emission)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

Add to `test/test_peano.py`:

```python
def test_if_else_parse_roundtrip():
    """if/else op with two blocks parses and round-trips."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %result : Index = if(%cond) ():
        |         %_ : Nil = return(10)
        |     else ():
        |         %_ : Nil = return(20)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    assert "if(" in asm_text or "if (" in asm_text
    assert "else" in asm_text
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_peano.py::test_if_else_parse_roundtrip -xvs`
Expected: FAIL — "Unknown op: if" (IfOp doesn't exist yet)

**Step 3: Add `if` op to builtin.dgen**

Add after the `return` op:

```
op if(cond: Index) -> Type:
    block then_body
    block else_body
```

**Step 4: Regenerate builtin.py**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

Verify the generated IfOp has `__blocks__ = ("then_body", "else_body")`.

**Step 5: Update parser to handle multiple blocks**

In `dgen/asm/parser.py`, function `parse_op_fields` (~line 234-253). Currently it only handles the first block. Replace the block-parsing section with:

```python
    # Body (indented blocks)
    if cls.__blocks__:
        for block_idx, block_name in enumerate(cls.__blocks__):
            if block_idx > 0:
                # Subsequent blocks: expect keyword at parent indent
                # The keyword is the block name with "_body" stripped
                keyword = block_name.removesuffix("_body")
                parser.skip_whitespace_and_newlines()
                # Read the keyword
                saved_pos = parser.pos
                try:
                    word = parser.parse_identifier()
                except RuntimeError:
                    parser.pos = saved_pos
                    break
                if word != keyword:
                    parser.pos = saved_pos
                    break
            parser.skip_whitespace()
            # Parse optional block args: (%name: type, ...)
            args = []
            if parser.peek() == "(":
                parser.expect("(")
                parser.skip_whitespace()
                if parser.peek() != ")":
                    args.append(parser._parse_param())
                    parser.skip_whitespace()
                    while parser.peek() == ",":
                        parser.expect(",")
                        parser.skip_whitespace()
                        args.append(parser._parse_param())
                        parser.skip_whitespace()
                parser.expect(")")
            parser.skip_whitespace()
            if block_idx == 0:
                parser.expect(":")
            else:
                parser.expect(":")
            ops = parser.parse_indented_block()
            kwargs[block_name] = Block(ops=ops, args=args)
```

**Step 6: Update formatter to emit multiple blocks**

In `dgen/asm/formatting.py`, function `op_asm` (~line 142-191). Replace the block-emission logic. The first block's args go on the op line (existing behavior). Subsequent blocks get a keyword line:

```python
    blocks = list(op.blocks)
    if blocks:
        # First block args on the op line
        _, first_block = blocks[0]
        block_args = ", ".join(
            f"%{tracker.track_name(a)}: {type_asm(a.type, tracker) if isinstance(a.type, Type) else format_expr(a.type, tracker)}"
            for a in first_block.args
        )
        parts.append(f" ({block_args}):")

    line = "".join(parts)
    yield line

    for block_idx, (block_name, block) in enumerate(blocks):
        if block_idx > 0:
            # Emit keyword line for subsequent blocks
            keyword = block_name.removesuffix("_body")
            block_args = ", ".join(
                f"%{tracker.track_name(a)}: {type_asm(a.type, tracker) if isinstance(a.type, Type) else format_expr(a.type, tracker)}"
                for a in block.args
            )
            yield f"{keyword} ({block_args}):"
        for child_op in block.ops:
            if _is_sugar_op(child_op):
                continue
            yield from indent(op_asm(child_op, tracker))
```

This replaces the existing block-emission code at the end of `op_asm`.

**Step 7: Run test to verify it passes**

Run: `python -m pytest test/test_peano.py::test_if_else_parse_roundtrip -xvs`
Expected: PASS

**Step 8: Run all existing tests to check nothing broke**

Run: `python -m pytest . -q`
Expected: all pass

**Step 9: Commit**

```bash
jj commit -m "support multi-block ops: if/else parsing and formatting"
```

---

### Task 3: LLVM codegen for `if`/`else`

**Files:**
- Modify: `dgen/codegen.py` (add IfOp emission)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

```python
def test_if_else_jit():
    """if/else executes correctly via JIT — returns 1 if n==0, else n-1."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %result : Index = if(%cond) ():
        |         %_ : Nil = return(1)
        |     else ():
        |         %val : Index = subtract_index(%n, 1)
        |         %_ : Nil = return(%val)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    from dgen import codegen
    exe = codegen.compile(module)
    assert exe.run(0) == 1
    assert exe.run(5) == 4
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_peano.py::test_if_else_jit -xvs`
Expected: FAIL — IfOp not handled in codegen

**Step 3: Add IfOp codegen**

In `dgen/codegen.py`, in `_emit_func`, add handling for IfOp. The if op needs to:
1. Emit `br i1` to branch on the condition (truncated to i1)
2. Emit then block with a label
3. Emit else block with a label
4. Emit a merge label with a phi node

```python
elif isinstance(op, builtin.IfOp):
    then_label = f"then_{name}"
    else_label = f"else_{name}"
    merge_label = f"merge_{name}"
    # Truncate i64 condition to i1
    lines.append(f"  %{name}_cond = trunc i64 {bare_ref(op.cond)} to i1")
    lines.append(f"  br i1 %{name}_cond, label %{then_label}, label %{else_label}")

    # Then block
    lines.append(f"{then_label}:")
    then_result = None
    for child in op.then_body.ops:
        child_name = tracker.track_name(child)
        if isinstance(child, builtin.ReturnOp):
            then_result = child.value
            lines.append(f"  br label %{merge_label}")
        elif isinstance(child, ConstantOp):
            # Register constant but don't emit
            pass
        else:
            # Emit child op (reuse existing dispatch)
            _emit_op(child, child_name, lines, constants, types, tracker, typed_ref, bare_ref, host_buffers)

    # Else block
    lines.append(f"{else_label}:")
    else_result = None
    for child in op.else_body.ops:
        child_name = tracker.track_name(child)
        if isinstance(child, builtin.ReturnOp):
            else_result = child.value
            lines.append(f"  br label %{merge_label}")
        elif isinstance(child, ConstantOp):
            pass
        else:
            _emit_op(child, child_name, lines, constants, types, tracker, typed_ref, bare_ref, host_buffers)

    # Merge block with phi
    lines.append(f"{merge_label}:")
    ty = types.get(id(op), "i64")
    lines.append(
        f"  %{name} = phi {ty} [{bare_ref(then_result)}, %{then_label}], [{bare_ref(else_result)}, %{else_label}]"
    )
```

Note: This is a sketch. The actual implementation will need to handle nested ops within the if/else blocks by recursively processing them through the same dispatch logic used for top-level ops. This likely means extracting the per-op dispatch into a helper function `_emit_op` and calling it for children of the if/else blocks. The constants/types dicts need to include child ops too.

The key insight is that child ops in the if/else blocks need to be pre-scanned for constants just like top-level ops, and the IfOp's ReturnOps yield values that feed into the phi node.

**Step 4: Pre-scan child ops for constants**

The existing pre-scan loop that populates `constants` and `types` dicts only iterates `f.body.ops`. Extend it to also iterate ops inside blocks:

In `_emit_func`, after the existing pre-scan loop, add:

```python
    # Also pre-scan ops inside blocks (e.g., if/else bodies)
    for op in f.body.ops:
        for _, block in op.blocks:
            for child in block.ops:
                vid = id(child)
                if isinstance(child, ConstantOp):
                    # same logic as the main pre-scan
                    mem = child.memory
                    layout = mem.layout
                    if _ctype(layout) is ctypes.c_void_p:
                        host_buffers.append(mem)
                        constants[vid] = f"ptr inttoptr (i64 {mem.address} to ptr)"
                        types[vid] = "ptr"
                    else:
                        lt = _llvm_type(layout)
                        raw = mem.unpack()[0]
                        val_str = format_float(raw) if isinstance(raw, float) else str(raw)
                        constants[vid] = f"{lt} {val_str}"
                        types[vid] = lt
                elif (rt := _result_type_str(child.type)) is not None:
                    types[vid] = rt
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_peano.py::test_if_else_jit -xvs`
Expected: PASS

**Step 6: Run all tests**

Run: `python -m pytest . -q`
Expected: all pass

**Step 7: Commit**

```bash
jj commit -m "codegen: emit if/else as branch + phi"
```

---

### Task 4: `call` op

**Files:**
- Modify: `dgen/dialects/builtin.dgen`
- Regenerate: `dgen/dialects/builtin.py`
- Modify: `dgen/codegen.py` (add call emission)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

```python
def test_call_op_roundtrip():
    """call op parses and round-trips."""
    ir = strip_prefix("""
        | %helper : Nil = function<Index>() (%n: Index):
        |     %_ : Nil = return(%n)
        |
        | %main : Nil = function<Index>() (%x: Index):
        |     %result : Index = call<%helper>(%x)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    assert "call<%helper>" in asm_text
```

**Step 2: Run to verify it fails**

Run: `python -m pytest test/test_peano.py::test_call_op_roundtrip -xvs`
Expected: FAIL — "Unknown op: call"

**Step 3: Add `call` to builtin.dgen**

```
op call<callee: Type>(args: list) -> Type
```

**Step 4: Regenerate builtin.py**

Run: `python -m dgen.gen dgen/dialects/builtin.dgen > dgen/dialects/builtin.py`

**Step 5: Run test again**

Run: `python -m pytest test/test_peano.py::test_call_op_roundtrip -xvs`
Expected: PASS (parse + format work generically)

**Step 6: Add codegen for builtin CallOp**

In `dgen/codegen.py`, add handling for `builtin.CallOp` (the high-level call). This emits an LLVM `call` instruction to the function named by the callee:

```python
elif isinstance(op, builtin.CallOp):
    callee_func = op.callee
    assert isinstance(callee_func, builtin.FunctionOp)
    callee_name = tracker.track_name(callee_func)
    a = ", ".join(typed_ref(arg) for arg in op.args)
    if isinstance(op.type, builtin.Nil):
        lines.append(f"  call void @{callee_name}({a})")
    else:
        ret_ty = types[id(op)]
        lines.append(f"  %{name} = call {ret_ty} @{callee_name}({a})")
```

**Step 7: Write a JIT test for call**

```python
def test_call_jit():
    """call op executes a helper function via JIT."""
    ir = strip_prefix("""
        | %helper : Nil = function<Index>() (%n: Index):
        |     %result : Index = add_index(%n, 1)
        |     %_ : Nil = return(%result)
        |
        | %main : Nil = function<Index>() (%x: Index):
        |     %result : Index = call<%helper>(%x)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    from dgen import codegen
    exe = codegen.compile(module)
    assert exe.run(5) == 6
```

**Step 8: Run test**

Run: `python -m pytest test/test_peano.py::test_call_jit -xvs`
Expected: PASS (may need codegen adjustments for multi-function emission)

Note: `emit_llvm_ir` already iterates `module.functions`, so both functions should be emitted. The `compile` function uses `module.functions[0]` as main — that's `%helper` here. We may need to adjust the test so `%main` is first, or update `compile` to accept a main function name. Adjust as needed.

**Step 9: Run all tests**

Run: `python -m pytest . -q`

**Step 10: Commit**

```bash
jj commit -m "add builtin call op with codegen"
```

---

### Task 5: Trait registration on Dialect

**Files:**
- Modify: `dgen/dialect.py:17-46`
- Modify: `dgen/type.py:162-192` (TypeType subclass support)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

```python
def test_natural_trait():
    """Natural trait is a TypeType subclass, registered in peano dialect."""
    from dgen.type import TypeType
    assert issubclass(Natural, TypeType)
    nat = Natural(concrete=Zero())
    assert type_constant(nat) == Zero()
    # Registered in dialect
    assert "Natural" in peano.types
```

**Step 2: Run to verify it fails**

Run: `python -m pytest test/test_peano.py::test_natural_trait -xvs`
Expected: FAIL — `Natural` is not defined

**Step 3: Add `trait()` method to Dialect**

In `dgen/dialect.py`, add after the `type()` method:

```python
    def trait(self, name: str) -> Callable[[builtins.type[_T]], builtins.type[_T]]:
        def decorator(cls: builtins.type[_T]) -> builtins.type[_T]:
            cls.asm_name = name
            cls.dialect = self
            self.types[name] = cls  # traits are in the type namespace
            return cls
        return decorator
```

**Step 4: Define Natural trait in test_peano.py**

Add after the `peano = Dialect("peano")` line:

```python
@peano.trait("Natural")
@dataclass(frozen=True)
class Natural(TypeType):
    """A TypeType constrained to Natural numbers (Zero or Successor)."""
    pass
```

And update Zero and Successor to declare `__traits__`:

```python
@peano.type("Zero")
@dataclass(frozen=True)
class Zero(Type):
    __traits__ = (Natural,)
    __layout__ = layout.Void()

@peano.type("Successor")
@dataclass(frozen=True)
class Successor(Type):
    pred: Value[TypeType]
    __params__: ClassVar[Fields] = (("pred", Type),)
    __traits__ = (Natural,)
    __layout__ = layout.Void()
```

**Step 5: Run test**

Run: `python -m pytest test/test_peano.py::test_natural_trait -xvs`
Expected: PASS

**Step 6: Test that Natural works in ASM parsing**

```python
def test_natural_in_asm():
    """peano.Natural parses as a type annotation."""
    ir = strip_prefix("""
        | import peano
        |
        | %main : Nil = function<Index>() ():
        |     %z : peano.Natural = peano.zero()
        |     %_ : Nil = return(%z)
    """)
    module = parse_module(ir)
    func = module.functions[0]
    zero_op = func.body.ops[0]
    assert isinstance(zero_op.type, Natural)
```

Note: This requires that when the parser sees `peano.Natural` as a type, it constructs `Natural()`. Since Natural extends TypeType which extends Type, and the parser resolves type names from the dialect registry, this should work. But Natural() has no `concrete` field default — we may need `concrete: Value[TypeType] = TypeType(concrete=...)` or provide a default. TypeType's `concrete` field doesn't have a default, so `Natural()` will fail. We need to handle this — either give Natural a default `concrete` or have the parser treat trait types specially.

The simplest fix: TypeType already requires `concrete`. When parsing `peano.Natural` without `<...>`, the parser calls `Natural()` which will fail because `concrete` is required. Since Natural is `@dataclass(frozen=True)` and inherits from TypeType which is also `@dataclass(frozen=True)`, the fields are inherited.

Solution: Override `__init__` or provide a sentinel default. Or: when used as a type annotation, it means "TypeType with this constraint" — the parser should construct it without concrete. This means Natural needs to allow construction without concrete. We could give it a default: `concrete: Value[TypeType] = Nil()` or similar sentinel.

Actually, looking at how TypeType is used in type annotations: `TypeType<peano.Zero>` means `TypeType(concrete=Zero())`. But `peano.Natural` (no params) should construct `Natural()` — which needs no-arg construction. Since TypeType has `concrete` as a required field and Natural inherits it, we need Natural to override it with a default.

```python
@peano.trait("Natural")
@dataclass(frozen=True)
class Natural(TypeType):
    concrete: Value[TypeType] = Nil()  # unknown until resolved
```

Wait, that doesn't work either — Nil() isn't a valid TypeType value. Let's use a simpler approach: Natural has no fields of its own, and we give concrete a default that's a sentinel:

```python
@peano.trait("Natural")
@dataclass(frozen=True)
class Natural(TypeType):
    pass
```

But this inherits `concrete: Value[TypeType]` with no default from TypeType. We need to check how `@dataclass(frozen=True)` handles inherited fields. Since TypeType has `concrete` with no default, Natural also requires it.

The cleanest solution: When parsing a bare trait name like `peano.Natural`, don't try to construct it — return the class itself. But that breaks the Value system.

Alternative: make Natural's concrete field optional with a sentinel. TypeType's `__layout__` property will fail if concrete isn't a real type, but that's fine — it only gets called when materializing to memory, not when used as an annotation.

Let's provide a sentinel type that means "any satisfying type":

```python
@peano.trait("Natural")
@dataclass(frozen=True)
class Natural(TypeType):
    concrete: Value[TypeType] = builtin.Nil()
```

This allows `Natural()` construction. The `Nil()` default means "unknown/any". We'll address the details during implementation.

**Step 7: Run all tests**

Run: `python -m pytest . -q`

**Step 8: Commit**

```bash
jj commit -m "add trait registration to Dialect, define peano.Natural"
```

---

### Task 6: Module-level staging (multi-function)

**Files:**
- Modify: `dgen/staging.py` (handle multiple functions)
- Modify: `dgen/codegen.py` (compile all functions, not just first)
- Test: `test/test_peano.py`

**Step 1: Write a failing test**

```python
def test_multi_function_call():
    """Two functions: helper adds 1, main calls helper."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%x: Index):
        |     %result : Index = call<%add_one>(%x)
        |     %_ : Nil = return(%result)
        |
        | %add_one : Nil = function<Index>() (%n: Index):
        |     %r : Index = add_index(%n, 1)
        |     %_ : Nil = return(%r)
    """)
    module = parse_module(ir)
    exe = compile_staged(module, infer=lambda m: m, lower=lambda m: m)
    assert exe.run(5) == 6
```

**Step 2: Run to verify it fails**

Run: `python -m pytest test/test_peano.py::test_multi_function_call -xvs`
Expected: FAIL — codegen or staging only handles first function

**Step 3: Update codegen to emit all functions**

`emit_llvm_ir` already iterates `module.functions`, so all functions are emitted. The issue is that `compile()` uses `module.functions[0]` as the main entry point. The main function should be the one the user wants to call — typically the first one, but with `call` ops referencing other functions, order matters.

For now: the convention is that the first function is `main`. The codegen already emits all functions. The callee lookup in codegen should work if the function is emitted before it's called (or if LLVM handles forward references).

The staging system's `_resolve_all_comptime` operates on `module.functions[0]`. For multi-function staging, we need to process all functions that have unresolved boundaries.

Update `_resolve_all_comptime` and `compile_staged` to iterate all functions. The approach: process functions in reverse order (callees before callers), resolving comptime boundaries in each.

This is the most complex change. The details will emerge during implementation — start with the test and iterate.

**Step 4: Run tests and iterate**

Run: `python -m pytest test/test_peano.py::test_multi_function_call -xvs`
Fix issues as they emerge.

**Step 5: Run all tests**

Run: `python -m pytest . -q`

**Step 6: Commit**

```bash
jj commit -m "support multi-function modules in staging and codegen"
```

---

### Task 7: Recursive Peano test

**Files:**
- Modify: `test/test_peano.py` (add the full recursive test)
- Possibly modify: `dgen/staging.py` (handle recursive callbacks)

**Step 1: Write the full recursive peano test**

```python
def test_recursive_peano():
    """Recursive function builds Successor chain from runtime Index, then resolves to Index.

    natural(3) builds Successor<Successor<Successor<Zero>>> via recursion,
    then value resolves it to 3.
    """
    ir = strip_prefix("""
        | import peano
        |
        | %natural : Nil = function<peano.Natural>() (%n: Index):
        |     %base_case : Index = equal_index(%n, 0)
        |     %value : peano.Natural = if(%base_case) ():
        |         %_ : Nil = return(peano.Zero)
        |     else ():
        |         %n_minus_one : Index = subtract_index(%n, 1)
        |         %predecessor : peano.Natural = call<%natural>(%n_minus_one)
        |         %_ : Nil = return(%predecessor)
        |     %successor : peano.Natural = peano.successor<%value>()
        |     %_ : Nil = return(%successor)
        |
        | %main : Nil = function<Index>() (%x: Index):
        |     %n : peano.Natural = call<%natural>(%x)
        |     %result : Index = peano.value<%n>()
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)

    print("\n=== Compile ===")
    exe = compile_staged(module, infer=lambda m: m, lower=lower_peano)

    print("\n=== Run natural(3) ===")
    result = exe.run(3)
    print(f"result = {result}")
    assert result == 3

    print("\n=== Run natural(0) ===")
    result = exe.run(0)
    print(f"result = {result}")
    assert result == 0
```

Note: natural(n) builds n+1 successors (it always wraps in one more Successor), so natural(3) would give Successor^4(Zero) = 4, not 3. The test expectations may need adjustment based on the actual semantics. The IR as written wraps the if result in one more Successor, so:
- natural(0): if returns Zero, then Successor(Zero) = 1
- natural(1): else recurses with 0 -> Successor(Zero), then Successor(Successor(Zero)) = 2
- natural(n): n+1

If we want natural(n) = n, adjust the IR so the base case returns Zero without the extra Successor wrapper. But this is a detail for implementation — the test can be adjusted to match.

**Step 2: Run test — expect failures initially**

Run: `python -m pytest test/test_peano.py::test_recursive_peano -xvs`
Expected: Various failures. Debug and fix each one:
- Staging needs to handle recursive function references
- The callback system needs to handle `call` ops triggering sub-JIT
- The lowering pass needs to handle ops inside if/else blocks

**Step 3: Iterate on staging/codegen fixes**

This is the integration task. The specific fixes will depend on what breaks. Likely areas:
- `_trace_dependencies` needs to handle cross-function references
- `_compile_with_callbacks` needs to handle `call` ops that reference functions with their own staging
- `lower_peano` needs to descend into blocks (if/else bodies) and handle `call` ops

**Step 4: Run all tests**

Run: `python -m pytest . -q`

**Step 5: Commit**

```bash
jj commit -m "recursive peano test: runtime-dependent type building via staging"
```

---

### Task 8: Format, lint, type-check

**Files:**
- All modified files

**Step 1: Format**

Run: `ruff format`

**Step 2: Lint**

Run: `ruff check --fix`

**Step 3: Type check**

Run: `ty check`

**Step 4: Run all tests one final time**

Run: `python -m pytest . -q`

**Step 5: Commit any fixes**

```bash
jj commit -m "format, lint, type-check fixes"
```
