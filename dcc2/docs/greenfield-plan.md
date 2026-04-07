# Greenfield dcc2: C Frontend for dgen

## Context

The existing `dcc/` C frontend has accumulated ~30% hacks/workarounds: struct field offsets hardcoded to 0, no lvalue model, no integer promotions, a scope system that conflates storage with value. An incremental migration plan (c-frontend-redesign.md) exists but isn't working in practice. This plan defines a **clean-room greenfield** implementation in `dcc2/` that builds from scratch, brick by brick, reaching the same endpoint. `dcc/` code is read-only design input, not migration source.

## Directory Structure

```
dcc2/
├── __init__.py                    # Package init, registers dialect path
├── __main__.py                    # python -m dcc2
├── cli.py                         # Compiler pipeline + run_c()
├── dialects/
│   ├── __init__.py                # Type constructors (c_int, c_ptr, etc.)
│   ├── c.dgen                     # Lvalue-based C dialect (fresh)
│   └── c.pyi                      # Generated
├── parser/
│   ├── __init__.py
│   ├── c_parser.py                # pycparser wrapper
│   ├── c_literals.py              # Integer/char literal parsing
│   ├── type_resolver.py           # C types -> dgen types
│   └── lowering.py                # AST -> IR (thin, generator-based)
├── passes/
│   ├── __init__.py
│   ├── c_struct_layout.py         # Compute struct field offsets
│   ├── c_implicit_conversions.py  # Insert promotion/decay/cast ops
│   ├── c_lvalue_to_memory.py      # Lvalues -> memory ops (alloca/load/store)
│   └── c_to_llvm.py               # C arithmetic -> LLVM ops
└── test/
    ├── __init__.py
    ├── test_c_frontend.py
    └── testdata/
```

## Compiler Pipeline (target state)

```python
c_compiler = Compiler(
    passes=[
        CStructLayout(),           # Brick 8: compute field offsets
        CImplicitConversions(),    # Brick 9: insert type promotions/conversions
        CLvalueToMemory(),         # Brick 5: lvalues -> alloca/load/store
        CToLLVM(),                 # Brick 10: C arithmetic -> LLVM ops
        ControlFlowToGoto(),       # Shared
        MemoryToLLVM(),            # Shared
        BuiltinToLLVM(),           # Shared
        AlgebraToLLVM(),           # Shared
    ],
    exit=LLVMCodegen(),
)
```

---

## Brick-by-Brick Implementation Plan

### Brick 1: C Dialect Definition + Type Constructors

**Builds:** The lvalue-based `c.dgen` and Python type constructors. Every subsequent brick references these.

**Files:**
- `dcc2/__init__.py` -- package init, `Dialect.paths.append`
- `dcc2/dialects/__init__.py` -- `c_int()`, `c_float32()`, `c_double()`, `c_void()`, `c_ptr()`
- `dcc2/dialects/c.dgen` -- full lvalue-based dialect

**c.dgen ops:**
- Lvalue-producing: `lvalue_var<name>(source)`, `lvalue_deref(ptr)`, `lvalue_subscript(base, index)`, `lvalue_member<field>(base)`, `lvalue_arrow<field>(ptr)`
- Lvalue-consuming: `lvalue_to_rvalue(lvalue)`, `address_of(lvalue)`, `assign(lvalue, rvalue)`, `compound_assign<operator>(lvalue, rvalue)`, `pre_increment(lvalue)`, `post_increment(lvalue)`, `pre_decrement(lvalue)`, `post_decrement(lvalue)`
- Conversions: `integer_promote(input)`, `arithmetic_convert(input)`, `array_decay(array)`, `function_decay(func)`, `null_to_pointer(zero)`, `scalar_to_bool(val)`
- Control: `c_return(value) -> Nil`, `c_sizeof<target_type>()`
- Arithmetic: `modulo(lhs, rhs)`, `shift_left(lhs, rhs)`, `shift_right(lhs, rhs)`, `logical_not(operand)`, `comma(lhs, rhs)`

**c.dgen types:**
- `Struct<tag, fields>` with `StructField<name, field_type, offset>`, `Union<tag, fields>`, `Enum<tag, underlying>`
- `CFunctionType<arguments, result_type, is_variadic, n_fixed_params>`

**Tests:** Dialect loads, all op/type classes exist and can be instantiated.

**Foundation:** All IR building blocks available for subsequent bricks.

---

### Brick 2: CLI, Compiler Pipeline, Test Harness

**Builds:** The compiler pipeline shape, `run_c()` helper, CLI entry point. System compiles and runs manually-constructed IR.

**Files:**
- `dcc2/cli.py` -- `c_compiler` definition, `run()` function
- `dcc2/__main__.py` -- `from dcc2.cli import main; main()`
- `dcc2/test/__init__.py`
- `dcc2/test/test_c_frontend.py` -- `run_c()` helper + smoke test

**Tests:**
```python
def test_pipeline_smoke() -> None:
    # Manually construct: int f() { return 42; }
    # Compile through pipeline, assert result == 42
```

**Foundation:** Pipeline exists. Once lowering produces a FunctionOp, it compiles and runs.

---

### Brick 3: pycparser Wrapper + Type Resolver

**Builds:** C parsing and type resolution. Clean rewrite of TypeResolver.

**Files:**
- `dcc2/parser/__init__.py`
- `dcc2/parser/c_parser.py` -- `parse_c_string()`, `parse_c_file()`
- `dcc2/parser/c_literals.py` -- integer/char literal parsing
- `dcc2/parser/type_resolver.py` -- clean TypeResolver

**TypeResolver improvements over dcc/:**
1. `Struct` with `StructField` instances (offset=0 initially, filled by layout pass)
2. `float` -> `Float32()`, `double` -> `Float64()` (distinct)
3. `CFunctionType` with `is_variadic`/`n_fixed_params` tracking
4. Parameter type adjustment: array params -> pointers, function params -> pointer-to-function
5. Struct field metadata in the type itself (via StructField)

**Tests:** Unit tests for type resolution (int, float, double, struct, variadic functions, pointer types).

**Foundation:** Any pycparser type node translates to the correct dgen type.

---

### Brick 4: Lowering -- Constants, Parameters, Arithmetic

**Builds:** Core lowering infrastructure: Scope class, Parser class with generator pattern, simplest C functions.

**Files:**
- `dcc2/parser/lowering.py` -- `lower()`, `Parser` class, `Scope` class

**Handles:** `c_ast.Constant` (int literals), `c_ast.ID` (param refs), `c_ast.Return`, `c_ast.BinaryOp` (+, -, *, /), `c_ast.UnaryOp` (negate). Two-pass extern registration.

**Design:** Follows toy/parser/lowering.py pattern -- generator methods yielding ops. Key methods: `_function()`, `_compound()`, `_stmt()`, `_expr()`, `_expr_lvalue()`, `_expr_rvalue()`.

For this brick, function body result IS the return expression directly (no c_return op yet -- same as Toy pattern).

**Tests:**
```python
def test_return_constant(self) -> None:
    assert run_c("int f() { return 42; }") == 42

def test_return_parameter(self) -> None:
    assert run_c("int f(int x) { return x; }", 7) == 7

def test_arithmetic(self) -> None:
    assert run_c("int f(int a, int b) { return a + b; }", 3, 4) == 7
    assert run_c("int f(int a, int b) { return a * b; }", 6, 7) == 42

def test_nested_arithmetic(self) -> None:
    assert run_c("int f(int a, int b) { return (a + b) * (a - b); }", 7, 3) == 40
```

**Foundation:** Lowering infrastructure exists. Adding C constructs means adding handlers.

---

### Brick 5: Local Variables (Lvalue Model + Lvalue Elimination Pass)

**Builds:** The core lvalue model -- the most architecturally significant brick. Introduces `lvalue_var`, `lvalue_to_rvalue`, `assign`, and the CLvalueToMemory pass.

**Files:**
- `dcc2/passes/c_lvalue_to_memory.py` -- lvalue elimination pass
- Updates to `dcc2/parser/lowering.py` and `dcc2/cli.py`

**Lowering:**
- `c_ast.Decl` with init -> `assign(lvalue_var("x"), init_expr)`
- `c_ast.ID` in rvalue context -> `lvalue_to_rvalue(lvalue_var("x", source=binding))`
- `c_ast.Assignment` -> `assign(lvalue_var("x"), rvalue)`

**CLvalueToMemory pass:**
- Scans function body for `lvalue_var` ops, creates one `memory.StackAllocateOp` per variable
- Per-variable mem-token dict for precise ordering
- `@lowering_for(LvalueToRvalueOp)` when lvalue is `LvalueVarOp`: emit `memory.LoadOp`
- `@lowering_for(AssignOp)` when lvalue is `LvalueVarOp`: emit `memory.StoreOp`, update mem token
- For inner blocks referencing allocas: add as captures

**Tests:**
```python
def test_local_variable(self) -> None:
    assert run_c("int f(int x) { int y = x + 1; return y; }", 5) == 6

def test_local_mutation(self) -> None:
    assert run_c("int f(int x) { int y = x; y = y + 10; return y; }", 5) == 15

def test_multiple_locals(self) -> None:
    assert run_c("int f(int a, int b) { int s = a + b; int d = a - b; return s * d; }", 7, 3) == 40
```

**Foundation:** Lvalue model proven. All subsequent lvalue ops (deref, subscript, member) slot into same pass.

---

### Brick 6: Control Flow (if/while/for)

**Builds:** Conditionals and loops using `control_flow.IfOp`, `control_flow.WhileOp`. Short-circuit `&&`/`||`.

**Lowering:**
- `c_ast.If` -> `control_flow.IfOp` with then/else body blocks
- `c_ast.While` -> `control_flow.WhileOp` with condition/body blocks
- `c_ast.For` -> desugar to initializer + WhileOp (update appended to body)
- `&&` -> `if(a) then b else 0`; `||` -> `if(a) then 1 else b`

**Key challenge:** Variable mutation across blocks -- lvalue elimination pass adds allocas as captures to inner blocks.

**Tests:**
```python
def test_if_mutation(self) -> None:
    assert run_c("int f(int x) { int r = 0; if (x) r = 1; return r; }", 1) == 1
    assert run_c("int f(int x) { int r = 0; if (x) r = 1; return r; }", 0) == 0

def test_while_loop(self) -> None:
    assert run_c(
        "int f(int n) { int s = 0; int i = 0; while (i < n) { s = s + i; i = i + 1; } return s; }", 5
    ) == 10

def test_for_loop(self) -> None:
    assert run_c(
        "int f(int n) { int r = 1; int i; for (i = 1; i <= n; i = i + 1) r = r * i; return r; }", 5
    ) == 120

def test_short_circuit_and(self) -> None:
    assert run_c("int f(int x, int y) { return x && y; }", 0, 42) == 0
    assert run_c("int f(int x, int y) { return x && y; }", 1, 42) == 1
```

**Foundation:** Programs can have loops, conditionals, and short-circuit evaluation.

---

### Brick 7: Function Calls and Multi-Function Programs

**Builds:** Cross-function calls via ExternOp + function.CallOp.

**Lowering:**
- First pass registers every function as an ExternOp
- `c_ast.FuncCall` -> `function.CallOp(callee=extern_op, arguments=pack(args))`
- `_add_callee_captures()` ensures ExternOps are captured in calling function bodies
- Void calls in statement position: ChainOp to block result

**Tests:**
```python
def test_function_call(self) -> None:
    assert run_c("int g(int x) { return x + 1; }\nint f(int x) { return g(x); }", 5) == 6

def test_mutual_recursion(self) -> None:
    assert run_c(
        "int is_even(int n); int is_odd(int n) { if (n == 0) return 0; return is_even(n - 1); }\n"
        "int is_even(int n) { if (n == 0) return 1; return is_odd(n - 1); }\n"
        "int f(int x) { return is_even(x); }", 4
    ) == 1
```

**Foundation:** Multi-function programs compile correctly.

---

### Brick 8: Pointers, Arrays, Structs + Struct Layout Pass

**Builds:** Pointer dereference, address-of, array subscript, struct member access, and the struct layout pass.

**Files:**
- `dcc2/passes/c_struct_layout.py` -- compute field offsets (System V AMD64 ABI)
- Updates to lowering + lvalue elimination pass

**Lowering:**
- `c_ast.UnaryOp("*")` -> `lvalue_deref(ptr)` (in lvalue context) or `lvalue_to_rvalue(lvalue_deref(ptr))` (rvalue)
- `c_ast.UnaryOp("&")` -> `address_of(lvalue)` (suppresses lvalue_to_rvalue)
- `c_ast.ArrayRef` -> `lvalue_subscript(base, index)`
- `c_ast.StructRef(".")` -> `lvalue_member<field>(base)`
- `c_ast.StructRef("->")` -> `lvalue_arrow<field>(ptr)`

**CLvalueToMemory additions:**
- `@lowering_for(LvalueDerefOp)`: the pointer IS the address; load from it
- `@lowering_for(AddressOfOp)`: return the alloca/address directly
- `@lowering_for(LvalueSubscriptOp)`: GEP(base, index)
- `@lowering_for(LvalueMemberOp)`: GEP with offset from StructField
- `@lowering_for(LvalueArrowOp)`: load pointer, GEP with offset

**CStructLayout pass:**
- Walk IR for Struct types, compute field offsets per System V AMD64
- Construct new Struct instances with offsets filled, `block.replace_uses_of` to swap

**Tests:**
```python
def test_pointer_deref(self) -> None:
    assert run_c("int f(int *p) { return *p; }", ???)  # needs pointer arg support

def test_address_of(self) -> None:
    assert run_c("int f(void) { int x = 42; int *p = &x; return *p; }") == 42

def test_struct_second_field(self) -> None:
    """THE correctness proof for struct layout -- fails with old GEP-index-0 stub."""
    assert run_c(
        "struct P { int x; int y; };\n"
        "int f(void) { struct P p; p.x = 10; p.y = 20; return p.y; }"
    ) == 20

def test_array_subscript(self) -> None:
    assert run_c(
        "int f(void) { int a[3]; a[0] = 10; a[1] = 20; a[2] = 30; return a[1]; }"
    ) == 20
```

**Foundation:** Aggregate types, pointers, and arrays work with correct memory layout.

---

### Brick 9: Implicit Conversion Pass

**Builds:** Type promotion and conversion insertion per C11 spec.

**Files:**
- `dcc2/passes/c_implicit_conversions.py`

**Handles:**
- Binary arithmetic: integer promotions (char/short -> int), usual arithmetic conversions
- Assignment: convert RHS to LHS type
- Function args: fixed -> param type; variadic -> default argument promotions
- Conditions: insert `scalar_to_bool`
- Array decay: `array_decay` in expression context (except sizeof, address_of operands)

**Tests:**
```python
def test_char_promotion(self) -> None:
    assert run_c("int f(void) { char a = 100; char b = 100; return a + b; }") == 200

def test_comparison(self) -> None:
    assert run_c("int f(void) { return 1 < 2; }") == 1
```

**Foundation:** C type semantics are correct at every expression boundary.

---

### Brick 10: C-to-LLVM Pass + Remaining Ops

**Builds:** Lower C-specific ops to LLVM/algebra ops.

**Files:**
- `dcc2/passes/c_to_llvm.py`

**Handlers:**
- `modulo` -> `llvm.srem`/`llvm.urem` (prerequisite: add these to `llvm.dgen`)
- `shift_left` -> `llvm.shl`, `shift_right` -> `llvm.ashr`/`llvm.lshr`
- `logical_not` -> compare to zero
- `c_sizeof` -> ConstantOp from type's layout size
- `c_return` -> extract value (mem-token deps keep stores reachable)
- `comma(lhs, rhs)` -> rhs (lhs evaluated for side effects via ChainOp)
- `compound_assign`, `pre/post_increment/decrement` -> load + binop + store pattern

**Tests:**
```python
def test_modulo(self) -> None:
    assert run_c("int f(int a, int b) { return a % b; }", 17, 5) == 2

def test_shift(self) -> None:
    assert run_c("int f(int x) { return x << 2; }", 3) == 12
    assert run_c("int f(int x) { return x >> 1; }", 8) == 4

def test_sizeof(self) -> None:
    assert run_c("int f(void) { return sizeof(int); }") == 4

def test_increment(self) -> None:
    assert run_c("int f(void) { int x = 5; x++; return x; }") == 6
    assert run_c("int f(void) { int x = 5; return x++; }") == 5
    assert run_c("int f(void) { int x = 5; return ++x; }") == 6
```

**Foundation:** All C-specific ops lower to LLVM. Full pipeline operational.

---

### Brick 11: Remaining C Constructs

**Builds:** switch, do-while, goto/label, break/continue, ternary, comma, string literals, compound assignment through pointers, cast expressions.

**Tests:** One correctness test per construct:
```python
def test_switch(self) -> None:
    assert run_c("int f(int x) { switch(x) { case 1: return 10; case 2: return 20; default: return 0; } }", 2) == 20

def test_do_while(self) -> None:
    assert run_c("int f(void) { int s = 0; int i = 0; do { s += i; i++; } while (i < 5); return s; }") == 10

def test_ternary(self) -> None:
    assert run_c("int f(int x) { return x > 0 ? x : -x; }", -5) == 5

def test_compound_assign(self) -> None:
    assert run_c("int f(void) { int x = 10; x += 5; x -= 3; x *= 2; return x; }") == 24
```

---

### Brick 12: Scale Test (sqlite3)

**Builds:** sqlite3 parsing and codegen verification. Not correctness -- coverage measurement.

**Tests:** Reuse the sqlite3 test infrastructure from dcc/: download, preprocess, parse, lower, count verified functions. Establish new ratchet thresholds.

---

## Verification

After each brick:
1. `pytest dcc2/test/ -q` -- all tests pass
2. `ruff format dcc2/` -- formatted
3. `ruff check dcc2/` -- no lint errors
4. `ty check` -- type-clean (annotations on everything, no Any/cast/ignore)

After all bricks:
1. `pytest . -q` -- dcc2 tests pass alongside dcc and toy tests
2. `python -m dcc2 dcc/test/testdata/simple.c --dump-ir` -- parses and lowers real C files
3. sqlite3 ratchet at or above dcc/'s level

## Key Design Decisions

1. **No migration** -- dcc2 is independent of dcc. They share only the dgen framework and shared dialects.
2. **Lvalue model from day one** -- no variable ops, no migration path for old ops.
3. **Mem-token threading, not ChainOp** -- precise ordering for memory ops. ChainOp only for void calls.
4. **Generator-based lowering** -- following toy/parser/lowering.py pattern exactly.
5. **Correctness-first testing** -- every feature proven with `run_c()` checking computed values, not IR shape.
6. **Each pass handles exactly one concern** -- no semantic logic in the parser.
