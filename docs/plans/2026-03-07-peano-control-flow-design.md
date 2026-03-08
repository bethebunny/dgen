# Control Flow, Call, and Traits

## Goal

Build real infrastructure for control flow (`if`/`else`), function calls (`call`), arithmetic (`equal_index`, `subtract_index`), and marker traits — demonstrated via a recursive Peano arithmetic test that builds dependent types at runtime.

## Target IR

```
import peano

%natural = function<peano.Natural>() (%n: Index):
    %base_case = equal_index(%n, 0)
    %value: peano.Natural = if(%base_case) ():
        %_: Nil = return(peano.Zero)
    else ():
        %n_minus_one = subtract_index(%n, 1)
        %predecessor: peano.Natural = call<%natural>(%n_minus_one)
        %_: Nil = return(%predecessor)
    %successor: peano.Natural = peano.successor<%value>()
    %_: Nil = return(%successor)

%main : Nil = function<Index>() (%x: Index):
    %n: peano.Natural = call<%natural>(%x)
    %result: Index = peano.value<%n>()
    %_ : Nil = return(%result)
```

When `main(3)` runs, the staging system JIT-compiles `natural` recursively, building `Successor<Successor<Successor<Zero>>>`, then `value` resolves it to `3`.

## 1. Multi-block ops (`if`/`else`)

Ops can declare multiple named blocks via `__blocks__`. The block names determine ASM keywords.

```python
@builtin.op("if")
@dataclass(eq=False, kw_only=True)
class IfOp(Op):
    cond: Value
    then_body: Block
    else_body: Block
    type: Type
    __operands__ = (("cond", Index),)
    __blocks__ = ("then_body", "else_body")
```

**ASM syntax**: Blocks after the first are preceded by the block name (with `_body` stripped):

```
%value: T = if(%cond) ():
    %_: Nil = return(%x)
else ():
    %_: Nil = return(%y)
```

**Parser changes** (`dgen/asm/parser.py`): `parse_op_fields` iterates all blocks in `__blocks__`. After parsing the first block's indented region, it checks if the next token at the parent indent level matches the next block's keyword (block name with `_body` stripped). If so, parses block args and the indented block.

**Formatter changes** (`dgen/asm/formatting.py`): `op_asm` emits the first block inline with the op. Subsequent blocks get their own keyword line at the same indent level as the op.

**Semantics**: `if` is an expression — it returns a value. Each branch ends with `return(value)` to yield from the block. The type of the `if` op is the result type.

## 2. `call` op

Builtin op for high-level function calls. The callee is a `__params__` field (compile-time: which function you call determines the return type). Args are `__operands__` (runtime values).

```python
@builtin.op("call")
@dataclass(eq=False, kw_only=True)
class CallOp(Op):
    callee: Value       # SSA ref to FunctionOp
    args: list[Value]
    type: Type
    __params__ = (("callee", Type),)
    __operands__ = (("args", Type),)
```

ASM: `%result: T = call<%func>(%arg1, %arg2)`

## 3. `equal_index` and `subtract_index`

Added to `builtin.dgen`, following `add_index`'s pattern:

```
op equal_index(lhs: Index, rhs: Index) -> Index
op subtract_index(lhs: Index, rhs: Index) -> Index
```

Lower to `icmp eq` + `zext` and `sub` in LLVM codegen.

## 4. Traits (marker only)

Traits are TypeType subclasses registered via `dialect.trait()`. Types declare trait satisfaction via `__traits__`.

```python
@peano.trait("Natural")
class Natural(TypeType):
    pass

@peano.type("Zero")
class Zero(Type):
    __traits__ = (Natural,)
    __layout__ = layout.Void()
```

Natural extends TypeType, so `type_constant()` works — Natural IS a TypeType. When a concrete type is known, the type becomes `Natural(concrete=Zero())`.

No enforcement: the type system doesn't yet check that values annotated as Natural actually hold a satisfying type.

**Dialect registration**: `Dialect` gets a `trait()` method that registers the trait in the type namespace (same as regular types for ASM parsing purposes).

## 5. Module-level staging

`compile_staged` is extended to handle multiple functions in one module. Helper functions (like `natural`) are compiled alongside `main`. The `call` op references them by SSA value.

The recursive case: `natural` has runtime-dependent `__params__` (successor's pred depends on the recursive call result). The staging system builds a callback for `natural`. At runtime, each call triggers JIT compilation with the concrete type value, building the Successor chain one layer at a time via recursion.

## 6. Codegen

| Op | LLVM emission |
|----|---------------|
| `if` | `br i1 %cond, label %then, label %else` + phi node to merge results |
| `call` | `call @func_name(args...)` |
| `equal_index` | `icmp eq i64 %lhs, %rhs` + `zext i1 to i64` |
| `subtract_index` | `sub i64 %lhs, %rhs` |

## 7. Changes summary

| Component | Change |
|-----------|--------|
| `builtin.dgen` | Add `equal_index`, `subtract_index`, `if`, `call` ops |
| `dgen/asm/parser.py` | Support multiple blocks per op (keyword-separated) |
| `dgen/asm/formatting.py` | Emit multiple blocks with keywords |
| `dgen/type.py` | Trait support: TypeType subclasses |
| `dgen/dialect.py` | `dialect.trait()` registration method |
| `dgen/staging.py` | Module-level staging (multiple functions) |
| `dgen/codegen.py` | Emit if/call/equal/subtract as LLVM IR |
| `test/test_peano.py` | Natural trait, updated lower_peano, recursive test |
