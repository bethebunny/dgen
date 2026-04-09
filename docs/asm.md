# IR Text Format

**Generic DGEN principles**: simple, predictable, composable

DGEN IR files use the `.ir` file extension.

## Simplifications

The IR text format is a simplified version of MLIR assembly. It can be parsed unambiguously without loading any dialect definitions.

Design priorities: explicit clarity and lack of ambiguity over brevity.

- No custom assembly formats for ops or types
- No special symbol or label syntax
- Not compatible with LLVM IR syntax
- Every op has exactly one result (zero results use `Nil`, multiple use a typed tuple/list)
- No variadic or optional parameters/operands

```
import function
import index
import toy

%main : function.Function<[], Nil> = function.function<Nil>() body():
    %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : Nil = toy.print(%0)
```

## JSON as Generalized Constants

Since DGEN knows the memory layout of all types, any value may be specified as a JSON literal. The type annotation drives deserialization:

```
%0 : index.Index = 42
%1 : number.Float64 = 3.14
%2 : String = "hello"
%3 : ndbuffer.NDBuffer<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

A JSON literal is syntactically equivalent to a `builtin.constant` op. The special literal `()` represents a `Nil` value.

## SSA Values in Parameters and Types

Type expressions and parameter expressions may be SSA references. This enables dependent types through DGEN's staging system:

```
%0 : index.Index = 3
%1 : %some_type = llvm.load(%ptr)
```

## Statement Grammar

All IR statements are either:

1. An import: `import dialect_name`
2. An assignment: `%name : type_expr = expression`

## Expression Grammar

Expressions are one of:

- SSA reference: `%name`
- JSON literal: `42`, `3.14`, `"str"`, `[1, 2]`, `{"k": 1}`, `()`
- Op expression: `dialect.op<params>(operands)`
- Type expression: `dialect.Type<params>`
- Constant expression: `Type(json_value)` -- a typed constant

Rules:

- If an op/type has no parameters, angle brackets are omitted
- Parentheses around operands must be present even for zero operands
- Builtin types/ops omit the `builtin.` prefix

## Block Syntax

Ops with blocks use indented block bodies:

```
%name : Type = op<params>(operands) block_name<%param: PType>(%arg: AType) captures(%cap):
    %inner : Type = ...
```

- `<%param: PType>` -- block parameters (compile-time)
- `(%arg: AType)` -- block arguments (runtime)
- `captures(%cap)` -- captured outer-scope values

## ASM Round-Trip

IR can be printed to text and parsed back. The `dgen/asm/formatting.py` module handles emission; `dgen/asm/parser.py` handles parsing. Round-trip correctness is heavily tested via `dgen/ir/equivalence.py` (Merkle fingerprint comparison).

## Key Files

| File | Role |
|------|------|
| `dgen/asm/formatting.py` | `SlotTracker`, op/block formatting |
| `dgen/asm/parser.py` | `parse()`, `Scope`, reader functions |
