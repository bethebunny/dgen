# Assembly format

**Generic DGEN principles**: simple, predictable, composable

DGEN asm files have the `.dgen.asm` file extension.

### Simplifications

The assembly format is meant to be a much simplified version of MLIR assembly.
It may be unambiguously parsed without loading in any dialects.

The language is meant to be read by humans, but not written by humans.
Any design tradeoffs should favor simple explicit clarity and disambiguity.

- There are no custom assembly formats for ops or types
- There is no special symbol or label syntax
- LLVM IR is not legal DGEN assembly
- Every op has exactly one result, never zero or several
    - A "zero result" op should return Nil
    - An op with multiple results should return a typed Tuple
    - An op with variadic results should return a typed List
- There are no variadic parameters, operands, or return values
- There are no optional parameters or operands

```dgen-asm
import affine

%0: Nil = affine.for<0, 3> (%i):
  %_: Nil = print(%i)
```


With these simplifications, the assembly format is substantially easier to parse
and more uniform.

DGEN then adds 2 new features to allow assembly to better generically express metaprogramming constructs:

### JSON as generalized constants

Since DGEN knows the memory layout of all types, and all types are composed of leaf types which may be described as JSON values, _any_
value may be specified as a constant in IR via a JSON literal.

```dgen-asm
import some_dialect
import toy

%0: List[Int32] = [2, 3, 4]
%1: Tuple[Index, Float64, String] = [2, 3.0, "hello"]
%2: toy.Tensor<[2, 3], f64> = [[2., 3.], [1., 2.], [0., 1.]]
%3: some_dialect.SomeStruct = {"foo": 1, "bar": [2, 3, 4]}
```

### SSA values in parameters and types

`type` and `parameters` may be SSA values. This allows flexible metaprogramming through DGEN's staging system.
The `type` and `parameters` may be generically fully resolved through DGEN's JIT passes, or translated to a runtime JIT.


```dgen-asm
import affine
import llvm
import toy

%0: affine.Shape<3> = [2, 3, 4]
%1: toy.Tensor<%0, f64> = [[2., 3.], [1., 2.], [0., 1.]]
%2: Index = 3
%3: Type = toy.Tensor<%0, f64>
%4: llvm.Ptr = llvm.alloca<%2, %3>
%5: %3 = llvm.load(%4)
```

## Specification

All ASM statements are either

1. An import, importing another dialect
2. A statement of the form `%ssa: type_expression = expression`

All ASM expressions consist of one of

- A reference to an ssa value (`%ssa`)
- A JSON literal
- An `op_expression`
- A `type_expression`
- The special literal `()` representing a `Nil` value

followed by an optional type specifier `: type_expression`.

A JSON literal is syntactically equivalent to a special builtin `ConstantOp` with no name.

Finally

- An `op_expression` has the form `op<parameters>(operands)`
- A `type_expression` has the form `type<parameters>`
- `parameters` and `operands` are comma-separated expressions

If `op` or `type` have no parameters, the angle brackets are omitted. The parentheses 
around operands must be present even if the op accepts no operands.
