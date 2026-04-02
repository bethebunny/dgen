# Dialect File Specification

Dialect files (`.dgen`) define the types, operations, and traits that make up a
dgen dialect. A dialect file is a language-independent specification -- it
describes *what* a dialect contains, not how it is implemented in any particular
compiler.

Dialect files are intentionally not a general-purpose description language.
They are specialized to express exactly three things: types, ops, and traits.
There is no facility for embedding target-language code.

## Lexical structure

The file format is pythonic: line-oriented with indentation-based nesting
and `#` comments.

- Each top-level declaration starts at column 0.
- Body lines are indented (any whitespace prefix).
- `#` begins a comment that extends to the end of the line. Comments may
  appear on their own line or after code.
- Blank lines are ignored.
- Identifiers follow these conventions:
  - **Types and traits**: UpperCamelCase (`Float64`, `AddMagma`)
  - **Ops**: lower\_snake\_case (`stack_allocate`, `dim_size`)
  - **Parameters and operands**: lower\_snake\_case (`element_type`, `lhs`)

## Imports

Imports bring names from other dialect files into scope.

```
from <module> import <Name>, <Name>, ...
import <module>
```

`from` imports bind specific names directly. `import` binds the module
as a namespace -- members are accessed as `module.Name`.

```dgen
from builtin import Index, Nil, Span
from number import Float64
import ndbuffer
```

After `import ndbuffer`, the type `Shape` defined in `ndbuffer.dgen` is
referenced as `ndbuffer.Shape`.

## Type references

A type reference names a type, optionally with angle-bracket parameters:

```
Name
Name<arg, arg, ...>
```

Parameters are themselves type references, so nesting is supported:
`Array<Index, rank>`, `Pointer<Span<Byte>>`.

`Type` is the top of the type hierarchy. All types are subtypes of `Type`.
In parameter positions, `element_type: Type` means the parameter accepts
any type. A trait name in the same position (e.g. `dtype: DType`) constrains
the parameter to types that implement that trait. Both `Type` and trait
names work the same way -- they specify the accepted set of types.

## Traits

A trait is a named property that types and ops can declare they possess.
Traits enable constraint checking: a `requires` clause can demand that an
operand's type implements a given trait.

### Declaration

```dgen
trait <Name>
```

### Examples

```dgen
trait AddMagma
trait TotalOrder
```

## Types

A type declaration introduces a named type into the dialect, optionally
parameterized by compile-time values.

### Declaration

```dgen
type <Name>
type <Name><param: Type, param: Type = default>
type <Name>:
    <body>
type <Name><params>:
    <body>
```

### Parameters

Parameters are compile-time values enclosed in angle brackets. Each
parameter has a name and a type annotation. Parameters may have defaults.

```dgen
type Array<element_type: Type, n: Index>
type NDBuffer<shape: Shape, dtype: Type = Float64>
```

### Body

A type body may contain, in any order:

**Data fields** -- describe the type's structure using typed fields:

```dgen
<name>: <TypeRef>
```

When no explicit `layout` is given, the data fields determine the layout.
A single data field produces that field's layout directly. Multiple data
fields produce a record layout.

```dgen
type Shape<rank: Index>:
    dims: Array<Index, rank>

type Reference<element_type: Type>:
    data: Pointer<Nil>
```

**Layout declaration** -- names the memory layout strategy for this type.
Layouts are typically only needed for builtin types that define fundamental
representations. Most user types should use data fields instead.

```dgen
layout <LayoutName>
```

Layout names refer to implementation-defined layout strategies (e.g.
`Void`, `Int`, `Float64`, `Byte`, `String`, `Pointer`, `Array`, `Span`,
`Record`). A type's layout determines its binary memory representation.

Data fields and `layout` are alternative ways to specify memory
representation. A type body uses one or the other.

**Trait implementation** -- declares that this type implements a trait:

```dgen
has trait <TraitName>
```

**Constraints** -- requirements on the type's parameters:

```dgen
requires <expression>
```

See [Constraints](#constraints) for the full syntax.

### Examples

```dgen
# Zero-parameter type with a named layout
type Nil:
    layout Void

# Parameterized type with data fields
type Tensor<shape: ndbuffer.Shape, dtype: Type = Float64>:
    data: Span<dtype>

# Parameterized type with a layout constructor
type Array<element_type: Type, n: Index>:
    layout Array
```

## Operations

An op declaration introduces a named operation into the dialect.

### Declaration

```dgen
op <name>(<operands>) -> <ReturnType>
op <name><params>(<operands>) -> <ReturnType>
op <name>(<operands>) -> <ReturnType>:
    <body>
```

### Parameters

Compile-time parameters in angle brackets, same syntax as type parameters.
Parameters are values known at compile time -- they are not passed at
runtime.

```dgen
op concat<axis: Index>(lhs: Tensor, rhs: Tensor) -> Tensor
op call<callee: Function>(arguments: Span) -> Nil
```

### Operands

Operands are runtime values passed to the op, listed in parentheses.
Each operand has a name and an optional type annotation.

```dgen
op add(lhs, rhs) -> Nil                       # untyped operands
op fadd(lhs: Float, rhs: Float) -> Float       # typed operands
op store(mem, value, ptr: Reference) -> Nil    # mixed
```

When a type annotation is given, it constrains the operand to that type
(or any parameterization of it). When omitted, the operand accepts any
type.

For variadic operands, use `Span`:

```dgen
op call<callee: String>(args: Span) -> Nil
```

### Return type

The `-> <TypeRef>` clause specifies the result type.

- **Concrete type**: `-> Nil`, `-> Float` -- the result always has this type.
- **Parameterized type name**: `-> Tensor`, `-> NDBuffer` -- the result has
  this type constructor, but the exact parameterization is supplied by the
  caller.
- **Omitted**: no `->` clause -- the result type is entirely determined by the
  caller at construction time.

### Body

An op body may contain, in any order:

**Block declarations** -- named regions of nested computation:

```dgen
block <name>
```

Blocks contain nested ops. An op may declare multiple blocks for
different control flow paths:

```dgen
op for<lower_bound: Index, upper_bound: Index>(initial_arguments: Span) -> Nil:
    block body

op while(initial_arguments: Span) -> Nil:
    block condition
    block body

op if(condition: Index, then_arguments: Span, else_arguments: Span) -> Nil:
    block then_body
    block else_body
```

**Trait implementation** -- declares that this op implements a trait:

```dgen
has trait <TraitName>
```

**Constraints** -- requirements on operands and parameters:

```dgen
requires <expression>
```

See [Constraints](#constraints) for the full syntax.

### Examples

```dgen
# Simple binary op with typed operands
op fadd(lhs: Float, rhs: Float) -> Float

# Parameterized op with a constraint
op dim_size<axis: Index>(input: Tensor) -> Index:
    requires axis < input.shape.rank

# Op with blocks (control flow)
op for<lower_bound: Index, upper_bound: Index>(initial_arguments: Span) -> Nil:
    block body

# Op with a trait
op sqrt(x) -> Float:
    has trait Elementwise
```

## Constraints

Constraints are `requires` clauses that declare invariants on an op's
operands and parameters, or on a type's parameters. They appear in the
body of type or op declarations.

### Trait constraints

Check that an operand's type or a parameter implements a trait:

```dgen
requires <subject> has trait <TraitName>
```

The subject is an operand name or parameter name.

```dgen
type Number<dtype: Type>:
    requires dtype has trait DType
```

### Type constraints

Check that an operand's type matches a named type constructor:

```dgen
requires <subject> has type <TypeRef>
```

The legacy `~=` syntax is equivalent:

```dgen
requires <subject> ~= <TypeRef>
```

```dgen
op tile(x) -> Tensor:
    requires x has type Tensor
```

### Expression constraints

Arbitrary expressions over operand and parameter properties:

```dgen
requires <expression>
```

Expressions may reference operand names, parameter names, and
attribute access on types:

```dgen
op concat<axis: Index>(lhs: Tensor, rhs: Tensor) -> Tensor:
    requires axis < lhs.shape.rank

op dim_size<axis: Index>(input: Tensor) -> Index:
    requires axis < input.shape.rank
```

## Complete grammar

```
file        = (import | trait | type | op | blank | comment)*

comment     = "#" (any text to end of line)

import      = "from" module "import" name ("," name)*
            | "import" module

trait       = "trait" Name

type        = "type" Name [params]
            | "type" Name [params] ":" NEWLINE type_body

type_body   = (INDENT (data_field | layout | has_trait
                      | constraint) NEWLINE)*

op          = "op" name [params] "(" operands ")" ["->" type_ref]
            | "op" name [params] "(" operands ")" ["->" type_ref] ":" NEWLINE op_body

op_body     = (INDENT (block_decl | has_trait | constraint) NEWLINE)*

params      = "<" param ("," param)* ">"
param       = name ":" type_ref ["=" default]

operands    = operand ("," operand)*
operand     = name [":" type_ref]

type_ref    = Name
            | Name "<" type_ref ("," type_ref)* ">"

data_field  = name ":" type_ref
layout      = "layout" Name
has_trait   = "has trait" Name
constraint  = "requires" name "has" "trait" Name
            | "requires" name "has" "type" type_ref
            | "requires" name "~=" type_ref
            | "requires" expression
block_decl  = "block" name
```

Names are identifiers. `Name` (capitalized) is UpperCamelCase by
convention. `name` (lowercase) is lower\_snake\_case by convention.
`module` is a dialect module name. `default` is a literal value.
`expression` is an arbitrary constraint expression.
