# Dialect files

**Generic DGEN principles**: simple, predictable, composable

Dialect files have the `.dgen` extension.

They are _simple_ files. A dialect file must be able to be used to generate
definitions in any language a compiler is authored in.

They may loosely be compared to TableGen `.td` files, but they depart substantially
in terms of syntax, scope, and expressiveness.

### Simplifications

- DGEN dialect files _are not_ a general description language! They are specialized
  to generate DGEN dialects only.
- DGEN dialect files only specify types, ops, and traits
- _No_ language specific codegen passthroughs, a la `extraClassDefinition`!

### Conventions

- ops are lower snake case
- types and traits are upper camel case
- builtin types and ops are referenced without a prefix
- other types and ops are imported by namespace, not directly

### Types have known memory layouts

### Example

```dgen
import affine

trait DType:
  static signed: Boolean
  static bitwidth: Index
  
trait Integral
trait FloatingPoint

type Float64:
  has trait DType
  has trait FloatingPoint
  
  static signed: Boolean = True
  static bitwidth: Index = 64
  
# Do Byte and Array have to be special?
type Byte
type Array<ElementType: Type, n: Index>
  
type Number<dtype: DType>:
  # Expressions allow the simple function syntax
  data: Array<Byte, dtype.bitwdith // 8>
  
type Shape<rank: Index>:
  dims: Array<Index, rank>
  
  # Method syntax
  method num_elements(self) -> Index:
    count: Index = 1
    for dim in self.dims:
      count = count * dim
    return count

# Open question: How to specify the layout of _instances_
# of the type
# Open question: How to specify parameters of parameters
type Tensor<shape: Shape, dtype: DType>:
  data: Pointer<Number<DType>>
  # alternatively
  data: Array<Number<DType>, shape.num_elements()>
  
# Open question: How to specify input and output types
# which are parameterized. Do we even need to?
op tile<axis: Index>(x: $X) -> $Result:
  requires $X ~= Tensor
  requires $Result ~= Tensor
  requires $X.rank == $Result.rank
  requires $X.dtype == $Result.dtype
  requires axis < $X.rank
  
trait Elementwise(x: $X) -> $Result:
  requires $X == $Result
  
op sqrt(x) -> $Result:
  has trait Elementwise
  
type Literal
op constant(json: Literal) -> $Result

# Generic functions and call
# Open question: Is `List<Type>` right?
# Open question: Are these parameters?
type Function<args: List<Type>, result: Type>

# Open question: what does this look like?
op function<args: List<Type>, result: Type>():
  block body
  
op call(function: $F, args: $Args) -> $Result:
  require $F ~= Function<...>
  require $Args ~= Tuple<$F.args>
  require $Result == $F.result
  
# Does this need to be special, or is there some way to spell
# what its layout must be?
type Tuple<types: List<Type>>
```

### Validation

- `require T ~= Pattern` where `Pattern` is a type pattern or trait
- `require T == U` where `T` and `U` are types or type parameters
  - uses the function definition mini-language

### Function definitions

- _Simple_ function language
- arguments, return type must be known Types
- attribute access, indexing, python style if and return

Everything in the function language needs to be expressible
in any possible compiler implementation language! The permissable
operations must be extremely simple.

Any functions which cannot be expressed with this simple language
should be implemented manually by the specific compiler.
Providing a rich language for function definitions on
types or ops is explicitly a non-goal.
