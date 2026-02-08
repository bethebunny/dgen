# DGEN

DGEN is a dialect generation tool inspired by LLVM's TableGen.

MLIR has been a paradigm shift for compilers. It brings modularity and composability to compiler stacks. The implementation of MLIR leaves a lot on the table. DGEN attempts to solve a few key weaknesses of MLIR through its design.

### MLIR is _very_ slow.

Its design enables hugely powerful, even futuristic JIT compilers, but its speed makes it impractical to use in a JIT scenario.

Much of this slowness is a consequence of MLIR's memory layouts. The core types are all multiple pointer indirections with dynamic sizes. Even trivial types and attributes are interned, which is hostile to CPU caches. The bytecode format is slow to read and write.

### MLIR is de facto C++ only.

The MLIR TableGen implementation deeply assumes that the code generation target is C++. There's a JSON backend, which _does_ allow alternate generation backends, but in practice dialects also heavily use `extraClassDefinition` to hardcode implementations for behaviors in C++.

MLIR pays a very high tax for using C++. A large percentage of the code is dedicated to data structures, algorithms, passes, and design patterns that would be unnecessary in a more modern programming language.

### It's de facto impossible to dynamically share MLIR passes. 

There's two separate reasons that it's hard to define a pass "plugin" infrastructure in MLIR. Conceptually these are 1) MLIR's pass **ABI** is subtley extremely finnicky, and 2) there's no defined pass **API**.

MLIR's pass infrastructure is C++. It's technically possible to share passes. MLIR re-implements its own RTTI in C++, and core dialect operations like "is this type the same as this other type" rely on the dialect metadata's TypeIDs. It's proven to be broadly impractical to share dialect definitions across shared object boundaries. I don't know of a single project allowing "pluggable" MLIR passes. Practically the best way to actually share passes would be outside the pass infrastructure, serializing MLIR to asm or bytecode to another process.

Even supposing you got through the technical hurdles, there's no clear API definition for a pass. Technically a pass is a mutating function on a single op. Practically passes refine large MLIR documents. There's no way to define the input and output types of these refinements; for instance, does a lowering pass from dialect A -> B accept a mixed A | B input document? Will it always lower all ops from A? Frequently canonicalization A -> A always results in a _subset_ of A, can we type that?

### MLIR has evolved beyond its data model

MLIR's data model has a number of quirks.

- A `trait` is just an `interface` with no methods.
- `Attribute`s and `Value`s aren't typed, rather in practice all values are `TypeValue`s and all attributes are `TypedAttributes`.
- For that matter, what is an `Attribute`? It's a `Value` that's known at compile time, but TableGen needs to generate C++ types for all attributes, so in practice users just define `<type>Attr` for all types of values that are useful at compile time.

## DGEN 

DGEN is a replacement for TableGen. DGEN doesn't aim to solve all of these problems at once, rather it is an incremental step towards an MLIR compiler stack that can effectively and performantly support JIT applications. Ideally DGEN may interoperate with MLIR, replacing just TableGen, while also enabling future evolutions.

Comparing DGEN to TableGen:

- DGEN is specialized for dialect generation. LLVM TableGen is very general, but that generality gets in the way of its effectiveness for dialect specification and generation.
- DGEN has a formal specification. TableGen's specification is its source code, which makes it very hard to build on top of or rely on.
- DGEN is target language independent. It does not support any `extraClassDefinition`. DGEN provides a simple language for function implementations that can be translated to the target language. All behaviors must be implmented this way, or defined directly in the target language for pass definitions.
- DGEN specifies a default memory representation for types and ops. This memory representation hasn't been finalized. Its goals are 1) facillitate performant JIT applications on CPU and GPU, 2) the wire format is the same as the memory format, ie. reading/writing/copying may be performed via mmap or memcpy, and 3) provide a default implementation for writing JITs and dependent type systems. It is not intented to preclude implemetations from specializing memory representations for performance in specific applications.
- DGEN does not hold itself to MLIR's data model. There are patterns that TableGen can generate (eg. untyped values) that DGEN cannot. However, DGEN provides a MLIR generation backend. It should be possible for most existing MLIR dialects to be expressed in DGEN and generated via the MLIR generation backend.

### DGEN data model

- Types
  - may be parameterized on values
  - may be values; the type of a type value is `!builtin.type`.
- Values
  - have a type
  - may or may not be known
- Regions
  - are a sequence of Ops
- Ops
  - are a statically known operation kind
  - have named regions depending on the operation kind
  - have 0 or more input values
  - always have exactly one output value
