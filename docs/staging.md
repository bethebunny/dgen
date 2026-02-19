# Compile-Time Types and Staging

## 1. Motivation

The codebase has an implicit pattern: every type has both a compile-time and a runtime face, but this is encoded ad-hoc:

| Type concept | Compile-time (IR literal) | Runtime (SSA Value) | Materialization |
|---|---|---|---|
| String | `StaticString` (`str`, serialized `"..."`) | `String` builtin type | not implemented |
| Scalar | `float` / `int` Python literal | `Value` of type `f64` / `index` | `ArithConstantOp` → `llvm.fconst` |
| Tensor | `list[float]` + `Shape` | `Value` of type `tensor<SxT>` | `toy.ConstantOp` → alloc+stores |
| Label | `StaticString` (e.g. `BrOp.dest`) | n/a (always compile-time) | direct emission |

Each has bespoke handling in the serializer (`_format_value`), parser (`_parse_value`), and lowering passes (`_lower_constant`). The connection between compile-time and runtime forms is implicit.

**This is analogous to quote and eval in Lisp:**
- **`quote`**: embed a value as compile-time data (`String` → `StaticString`)
- **`eval`**: materialize compile-time data as a runtime value (`StaticString` → `String`)
- Every "constant" op is a `quote` — it introduces compile-time data into the IR as a runtime value

**Goals:**
1. Types declare their runtime memory layout. Compile-time and runtime layouts are the same.
2. Per-dialect constant ops are replaced by a generic **`constant`** operation (a generalized `quote`)
3. Materializing compile-time data to runtime is generic, driven by the type's memory layout
4. A staging pass splits code into compile phases

---

## 2. Core Concepts

### 2.1 Types Have Two Faces

Every type T has:

- **Compile-time Type** — the abstract type as known to the compiler (`String`, `Tensor<2x3xf64>`, `f64`)
- **IR representation IR(T)** — how literal values appear in IR text.
  - Generally JSON types
  - Binary blob strings `b"..."` for types that don't translate well from JSON.
- **Runtime memory layout RT(T)** — how values are stored in memory at execution time

**Key principle:** The compiler and runtime share a memory layout. This enables JIT-friendly patterns: wire format = memory format, mmap/memcpy-friendly.

**Constraint (for now):** Each type has exactly one memory layout. The layout is an attribute of the type itself.

### 2.2 Memory Layouts

A layout is a declarative description of how a type is stored in memory. These will _not_ necessarily be convenient types for Python to use. They are designed to be language-agnostic representations.

The Python bindings can generate helpers for conveniently using these types in passes, but it should respect the correct memory representation.

For now the names for primitive memory representation types are based on the Mojo names for those types, eg. `Pointer`, `Int`, `Float32`, `UInt8`/`Byte`.

```
class Struct:
    pass
    
# Inline array type
class Array[T: Struct, n: int]
    ???
    
class Int(Struct):
    data: Array[Byte, 8]
    
class Pointer[T: Struct](Struct):
    data: Int
    
class FatPointer[T: Struct](Struct):
    data: Pointer[T]
    size: Int
```

```
class Type:
    __layout__: type[Struct]
    
class String:
    __layout__ = FatPointer[Byte]
    
class List[T]:
    __layout__ = FatPointer[T]
```

### 2.3 Constant: The Generic Quote

`constant` is a builtin op that replaces all per-dialect constant ops. It takes compile-time data and introduces it as a runtime SSA value:

```
# Today (ad-hoc per-dialect):
%0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
%1 = affine.arith_constant(1.0)
%2 = llvm.fconst(3.14)

# Proposed (generic):
%0 = constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
%1 = constant(1.0) : f64
%2 = constant("hello") : String
```

The `type` annotation determines:
- How `data` is serialized (via IR(T))
- How `data` is materialized at runtime (via RT(T))

`constant` is the canonical way to introduce compile-time-known values into the IR. "By the end we should be able to eliminate all of them" — every `constant` gets materialized (lowered into memory operations) by the appropriate lowering pass.

### 2.4 Materialization: How Constant Gets Lowered

Materialization replaces a `constant` op with the runtime operations that put the data in memory. This is driven by the type's memory layout. For JIT lowering this is just a memcpy of the compile-time memory layout. For printing to ASM any pointers need to be serialized to data in the executable. These can be local literal binary buffers in the function source for now.

### 2.5 Op Fields: Compile-Time vs Runtime

Op fields implicitly declare whether they hold compile-time or runtime data based on their Python type annotation:

| Field type hint | Meaning | Stage |
|---|---|---|
| `Value[T]` | SSA reference of type T | Runtime |
| `T` | Constant | Compile-time |

### 2.6 Quote: The Inverse Direction

While `constant` is eval (CT → RT: compile-time data becomes a runtime value), **quote** is the inverse (RT → CT): taking a value that could exist at runtime and embedding it as compile-time data.

Quote is implicit in how compile-time computations produce data. If a stage-0 computation produces a value, that value becomes compile-time data available for stage-1. The staging pass (§2.7) determines which computations can be "lifted" to an earlier stage.

Compile-time data enters the IR only through literal `constant` ops.

### 2.7 Staging Model

A **stage** is a unit of compilation + execution:

| Stage | Role | When it runs |
|---|---|---|
| <1 | Main compiler | At compile time |
| 1  | Main executable | At run time |
| 2+ | JIT-compiled code | Generated and executed dynamically |

Stages compile and execute in ascending order: stage 0 produces stage 1's code, stage 1 runs, and if it generates code, that's stage 2 (JIT).

**`constant` is the stage boundary.** A `constant` op contains data known at stage N and produces a value for stage N+1. Materialization is what happens at the boundary: the type's layout drives the code that stores compile-time data into runtime memory.

**Relation to current pipeline:**
- Stage <1: All lowering passes (`toy_to_affine`, `affine_to_llvm`, `codegen`) — the compiler
- Stage 1: The JIT-compiled LLVM IR that runs via llvmlite
- No stage 2 yet (but the regex JIT milestone will need it: regex pattern → NFA → JIT matcher)

**Multiple stages:** If a compile-time computation depends on another compile-time computation, this creates a new lower stage. For instance, a type of an op may depend on an SSA value. A staging pass examines the IR and assigns each op to a stage. All `constant` ops at stage N must be materialized before stage N+1 can execute.
