# Memory and Layout

## Overview

Every DGEN type declares a `__layout__` describing its binary memory representation. `Layout` is the structural description; `Memory` is a live buffer typed by a DGEN type. Together they provide three capabilities:

1. **Passing values between stages** — compile-time data materializes to runtime memory through the same layout the JIT code reads.
2. **Generic runtime JIT** — function arguments and return values are marshalled through `Layout.prepare_arg` and `Memory`, so the staging evaluator works for any type without per-type special cases.
3. **Generic literal parsing** — `Memory.from_asm(type, text)` parses an ASM literal string into a packed buffer using only the type's layout, with no type-specific parser code.

## Layout

`Layout` (`dgen/layout.py`) is a value-level descriptor of binary encoding. Each layout wraps a `struct.Struct` format string:

| Layout | Format | Size | Example type |
|--------|--------|------|--------------|
| `Byte` | `B` | 1 | — |
| `Int` | `q` | 8 | `index` |
| `Float64` | `d` | 8 | `f64` |
| `Array(elem, n)` | `n × elem` | `n * elem.size` | `Tensor([2,3], f64)` → `Array(Float64, 6)` |
| `Pointer(T)` | `P` | 8 | — |
| `FatPointer(T)` | `PQ` | 16 | `String` |

Primitive layouts are module-level singletons (`INT`, `FLOAT64`, `BYTE`). Compound layouts are constructed.

A type declares its layout as a class attribute:

```python
@builtin.type("f64")
@dataclass(frozen=True)
class F64Type:
    __layout__ = FLOAT64

@dataclass(frozen=True)
class TensorType:
    shape: list[int]

    @cached_property
    def __layout__(self):
        return Array(FLOAT64, math.prod(self.shape))
```

The `Type` protocol requires `__layout__: Layout`, so every type in the system has one.

### `parse` and `prepare_arg`

Each layout has two methods that enable generic value handling:

- **`parse(obj)`** validates and coerces a Python value (from IR parsing or user input) to the layout's expected form. `Int.parse` asserts an int; `Array.parse` recursively parses each element.

- **`prepare_arg(value, type)`** converts a Python value into a ctypes-compatible argument for JIT function calls. Scalars pass through directly. Arrays allocate a `Memory` buffer, pack the values, and return a pointer:

```python
class Array(Layout):
    def prepare_arg(self, value, type=None):
        mem = Memory(type) if type is not None else Memory._from_layout(self)
        mem.pack(*value)
        return mem.ptr, [mem]   # pointer + ref to keep buffer alive
```

## Memory

`Memory` (`dgen/layout.py`) is a typed buffer — a `bytearray` paired with the DGEN `Type` that gives it meaning. It provides:

- **`pack(*values)` / `unpack()`** — write/read Python values via the type's `struct.Struct`.
- **`ptr`** — a `ctypes.c_void_p` to the buffer, suitable for passing to JIT-compiled functions.
- **`address`** — the raw integer address, used in codegen to emit `inttoptr` constants.

### Construction

```python
# From a Type (allocates a zeroed buffer)
mem = Memory(some_type)

# From a Type + Python value (parse + pack)
mem = Memory.from_value(F64Type(), 3.14)

# From a Type + ASM literal string (parse text + parse value + pack)
mem = Memory.from_asm(TensorType(shape=[3]), "[1.0, 2.0, 3.0]")
```

## How the pieces connect

### 1. Constant materialization in codegen

When codegen encounters a `ConstantOp` with an array value, it creates a `Memory` from the op's type, packs the compile-time data, and emits the buffer's address as an `inttoptr` constant. The JIT code reads directly from that buffer — no serialization boundary, the compile-time layout IS the runtime layout.

```python
# dgen/llvm/codegen.py
mem = Memory(op.type)
mem.pack(*op.value)
host_buffers.append(mem)  # prevent GC
constants[vid] = f"ptr inttoptr (i64 {mem.address} to ptr)"
```

### 2. Argument marshalling in staging

The staging evaluator JIT-compiles subgraphs to resolve compile-time values. When a subgraph depends on function parameters (stage-1), the runtime arguments must be passed to the JIT. `_prepare_ctypes_args` walks the parameters and uses each type's layout to marshal:

```python
# dgen/passes/staging.py
for arg, param in zip(python_args, block_args):
    ct_val, refs = param.type.__layout__.prepare_arg(arg, param.type)
```

For scalars, `prepare_arg` returns the value directly. For arrays, it allocates a `Memory`, packs, and returns the pointer. The staging evaluator doesn't need to know what type it's dealing with — the layout handles it.

### 3. CLI argument parsing

The CLI accepts arguments as strings. `run()` parses them via the ASM expression parser (which handles ints, floats, and lists), then they flow through the same `prepare_arg` path:

```python
# toy/cli.py
args = [_parse_arg(a) if isinstance(a, str) else a for a in args]
```

```
$ python -m toy.cli program.toy '[1.0, 2.0, 3.0]'
```

`Memory.from_asm` composes these steps — parse the text, validate against the layout, pack into a buffer — for cases where you want a typed buffer directly from a string literal.

## Design properties

**Wire format = memory format.** The layout that `struct.Struct` describes is the same layout the JIT code reads. No marshalling layer, no serialization protocol. `Memory.address` gives you a pointer the JIT can dereference.

**Type-driven, not value-driven.** The type's `__layout__` determines everything: buffer size, pack/unpack format, how to prepare arguments, how to parse literals. Adding a new type with a new layout requires no changes to codegen, staging, or the CLI.

**One mechanism for all stages.** The same `Memory` + `Layout` pair handles: compile-time constant buffers in codegen, argument passing for stage-1 JIT evaluation, and CLI input parsing. There's one path through the system, not three.
