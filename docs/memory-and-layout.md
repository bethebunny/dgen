# Memory and Layout

## Layout Hierarchy

Every type declares a `__layout__: Layout` -- a language-agnostic description of its binary memory representation. The layout drives serialization (`to_json`), deserialization (`from_json`), and LLVM type mapping.

**Wire format = memory format.** Values are stored identically in IR serialization and at runtime. This enables mmap/memcpy-friendly JIT patterns.

```
Layout
  ├── Void          0 bytes, register_passable
  ├── Byte          1 byte (uint8), register_passable
  ├── Int           8 bytes (i64), register_passable
  ├── Float64       8 bytes (f64), register_passable
  ├── Pointer(T)    8 bytes (ptr to T), register_passable
  ├── Array(T, n)   n * sizeof(T) bytes, inline
  ├── Span(T)       16 bytes (ptr + i64 length)
  ├── Record(fields) sequential field layout
  ├── String        Span(Byte) with str ↔ bytes conversion
  └── TypeValue     8-byte pointer to self-describing Record
```

Each layout has a `struct: struct.Struct` for binary pack/unpack and a `byte_size` property.

### register_passable

Scalars and pointers (`Void`, `Byte`, `Int`, `Float64`, `Pointer`) are `register_passable = True` -- they fit in a CPU register and are passed by value in function calls.

Aggregates (`Array`, `Span`, `Record`, `String`) are `register_passable = False` -- they are passed by pointer. Codegen maps non-register-passable types to `ptr` in LLVM IR.

```python
from dgen.layout import Int, Span, Byte

Int().register_passable      # True  → LLVM "i64", ctypes c_int64
Span(Byte()).register_passable  # False → LLVM "ptr", ctypes c_void_p
```

## Memory[T]: Typed Buffer

`Memory[T]` pairs a `Layout` with a `bytearray` buffer. It is the ABI-level representation of any value.

### Construction

```python
from dgen.memory import Memory
from dgen.dialects.index import Index

# From a Python value (JSON-compatible)
mem = Memory.from_json(Index(), 42)

# From a Python value with str/bytes conversion
mem = Memory.from_value(String(), "hello")

# From a raw pointer (e.g. JIT result)
mem = Memory.from_raw(Index(), address)
```

### Readback

```python
mem.to_json()    # → 42 (Python value)
mem.unpack()     # → (42,) (raw struct.unpack tuple)
mem.address      # → raw memory address of the buffer
```

### Origins

For pointer-based layouts (`Span`, `Pointer`), the buffer contains raw pointers into backing data. `Memory.origins` is a list of objects whose lifetime must extend through the Memory's lifetime. Origins are shared (not copied) on `deepcopy` so packed pointers stay valid.

## TypeValue Layout

`TypeValue` is an 8-byte pointer to a self-describing `Record`. The record mirrors the JSON structure from `Type.to_json()`:

```
Record([
    ("tag", String()),                              # e.g. "ndbuffer.Shape"
    ("params", Record([
        ("rank", Record([
            ("type", TypeValue()),                  # recursive: type of the param
            ("value", Int()),                       # param value in its own layout
        ])),
    ]))
])
```

This makes every type value self-describing in memory: you can read the tag to determine the concrete type, then read each parameter's type descriptor to know how to interpret its value bytes.

```python
from dgen.layout import TypeValue

tv = TypeValue()
tv.byte_size          # 8 (one pointer)
tv.register_passable  # False (indirection through pointer)
```

## Constant Materialization in Codegen

When codegen encounters a `Constant` (or `ConstantOp`), it materializes the value differently based on the layout:

**Register-passable** values are emitted as immediate LLVM constants:

```python
# Index constant 42 → "42" in LLVM IR
# Float64 constant 3.14 → "3.14" in LLVM IR
# Pointer constant → "inttoptr (i64 <addr> to ptr)"
```

**Non-register-passable** values (aggregates) are materialized via `inttoptr`: the `Memory` buffer's raw address is embedded as an integer constant and cast to a pointer. The `Memory` object is kept alive in `EmitContext.host_buffers` for the lifetime of the JIT engine.

```python
# Span/Array/Record constant:
#   inttoptr (i64 140234567890 to ptr)
# The Memory buffer lives in host_buffers, accessible via this address.
```

This avoids copying aggregate constants into the LLVM module -- the JIT code reads directly from the Python-side `Memory` buffers.

## Key Files

| File | Role |
|------|------|
| `dgen/layout.py` | Layout classes, TypeValue, binary pack/unpack |
| `dgen/memory.py` | `Memory[T]`, from_json/from_raw/from_value, to_json |
| `dgen/llvm/codegen.py` | `value_reference` -- constant materialization |
| `dgen/type.py` | `Type.to_json` / `Type.from_json` -- TypeValue serialization |
