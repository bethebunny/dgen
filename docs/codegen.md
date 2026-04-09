# LLVM Codegen

The codegen (`dgen/llvm/codegen.py`) emits LLVM IR text from dgen's graph-based IR and JIT-compiles it via llvmlite.

## Architecture

```mermaid
graph LR
    A[dgen Value] --> B[LLVMCodegen.run]
    B --> C[emit_llvm_ir]
    C --> D[prepare_function]
    D --> E[emit]
    E --> F[LLVM IR text]
    F --> G[Executable]
    G --> H[llvmlite JIT]
```

`LLVMCodegen` implements the `ExitPass[Executable]` protocol. It is the terminal pass in every compilation pipeline.

## Emitter Dispatch

Op-specific LLVM emission is registered via `@emitter_for`:

```python
from dgen.llvm.codegen import emitter_for

@emitter_for(llvm.AllocaOp)
def emit_alloca(op: llvm.AllocaOp) -> Iterator[str]:
    yield f"  alloca double, i64 {op.elem_count.__constant__.to_json()}"

@emitter_for(llvm.IcmpOp)
def emit_icmp(op: llvm.IcmpOp) -> Iterator[str]:
    pred = string_value(op.pred)
    ty = llvm_type(op.lhs.type)
    yield f"  icmp {pred} {ty} {vr(op.lhs)}, {vr(op.rhs)}"
```

Emitters are stored in the `EMITTERS` dict keyed by op type. The `emit(value)` function dispatches to the registered emitter and prepends `%name =` for value-producing ops. Ops listed in `_NO_ASSIGN_OPS` (structural ops like `RegionOp`, `LabelOp`, `BranchOp`, `PackOp`, `ConstantOp`, `ChainOp`) handle their own output formatting.

Unrecognized ops raise `ValueError`.

## Two-Phase Function Emission

Each `FunctionOp` is emitted in two phases:

### Phase 1: prepare_function

A single walk over the block tree that:

1. **Pre-registers SSA names** via `CodegenSlotTracker` so phi nodes can forward-reference values
2. **Records `param_to_owner`** mappings for branch target resolution (`%self` -> owning RegionOp)
3. **Populates `self_params`** set for self-parameter identification
4. **Tracks the current LLVM basic block name** through label/region nesting
5. **Records branch predecessors** with source block names and argument values

### Phase 2: emit

Walks each function's body via `emit_linearized`, which iterates `block.ops` (topological use-def order) and calls `emit(op)` for each. Control flow ops (`RegionOp`, `LabelOp`, `BranchOp`, `ConditionalBranchOp`) emit LLVM basic block labels and terminators.

### SSA naming

`CodegenSlotTracker` uses `%_0`, `%_1`, ... for unnamed values (instead of LLVM's `%0`, `%1`) to avoid LLVM's sequential numbering requirement, since pre-registration order may differ from emission order.

## Layout-to-LLVM Type Mapping

```python
llvm_type(t: Value[TypeType]) -> str
```

Maps dgen types to LLVM IR types using a combination of pattern matching on known types and fallback to layout:

| dgen Type | LLVM Type |
|-----------|-----------|
| `index.Index` | `i64` |
| `number.Float64`, `llvm.Float` | `double` |
| `number.Boolean` | `i1` |
| `number.SignedInteger(n)` | `i{max(n, layout_bits)}` |
| `memory.Reference`, `llvm.Ptr` | `ptr` |
| `builtin.Nil`, `llvm.Void` | `void` |
| `goto.Label` | `label` |
| Non-register-passable | `ptr` |
| Register-passable fallback | Layout struct format lookup |

### Calling convention

Register-passable types are passed by value; non-register-passable types are passed by pointer. The `call()` function handles the ctypes bridge:

```python
# Register-passable: unpack scalar from Memory, pass directly
raw_args = [m.unpack()[0] if t.__layout__.register_passable else m.address
            for m, t in zip(memories, input_types)]
```

## Executable

`Executable` wraps compiled LLVM IR text with type metadata:

```python
@dataclass
class Executable:
    ir: str                    # LLVM IR text
    input_types: list[Type]    # argument types
    result_type: Type          # return type
    main_name: str             # entry function name
    host_refs: list            # objects kept alive for JIT lifetime
```

JIT compilation is lazy (via `@cached_property`): the first call to `run()` parses the IR, creates an MCJIT engine, and looks up the function address.

```python
exe = compiler.compile(ir)
result = exe.run(42, 3.14)    # → Memory object
result.to_json()              # → Python value
```

`run()` accepts `Memory` objects or raw Python values (auto-converted via `Memory.from_value`). The result is always a `Memory` object.

## Control Flow Emission

### RegionOp (inline regions)

Regions execute inline. Two emission modes:

1. **With initial args** (loops): single block with phi nodes at entry, body inline
2. **With merge args but no initial args** (if-merge): two blocks -- `{name}_entry` for dispatch, `{name}` for the merge phi

Every region emits its `%exit` parameter as a separate LLVM basic block after the region body.

### LabelOp (jump targets)

Labels are jump targets only, not reachable by fall-through. Emission:

1. `br label %{name}_exit` -- skip the label body (unreachable by fall-through)
2. `{name}:` -- the label block with phi nodes
3. Body ops
4. `{name}_exit:` -- resume point

### Phi generation

Phi nodes are generated from the predecessor map built during `prepare_function`. For each block arg, emit:

```llvm
%arg = phi i64 [ %val0, %block0 ], [ %val1, %block1 ]
```

### Branch emission

- `goto.BranchOp` -> `br label %target`
- `goto.ConditionalBranchOp` -> `br i1 %cond, label %true, label %false` (with automatic `icmp ne` conversion for non-i1 conditions)

## Key Files

| File | Role |
|------|------|
| `dgen/llvm/codegen.py` | `LLVMCodegen`, `emit_llvm_ir`, `Executable`, `call`, emitter dispatch |
| `dgen/llvm/algebra_to_llvm.py` | Algebra dialect emitters |
| `dgen/llvm/builtin_to_llvm.py` | Builtin dialect emitters |
| `dgen/llvm/memory_to_llvm.py` | Memory dialect emitters |
