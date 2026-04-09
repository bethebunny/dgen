# Values, Types, and Staging

## Everything Is a Value

The core abstraction is `Value[T]`, where `T` is the value's type. Ops, types, constants, block arguments, and block parameters are all `Value` subclasses.

```
Value[T]
  ├── Type           (a value whose type is TypeType)
  │     ├── TypeType (metatype; its own type is itself)
  │     └── ...      (all dialect types: Index, Float64, etc.)
  ├── Constant[T]    (compile-time data of type T)
  ├── Op             (a computation producing a value)
  ├── BlockArgument  (runtime input to a block)
  └── BlockParameter (compile-time structural input to a block)
```

Types are values of type `TypeType`. `TypeType` is its own metatype -- the recursion terminates because `TypeType.type` returns `self`.

```python
idx = Index()
idx.type        # TypeType()
idx.type.type   # TypeType() (same object — self-referential)
```

## Op Fields: Operands vs Parameters

Op subclasses are `@dataclass` classes. Fields annotated `Value` are **operands** (runtime SSA values); fields annotated as a `Type` subclass are **parameters** (compile-time values). This distinction drives staging.

```python
# From goto.dgen:
#   op branch<target: Label>(arguments: Span) -> Nil
#
# Generated class:
@dataclass(eq=False, kw_only=True)
class BranchOp(Op):
    target: Value[Label]       # parameter — compile-time
    arguments: Value[Span]     # operand — runtime
    type: Value[TypeType]
    __params__ = (("target", Label),)
    __operands__ = (("arguments", Span),)
```

Access via properties:

```python
op.operands    # Iterator[(name, Value)]    — runtime SSA values
op.parameters  # Iterator[(name, Value[T])] — compile-time values
op.blocks      # Iterator[(name, Block)]    — nested regions
```

## Constant: The Stage Boundary

`Constant[T]` wraps a `Memory[T]` buffer — a typed byte array holding the value's binary representation. `ConstantOp` is both an `Op` and a `Constant`: it embeds compile-time data as a runtime SSA value in the graph.

```python
# Create a constant integer
val = Index().constant(42)          # Constant[Index]
op  = ConstantOp.from_constant(val) # ConstantOp — an Op in the graph

# In IR text:
# %0 : index.Index = 42
```

Every per-dialect constant op (e.g. `arith.constant`, `toy.constant`) in MLIR is replaced by a single generic `builtin.constant`. The type's `__layout__` drives serialization and materialization.

Analogy: `constant` is `eval` (compile-time data becomes a runtime value). The inverse -- embedding a runtime value as compile-time data -- happens implicitly when staging resolves expressions.

## Stage Numbers

Every `Value` is assigned a **stage number** indicating when it can be evaluated:

| Value kind | Stage |
|---|---|
| `Constant`, `Type`, `FunctionOp`, `BlockParameter` | 0 |
| `BlockArgument` (function input) | 1 |
| Op (computed) | see formula |

For ops:

```
stage(op) = max(
    stage(v)     for v in __operands__,
    stage(v) + 1 for v in __params__ if stage(v) > 0 else stage(v),
)
```

Parameters only bump the stage when they depend on runtime values. A parameter that is already a constant (stage 0) does not add a stage.

Example:

```
%c2  = constant(2)             # stage 0
%c3  = constant([1,2,3])       # stage 0
%sum = add_index(%c2, %c2)     # operands only, max(0,0) = 0
%x   = block_arg               # stage 1
%t1  = tile<%sum>(%c3)         # param stage=0, no bump → stage 0
%n   = nonzero_count(%x)       # operands only, max(1) = 1
%t2  = tile<%n>(%c3)           # param stage=1 → 1+1=2 → stage 2
```

### Op.ready

`Op.ready` checks whether all `__params__` and the type are resolved (concrete `Constant` or `Type` instances, not unresolved SSA references):

```python
@property
def ready(self) -> bool:
    return self.type.ready and all(val.ready for _, val in self.parameters)
```

## Self-Describing TypeValue Format

Types serialize to a self-describing JSON dict via `Type.to_json()`:

```python
Index().to_json()
# {"tag": "index.Index", "params": {}}

from dgen.dialects.ndbuffer import Shape
Shape(rank=Index().constant(2)).to_json()
# {
#   "tag": "ndbuffer.Shape",
#   "params": {
#     "rank": {"type": {"tag": "index.Index", "params": {}}, "value": 2}
#   }
# }
```

Each parameter carries both its type descriptor and value, so deserialization needs no external schema. `Type.from_json()` reconstructs the type:

```python
data = shape_type.to_json()
reconstructed = Type.from_json(data)  # same Shape type
```

In memory, `TypeValue` is an 8-byte pointer to a `Record` laid out as:

```
Record([
    ("tag", String()),
    ("params", Record([
        (name, Record([("type", TypeValue()), ("value", <param_layout>)])),
        ...
    ]))
])
```

This makes type values fixed-size (one pointer) regardless of parameterization.

## Compile-Time Resolution

The staging system (`dgen/passes/staging.py`) resolves `__params__` boundaries before the pass pipeline runs.

### Stage-0: JIT in Isolation

When a parameter's dependency subgraph consists entirely of constants (stage 0), the staging engine:

1. Extracts the subgraph
2. Wraps it in a mini `FunctionOp`
3. Compiles it through the full pipeline (passes + codegen)
4. Executes the result
5. Patches the parameter with a `ConstantOp` holding the result

```python
# Before staging:
#   %0 = constant(2)
#   %1 = constant(2)
#   %2 = add_index(%0, %1)        ← stage 0
#   %3 = tile<%2>(%data)          ← param is unresolved Value

# After staging:
#   %2 = constant(4)              ← resolved by JIT
#   %3 = tile<%2>(%data)          ← param is now Constant, op is ready
```

### Stage-1+: Callback Thunks

When parameters depend on `BlockArgument` values (runtime inputs), isolated JIT is impossible. The staging system builds a **callback thunk**: a compiled function that passes its runtime arguments to a host callback, which then:

1. Deep-copies the IR template
2. Substitutes `BlockArguments` with the runtime values
3. Resolves remaining boundaries
4. Compiles and executes the specialized IR

```
compile_value(root, compiler)
├── resolve_stage0(root, compiler)     # resolve all stage-0 boundaries
├── if no remaining boundaries → compiler.run(resolved)
└── if stage-1+ boundaries remain:
    ├── for each function with boundaries:
    │     build_callback_thunk(func, on_call)
    │     register as global symbol (for cross-function calls)
    └── return entry executable
```

## Key Files

| File | Role |
|------|------|
| `dgen/type.py` | `Value`, `Type`, `Constant`, `TypeType`, formatting |
| `dgen/memory.py` | `Memory[T]` — typed buffer, from_json/from_raw/from_value |
| `dgen/layout.py` | Layout hierarchy, TypeValue, binary pack/unpack |
| `dgen/passes/staging.py` | `compute_stages`, `resolve_stage0`, callback thunks |
| `dgen/builtins.py` | `ConstantOp`, `PackOp` |
