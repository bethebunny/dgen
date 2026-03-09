# Remove Variadics Phase 1: generic_call and phi

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete `toy.generic_call` (use `builtin.call` instead) and make `llvm.phi` take exactly 2 inputs (chainable for 3+).

**Architecture:** generic_call is redundant with builtin.call — only differs by using string callee vs SSA ref. phi currently takes variadic lists; switching to 2-input makes it the same fixed-arity pattern as all other ops.

**Tech Stack:** Python, pytest

---

### Task 1: Delete generic_call from toy.dgen, switch lowering to builtin.call

**Files:**
- Modify: `toy/dialects/toy.dgen` — remove `generic_call` line
- Regenerate: `toy/dialects/toy.py`
- Modify: `toy/parser/lowering.py` — produce `builtin.CallOp` instead of `GenericCallOp`
- Modify: `toy/passes/shape_inference.py` — handle `builtin.CallOp` instead of `GenericCallOp`

**Step 1: Remove generic_call from toy.dgen**

Delete line 18: `op generic_call<callee: String>(args: list) -> Tensor`

Regenerate: `python -m dgen.gen toy/dialects/toy.dgen -I affine=toy.dialects.affine > toy/dialects/toy.py`

**Step 2: Update lowering.py**

In `toy/parser/lowering.py`, change the "Generic call" section (lines 211-221).

The lowering currently creates:
```python
op = toy.GenericCallOp(
    callee=builtin.String().constant(call.callee),
    args=args,
    type=toy.InferredShapeTensor(),
)
```

Change to create a `builtin.CallOp` with a forward-reference callee:
```python
from dgen import Value
from dgen.dialects.builtin import CallOp, PackOp, List as ListType

# Forward-reference to the callee function
callee_ref = Value(name=call.callee, type=builtin.Nil())
# Wrap args in PackOp
element_type = toy.InferredShapeTensor()
pack = PackOp(values=args, type=ListType(element_type=element_type))
yield pack
op = CallOp(
    callee=callee_ref,
    args=pack,
    type=toy.InferredShapeTensor(),
)
```

**Step 3: Update shape_inference.py**

Change the `GenericCallOp` branch (lines 90-113) to handle `builtin.CallOp`:
- Access args via `op.args.values` (PackOp) instead of `op.args` (list)
- Look up callee via `op.callee.name` instead of `string_value(op.callee)`

```python
elif isinstance(op, builtin.CallOp):
    args_list = op.args.values if isinstance(op.args, PackOp) else [op.args]
    resolved = [type_of.get(id(a)) for a in args_list]
    arg_types = [t for t in resolved if t is not None]
    if len(arg_types) == len(resolved):
        callee = func_map.get(op.callee.name)
        ...
```

**Step 4: Run tests, fix test expectations**

Tests in `toy/test/` will need IR strings updated:
- `toy.generic_call<"name">([...])` → `call<%name>([...])`

Run: `python -m pytest toy/test/ -q`

**Step 5: Commit**

```
jj commit -m "delete toy.generic_call, use builtin.call instead"
```

### Task 2: Make llvm.phi take exactly 2 inputs

**Files:**
- Modify: `dgen/dialects/llvm.dgen` — change phi signature
- Regenerate: `dgen/dialects/llvm.py`
- Modify: `dgen/codegen.py` — emit phi from 2 fixed fields
- Modify test IR strings

**Step 1: Change phi in llvm.dgen**

Change line 28 from:
```
op phi<labels: list<String>>(values: list) -> Nil
```
To:
```
op phi<label_a: String, label_b: String>(a, b) -> Nil
```

Regenerate: `python -m dgen.gen dgen/dialects/llvm.dgen > dgen/dialects/llvm.py`

**Step 2: Update codegen.py**

Change phi code generation (lines 234-240) from iterating lists to using 2 fixed fields:

```python
elif isinstance(op, llvm.PhiOp):
    ty = types.get(id(op), "i64")
    lines.append(
        f"  %{name} = phi {ty} "
        f"[ {bare_ref(op.a)}, %{string_value(op.label_a)} ], "
        f"[ {bare_ref(op.b)}, %{string_value(op.label_b)} ]"
    )
```

Also update type registration (lines 140-142):
```python
elif isinstance(op, llvm.PhiOp):
    types[vid] = types.get(id(op.a), "i64")
```

**Step 3: Update passes that create PhiOp**

`dgen/passes/builtin_to_llvm.py` (lines 130-137) — already creates exactly 2:
```python
phi_op = llvm.PhiOp(
    a=then_result,
    b=else_result,
    label_a=String().constant(then_source_label),
    label_b=String().constant(else_source_label),
)
```

`toy/passes/affine_to_llvm.py` (lines 188-196) — already creates exactly 2:
```python
phi_op = llvm.PhiOp(
    a=init_op,
    b=back_edge,
    label_a=String().constant(prev_label),
    label_b=String().constant(body_label),
)
```

**Step 4: Update test IR strings**

Change phi syntax in tests from:
```
%5 : Nil = llvm.phi<["entry", "loop_exit1"]>([%2, %33])
```
To:
```
%5 : Nil = llvm.phi<"entry", "loop_exit1">(%2, %33)
```

**Step 5: Run tests**

Run: `python -m pytest . -q`

**Step 6: Commit**

```
jj commit -m "make llvm.phi take exactly 2 inputs, remove variadic param"
```
