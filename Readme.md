<p align="center">
  <img src="assets/logo.svg" alt="dgen — the dialect generation toolkit" width="640" />
</p>

A language-agnostic specification, ABI, and toolchain for defining IR dialects — and a JIT-first research compiler framework built on top of it.

## Declaring a dialect

The whole data model is `Value`, `Type`, `Trait`, `Op`, `Block`:

- **Everything is a value**, including types.
- **Every value has a type**, which is itself a value in the same SSA graph.
- **Types are first-class.** Ops can take types as parameters and return types as results.
- **Every op has exactly one result**, so an op *is* its result value. Same for blocks.
- **Types declare their runtime layout.** Any value (including a type) round-trips to and from a JSON literal, and any compiler built on dgen gets a JIT for free.

Dialect declarations are formally specified `.dgen` files — here is a slice of the Toy dialect:

```python
from builtin import Index, Nil, Span
from number import Float64
import ndbuffer

type Tensor<shape: ndbuffer.Shape, dtype: Type = Float64>:
    data: Span<dtype>

op transpose(input: Tensor) -> Tensor
op mul(lhs: Tensor, rhs: Tensor) -> Tensor
op concat<axis: Index>(lhs: Tensor, rhs: Tensor) -> Tensor:
    requires axis < lhs.shape.rank
op print(input: Tensor) -> Nil
```

## IR and ASM

- **Text assembly** with a clear specification and unambiguous parsing
- **Local structure** — blocks must explicitly declare captured values, so any analysis can run on a block in isolation
- **Acyclic** — no forward references; the IR and the textual ASM are always topologically sorted
- **Fully use-def connected** — ops not reachable from a block's result are dead and ignored, with no implicit linearization

```mlir
import function
import index
import ndbuffer
import number
import toy

%main : function.Function<[], Nil> = function.function<Nil>() body():
    %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%0)
    %2 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.transpose(%1)
    %3 : Nil = toy.print(%2)
```

## Transforming IR

Passes are decorator-registered handlers that pattern-match on op types. Here is the entire double-transpose elimination pass:

```python
from dgen.passes.pass_ import Pass, lowering_for
from toy.dialects import toy

class EliminateDoubleTranspose(Pass):
    @lowering_for(toy.TransposeOp)
    def fold(self, op):
        if isinstance(op.input, toy.TransposeOp):
            return op.input.input
        return None
```

Run it on an IR file:

```python
from dgen import asm

input_ir = asm.parse(open("input.asm").read())
output_ir = EliminateDoubleTranspose().run(input_ir)
print(asm.format(output_ir))
```

Input:

```mlir
%main : function.Function<[], Nil> = function.function<Nil>() body():
    %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %1 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([3, 2]), number.Float64> = toy.transpose(%0)
    %2 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = toy.transpose(%1)
    %3 : Nil = toy.print(%2)
```

Output:

```mlir
%main : function.Function<[], Nil> = function.function<Nil>() body():
    %0 : toy.Tensor<ndbuffer.Shape<index.Index(2)>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    %3 : Nil = toy.print(%0)
```

Passes compose into progressive lowerings — `toy → structured → control_flow → goto → ndbuffer → memory → llvm` — and the result is JIT-executed via `llvmlite`.

## JIT and generic execution

Dependent types — types parameterized on values — work the same way whether the value is known at compile time or only at runtime. Both fall out of the staging engine: any value referenced as a compile-time dependency that hasn't yet become a `Constant` is either folded ahead of time (if its subgraph reaches no runtime input) or deferred to a callback thunk that specializes the function on every call.

**Compile time.** A `number.SignedInteger` whose bit width is computed from a constant expression:

```mlir
import algebra
import index
import number

%w1 : index.Index = 8
%w2 : index.Index = 16
%w  : index.Index = algebra.add(%w1, %w2)
%x  : number.SignedInteger<%w> = 42
```

Before the pass pipeline runs, the staging engine extracts the dependency subgraph for `%w` — `algebra.add(%w1, %w2)` — JIT-evaluates it in isolation, and patches `%x.type.bits` with the resulting `Constant(24)`. The value then lowers exactly as if you had written `number.SignedInteger<index.Index(24)>` directly:

```bash
$ python -m dgen examples/dependent_types/compile_time_signed_integer.dgen.asm
42
```

**Runtime.** The same shape, but with the width as a function argument:

```mlir
import algebra
import function
import index
import number

%main : function.Function<[index.Index, index.Index], number.SignedInteger<index.Index(64)>> = function.function<number.SignedInteger<index.Index(64)>>() body(%bits: index.Index, %v: index.Index):
    %x : number.SignedInteger<%bits> = algebra.cast(%v)
```

The compiler can't fold `%bits` ahead of time — it depends on a function argument — so the staging engine builds a **callback thunk**: a stage-1 LLVM function whose body calls back into the host on every invocation. On each call, the host substitutes `%bits` with the runtime value, re-runs the resolver (which can now fold `number.SignedInteger<%bits>` because every leaf is a constant), JIT-compiles a stage-2 specialization for that exact width, and runs it. Because every type carries a memory `Layout`, dgen can read and write any function's arguments and return values directly to and from JSON — `python -m dgen` is a generic JSON-in/JSON-out runner for any compiled IR file:

```bash
$ python -m dgen examples/dependent_types/runtime_signed_integer.dgen.asm '8' '42'
42
$ python -m dgen examples/dependent_types/runtime_signed_integer.dgen.asm '24' '999'
999
$ python -m dgen examples/dependent_types/runtime_signed_integer.dgen.asm '64' '12345'
12345
```

The same compiled program produces a fresh stage-2 body for every distinct runtime width. Constant folding, dependent types, and runtime specialization are three faces of one mechanism.

## Quick start

```bash
git clone https://github.com/bethebunny/dgen.git && cd dgen
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

## Read more

- [`docs/dialect-files.md`](docs/dialect-files.md), [`docs/asm.md`](docs/asm.md) — the `.dgen` language and the textual IR
- [`docs/memory-and-layout.md`](docs/memory-and-layout.md) — the layout system
- [`docs/staging.md`](docs/staging.md), [`docs/staged-computation.md`](docs/staged-computation.md) — dependent types and runtime specialization
- [`docs/passes.md`](docs/passes.md), [`docs/codegen.md`](docs/codegen.md) — the pass framework and the LLVM backend
- [`examples/toy/`](examples/toy/), [`examples/dcc/`](examples/dcc/), [`examples/actor/`](examples/actor/) — worked example dialects
