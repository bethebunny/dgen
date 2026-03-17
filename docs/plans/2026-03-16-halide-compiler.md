# Plan: Halide-Style Image Pipeline Compiler on DGEN

## Context

Halide (Ragan-Kelley et al., PLDI 2013) separates image pipeline *algorithms* (what to
compute) from *schedules* (how to compute it). The algorithm is a pure functional
definition; the schedule controls loop nesting, tiling, fusion, vectorization, and
parallelism. Changing the schedule never changes the output — only performance.

DGEN's staging system is a natural fit for this separation. Schedule parameters become
`__params__` on ops. When schedule values are compile-time constants (stage-0), the
compiler specializes loop nests eagerly. When schedule values depend on runtime input
(stage-1+), the staging system generates JIT callback thunks that specialize at runtime.
Same lowering code, same pass pipeline, different timing. This is DGEN's differentiator
over Halide's own implementation, where AOT and JIT are separate code paths.

### Prior art

- Halide (MIT/Adobe, 2013): algorithm/schedule separation, `compute_at`/`store_at`,
  auto-tuning via OpenTuner or Halide's learned auto-scheduler
- MLIR Affine dialect: polyhedral loop representation with affine maps
- TVM (Apache): Halide-inspired scheduling for deep learning, with auto-tuning
- Tiramisu (MIT, 2019): polyhedral compiler with Halide-like scheduling API

### Relationship to existing code

The affine dialect (`toy/dialects/affine.dgen`) provides `ForOp`, `LoadOp`, `StoreOp`,
`AllocOp`, `MemRef`. The `toy_to_affine.py` pass builds nested loop nests from tensor
shapes. The `affine_to_llvm.py` pass lowers loops to LLVM CFG form with phi nodes and
index linearization. The actor framework's fusion decision is a rudimentary `compute_at`.

## Design

### Phase 1: Algorithm representation

#### New ops: `halide` dialect

```dgen
from builtin import Index, Nil, F64, HasSingleBlock
from affine import Shape, MemRef

op func<name: String, shape: Shape>(args: List):
  block body
  has trait HasSingleBlock

op var<dim: Index>()

op call(func, args: List)
```

**`func`**: A Halide `Func` — a pure function from coordinates to values. The `shape`
parameter defines the output domain. The body computes one output element given
coordinate variables. `args` lists input `MemRef`s or other `func` results.

**`var`**: A dimension variable (like Halide's `Var`). Represents a coordinate in the
output domain. `dim` identifies which axis (0=x, 1=y, ...).

**`call`**: Reads from a `func` at given coordinates. In a stencil, `call(blur_x, [x-1, y])`
reads the neighbor. The lowering decides whether this is an inline computation (fused)
or a buffer load (pre-computed).

#### Example: 3x3 box blur

```
// Algorithm
%x = halide.var<0>()
%y = halide.var<1>()

%blur_x = halide.func<"blur_x", [W, H]>([%input]) ():
  %left  = halide.call(%input, [sub(%x, 1), %y])
  %mid   = halide.call(%input, [%x, %y])
  %right = halide.call(%input, [add(%x, 1), %y])
  %sum   = add_f(%left, add_f(%mid, %right))
  %avg   = div_f(%sum, 3.0)
  return(%avg)

%blur_y = halide.func<"blur_y", [W, H]>([%blur_x]) ():
  %top = halide.call(%blur_x, [%x, sub(%y, 1)])
  %mid = halide.call(%blur_x, [%x, %y])
  %bot = halide.call(%blur_x, [%x, add(%y, 1)])
  %sum = add_f(%top, add_f(%mid, %bot))
  %avg = div_f(%sum, 3.0)
  return(%avg)
```

### Phase 2: Schedule representation

Schedules are separate ops applied to `func` results. They don't change the algorithm —
they control how the lowering emits loops.

```dgen
from builtin import Index, String

op split<axis: Index, factor: Index>(func)
op tile<x_factor: Index, y_factor: Index>(func)
op reorder<order: List<Index>>(func)
op fuse<axis_a: Index, axis_b: Index>(func)
op vectorize<axis: Index>(func)
op parallel<axis: Index>(func)
op unroll<axis: Index, factor: Index>(func)

op compute_at<producer: String, consumer: String, axis: Index>()
op store_at<producer: String, consumer: String, axis: Index>()
op compute_root<func_name: String>()
```

All schedule parameters are `__params__` (angle-bracket syntax). When they are stage-0
constants, the schedule resolves at compile time. When stage-1, the staging system
generates a JIT callback.

#### Example: scheduled blur

```
// Schedule: tile blur_y, compute blur_x inside blur_y's x loop
halide.tile<256, 32>(%blur_y)
halide.vectorize<0, 8>(%blur_y)    // vectorize inner x by 8
halide.parallel<1>(%blur_y)        // parallelize outer y
halide.compute_at<"blur_x", "blur_y", 0>()  // fuse blur_x into blur_y's x loop
```

### Phase 3: Schedule-aware lowering

The `HalideToAffine` pass consumes func + schedule ops and emits affine loops:

1. **Collect funcs and schedules**: Walk the module, build a map from func names to
   their schedule directives.
2. **Apply transforms top-down**: Starting from the output func, apply schedule
   directives to determine loop structure:
   - `split` → split one `ForOp` into outer + inner
   - `tile` → split both x and y, reorder to `yo, xo, yi, xi`
   - `reorder` → permute the loop nesting
   - `compute_at` → inline the producer's computation inside the consumer's loop,
     allocating a line buffer sized to the consumer's footprint at that loop level
   - `compute_root` → compute the entire producer before the consumer (no fusion)
3. **Emit affine ops**: `ForOp` nests with `LoadOp`/`StoreOp` and intermediate `AllocOp`
   buffers.

### Phase 4: Loop transformations on affine IR

These are reusable passes independent of the Halide dialect:

#### 4a: Loop identity

Add axis metadata to `ForOp` so transformation passes can reference specific loops:

```dgen
op for<lo: Index, hi: Index, axis: Index>():
  block body
  has trait HasSingleBlock
```

Or: maintain a side-table mapping ForOps to axis identifiers in the pass state.

#### 4b: Affine index arithmetic

Add `AffineApplyOp` for index expressions inside loop bodies:

```dgen
op affine_apply<coefficients: List<Index>>(indices: List<Index>) -> Index
```

Represents `c0 * i0 + c1 * i1 + ... + offset`. Needed for tiled index computation
(`xo * tile_factor + xi`).

#### 4c: Split pass

Given `for i in [0, N)`, produce:

```
for io in [0, N / factor):
  for ii in [0, factor):
    // replace all uses of i with (io * factor + ii)
```

Requires: walking all ops in the loop body, rewriting index references.

#### 4d: Tile pass

Split on two dimensions, then reorder to `[yo, xo, yi, xi]`.

### Phase 5: Bound inference

When `compute_at` fuses a producer into a consumer's loop, the producer needs enough
of a buffer to cover the consumer's access pattern at that loop level.

For a 3x3 stencil accessing `[x-1..x+1, y-1..y+1]` inside a loop over `[xo*256..(xo+1)*256]`:
- Required x range: `[xo*256 - 1, (xo+1)*256 + 1)` — width 258
- Required y range: just the current `y` ± 1 — height 3 (sliding window)

Implementation: analyze `halide.call` index expressions to determine the access footprint
relative to the consumer's loop variables. Allocate the intermediate buffer accordingly.

### Phase 6: Vectorization (optional, high effort)

#### 6a: LLVM vector types

Extend `llvm.dgen`:

```dgen
type Vec<width: Index, element: Type>:
  data: Nil  // opaque at layout level

op vec_load(ptr: Ptr) -> Vec
op vec_store(value: Vec, ptr: Ptr) -> Nil
op vec_add(lhs: Vec, rhs: Vec) -> Vec
op vec_mul(lhs: Vec, rhs: Vec) -> Vec
op vec_broadcast(scalar: Float) -> Vec
```

#### 6b: Vectorization pass

Replace the innermost loop (after tiling) with vector operations:

```
// Before:
for xi in [0, 8):
  %v = load(%ptr + xi)
  %r = fmul(%v, %c)
  store(%r, %out + xi)

// After:
%v = vec_load(%ptr)
%bc = vec_broadcast(%c)
%r = vec_mul(%v, %bc)
vec_store(%r, %out)
```

#### 6c: LLVM codegen

Emit LLVM vector IR: `<8 x double>`, `load <8 x double>`, vector `fmul`, etc.

### Phase 7: Parallelization (optional, high effort)

Add a threading runtime (either OpenMP via `@llvm.experimental.parallel` intrinsics,
or a minimal C thread pool linked into the JIT). Replace a `ForOp` marked `parallel`
with dispatch to the thread pool.

## Implementation order and effort

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Algorithm representation (halide dialect, func/var/call ops) | 1 week | — |
| 2 | Schedule representation (schedule ops as params) | 1 week | Phase 1 |
| 3 | Schedule-aware lowering (HalideToAffine pass) | 2-3 weeks | Phases 1-2 |
| 4a | Loop identity (axis metadata on ForOp) | 2-3 days | — |
| 4b | Affine index arithmetic (AffineApplyOp) | 3-4 days | — |
| 4c-d | Split and tile passes | 1-2 weeks | Phases 4a-b |
| 5 | Bound inference for compute_at | 1-2 weeks | Phase 3 |
| 6 | Vectorization (types, pass, codegen) | 2-4 weeks | Phase 4 |
| 7 | Parallelization (threading runtime) | 2-4 weeks | Phase 4 |

**Minimum viable demo (Phases 1-3, 4a-b):** ~5-6 weeks. Demonstrates algorithm/schedule
separation, tiling, compute_at fusion, and the staging story (same pipeline, static vs
dynamic schedules).

**Full system (all phases):** ~3-5 months.

## The staging story

The key demo for a blog post:

```python
# Define algorithm once
blur = build_box_blur_pipeline()

# Static schedule: compile-time specialization
static_schedule = Schedule(tile_x=256, tile_y=32, vectorize=8)
exe_static = compiler.compile(blur, schedule=static_schedule)
# → Single optimized binary, no JIT overhead

# Dynamic schedule: runtime specialization
# tile factors are block args, resolved at runtime
exe_dynamic = compiler.compile(blur, schedule=dynamic_schedule)
exe_dynamic.run(image, tile_x=256, tile_y=32)
# → JIT callback fires, compiles specialized code, caches it
exe_dynamic.run(image, tile_x=128, tile_y=64)
# → Different specialization, also cached
```

Same `build_box_blur_pipeline()`. Same `HalideToAffine` pass. Same LLVM codegen.
The staging system handles the difference transparently. This is not possible in
Halide itself, where AOT and JIT are distinct compilation modes with different APIs.

## Risks

- **Bound inference complexity**: Computing tight bounds for stencil access patterns
  is the hardest algorithmic piece. Start with rectangular bounds (conservative
  over-allocation) and tighten later.
- **Index rewriting in split/tile**: Replacing all uses of a loop variable with a
  compound expression (`outer * factor + inner`) requires careful traversal of the
  op graph. The existing `Rewriter.replace_uses` handles op-level replacement but
  may need extension for index expression rewriting.
- **Vectorization tail handling**: When the loop bound isn't a multiple of the vector
  width, a scalar cleanup loop is needed. This is mechanical but fiddly.
