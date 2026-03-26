# Plan: Low-Level Dialect Decomposition

## Context

The current dialect structure conflates several concerns. The `affine` dialect
contains memory ops, arithmetic, structured loops, and types that belong at
different abstraction levels. The `builtin` dialect contains function ops, control
flow, index arithmetic, and IR sugar. The `llvm` dialect mixes target-specific
pointer ops with generic control flow and arithmetic.

This plan decomposes the IR into focused dialects with clear responsibilities,
following MLIR's approach but tailored to dgen's design.

## Proposed Dialects

### `goto` — Unstructured Control Flow

Labels and branches. The CFG primitives that structured control flow lowers to.

```dgen
from number import Boolean

type Label:
    data: Nil

op label() -> Label:
    block body
op branch<target: Label>(arguments: Span) -> Nil
op conditional_branch<true_target: Label, false_target: Label>(
    condition: Boolean, true_arguments: Span, false_arguments: Span
) -> Nil
```

**Currently in:** `llvm` (LabelOp, BrOp, CondBrOp)

### `control_flow` — Structured Control Flow

Loops and conditionals that preserve structure for analysis and optimization.

```dgen
from index import Index

op for<lower_bound: Index, upper_bound: Index>(initial_arguments: Span) -> Nil:
    block body

op if(condition: Index, then_arguments: Span, else_arguments: Span) -> Type:
    block then_body
    block else_body
```

**Currently in:** `affine` (ForOp), `builtin` (IfOp)

### `function` — Functions and Calls

```dgen
type Function<result: Type>:
    layout Void

op function<result: Type>() -> Nil:
    block body

op recursive<result: Type>() -> Function:
    block body

op call<callee: Function>(arguments: Span) -> Type
```

`function.recursive` introduces `%self` as the body block's first parameter,
typed `Function<result>`. The body may call `%self` to recurse. `%self` is a
block parameter (a leaf in the DAG), so the use-def graph remains acyclic —
same mechanism as `goto.label`'s `%self` for loop back-edges. See
`docs/block-scoping.md` §3.1 for the full design.

**Currently in:** `builtin` (FunctionOp, CallOp). `recursive` is new — currently
recursive functions violate the DAG property (tracked in TODO.md).

### `algebra` — Operations on Algebraic Structures

Polymorphic operations dispatched by the type's algebraic traits. Inspired by
Lean's Mathlib: types declare which algebraic structures they implement, and
`algebra` ops are defined over those structures.

#### Trait hierarchy

```
Magma              — closed binary operation (op: add or mul)
  Semigroup        — associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)
    Monoid         — identity element: a ∘ e = a
      Group        — inverse: a ∘ a⁻¹ = e
        AbelianGroup — commutativity: a ∘ b = b ∘ a
    CommutativeSemigroup — commutativity
      CommutativeMonoid  — identity + commutativity

Semiring           — (CommutativeMonoid under add) + (Monoid under mul) + distributivity
  Ring             — Semiring where add forms an AbelianGroup
    CommutativeRing — Ring with commutative mul
      Field        — CommutativeRing where every nonzero element has mul inverse

TotalOrder         — reflexive, antisymmetric, transitive, total
  OrderedSemiring  — Semiring + TotalOrder compatible with both operations

Lattice            — has meet and join
  BoundedLattice   — has top and bottom elements
    ComplementedLattice — has complement
      BooleanAlgebra    — distributive + complemented
```

This is not the full Mathlib hierarchy — it's the subset useful for optimization.
A `CommutativeMonoid` tells the optimizer it can reorder additions. A `Ring` tells
it `a * (b + c) = a * b + a * c`. A `TotalOrder` says comparisons are meaningful.
A `BooleanAlgebra` says bitwise operations are available and that
`(symmetric_difference, meet)` forms a `CommutativeRing` (the GF(2)^n structure).

#### Ops

```dgen
// Additive operations (require AddMagma, AddSemigroup, AddMonoid, etc.)
op add(left: Type, right: Type) -> Type
op negate(input: Type) -> Type           // requires AddGroup
op subtract(left: Type, right: Type) -> Type  // sugar: add(left, negate(right))

// Multiplicative operations
op multiply(left: Type, right: Type) -> Type
op reciprocal(input: Type) -> Type       // requires MulGroup (Field)
op divide(left: Type, right: Type) -> Type    // sugar: multiply(left, reciprocal(right))

// Lattice / bitwise operations (require Lattice, BooleanAlgebra, etc.)
op meet(left: Type, right: Type) -> Type              // AND; requires Lattice
op join(left: Type, right: Type) -> Type              // OR; requires Lattice
op complement(input: Type) -> Type                    // NOT; requires ComplementedLattice
op symmetric_difference(left: Type, right: Type) -> Type  // XOR; requires BooleanAlgebra

// Shifts — not algebraic binary ops; the shift amount is an Index acting on the value
op shift_left(value: Type, amount: Index) -> Type
op shift_right(value: Type, amount: Index) -> Type

// Comparison (requires TotalOrder or Eq)
op equal(left: Type, right: Type) -> Boolean          // requires Eq
op not_equal(left: Type, right: Type) -> Boolean      // requires Eq
op less_than(left: Type, right: Type) -> Boolean      // requires TotalOrder
op less_equal(left: Type, right: Type) -> Boolean     // requires TotalOrder
op greater_than(left: Type, right: Type) -> Boolean   // requires TotalOrder
op greater_equal(left: Type, right: Type) -> Boolean  // requires TotalOrder

// Conversions
op cast(input) -> Type
```

`meet` and `join` are the lattice names for AND and OR. `symmetric_difference`
is XOR. Using algebraic names makes the optimizer's reasoning explicit: `meet`
distributes over `join`, `symmetric_difference` is self-inverse, and
`(symmetric_difference, meet)` forms a CommutativeRing — identities like
`a ^ a = 0` and `a & (b ^ c) = (a & b) ^ (a & c)` follow from the ring axioms.

Shifts have a single `shift_right` op because the signed/unsigned type split
determines the semantics: `SignedInteger` gets arithmetic right shift (fills
with sign bit), `UnsignedInteger` gets logical right shift (fills with zero).
The type carries the semantics, not the op name.

Each comparison op is a separate op rather than a single `compare` with a string
predicate. This makes comparison semantics visible in the IR structure —
`algebra.less_than` on `SignedInteger` lowers to `icmp slt`, on
`UnsignedInteger` to `icmp ult`, without inspecting a string parameter to decide.

#### Type registrations and numeric subtleties

The correct algebraic classification of machine numeric types is not obvious and
we should learn from Mathlib's approach of modeling them precisely rather than
pretending they satisfy properties they don't.

**Signed integers** (`Integer<64>`, etc.) form a `CommutativeRing` under wrapping
arithmetic — addition and multiplication are associative, commutative, and
distributive, with `0` and `1` as identities. They do NOT form an ordered ring
in the usual sense because wrapping breaks order compatibility (adding a positive
number can produce a smaller result). They have a `TotalOrder` under signed
comparison, but it is not compatible with the ring operations.

**Unsigned integers** are a different type with different algebraic structure.
Under wrapping arithmetic they also form a `CommutativeRing`, but their ordering
(unsigned comparison) is distinct from signed ordering. The optimizer must not
confuse signed and unsigned comparisons — `icmp slt` vs `icmp ult` in LLVM
exist precisely because of this distinction. dgen should model them as separate
types: `number.SignedInteger<bits>` and `number.UnsignedInteger<bits>`, each
with their own trait registrations.

**Floating-point numbers** are NOT a field, NOT a ring, and strictly speaking
not even a semigroup under addition — IEEE 754 addition is not associative:
`(a + b) + c ≠ a + (b + c)` in general. Mathlib models floats as having
approximate algebraic structure, not exact. For dgen:

- By default, `Float64` implements only `CommutativeMagma` under addition and
  multiplication (closed, commutative, but NOT associative). This means the
  optimizer cannot reassociate `(a + b) + c` to `a + (b + c)`.
- Under a `FastMath` mode (analogous to `-ffast-math`), `Float64` additionally
  implements `CommutativeMonoid` under both operations, enabling reassociation,
  identity elimination (`a + 0.0 → a`), and other optimizations that depend on
  associativity. This is a deliberate, user-requested relaxation, not a default.
- `Float64` implements `TotalOrder` under comparison (with the caveat that NaN
  breaks totality — the optimizer should be aware of this).

**Boolean** (`number.Boolean`) implements `CommutativeRing` where addition is
XOR and multiplication is AND, with identities `false` and `true` respectively.
This is the two-element field GF(2).

**Index** implements `CommutativeRing` and `TotalOrder`. Its width is
target-dependent (see the `index` dialect section).

**String** implements `AddMonoid` (concatenation is associative with empty string
as identity) but NOT `AddCommutativeMonoid` — `"ab" + "cd" ≠ "cd" + "ab"`.

```
number.SignedInteger<64>    implements CommutativeRing, TotalOrder, BooleanAlgebra
number.UnsignedInteger<64>  implements CommutativeRing, TotalOrder, BooleanAlgebra
number.Float64              implements CommutativeMagma (default)
                            implements CommutativeMonoid (under FastMath)
number.Boolean              implements CommutativeRing (GF(2)), BooleanAlgebra
index.Index                 implements CommutativeRing, TotalOrder
builtin.String              implements AddMonoid
```

`algebra.add` on two `String` values is concatenation — the `AddMonoid` trait
supplies the identity (empty string) and associativity. `algebra.add` on two
`Float64` values is floating-point addition — but because `Float64` only implements
`CommutativeMagma` by default, the optimizer knows it can commute `a + b` to
`b + a` but cannot reassociate. Same op, different algebraic structure, different
optimizations available.

#### What this enables

- **Commutativity-aware CSE.** If `a + b` and `b + a` appear in the same block,
  and the type implements `CommutativeMagma` under addition, they can be unified.
  This is valid even for `Float64` (IEEE 754 addition is commutative).
- **Reassociation.** `(a + b) + c` can be rewritten to `a + (b + c)` only if the
  type implements `Semigroup` under addition. For `Float64` this requires `FastMath`
  mode. For `SignedInteger` this is always valid (wrapping is associative).
- **Distributivity.** `a * (b + c)` → `a * b + a * c` only if the type implements
  `Semiring`. Valid for integers, not for floats by default.
- **Strength reduction.** `a * 2` → `a + a` if the type implements `Semiring`
  (requires distributivity to prove equivalence).
- **Identity elimination.** `a + 0` → `a` if the type implements `Monoid`.
  Valid for integers. For floats, only under `FastMath` (because `−0.0 + 0.0 =
  +0.0 ≠ −0.0` in IEEE 754).
- **Signed vs unsigned awareness.** The type system prevents mixing signed and
  unsigned operations. `algebra.compare<"less_than">` on `SignedInteger` lowers
  to `icmp slt`; on `UnsignedInteger` it lowers to `icmp ult`.

The lowering to `llvm` resolves the polymorphism: `algebra.add` on `Float64` →
`llvm.fadd`, on `SignedInteger<64>` → `llvm.add`, on `String` → a runtime
concatenation call.

### `number` — Numeric Types

The type definitions for numeric values.

```dgen
type Boolean:
    data: Byte  // full byte, not a single bit — MLIR learned this the hard way

type SignedInteger<bits: Index>:
    data: Index

type UnsignedInteger<bits: Index>:
    data: Index

type Float64:
    data: F64
```

`Boolean` uses a full byte layout, not a single bit. Sub-byte types create
pervasive complexity in memory layout, ABI, and vectorization. MLIR's `i1` type
was a persistent source of bugs and special cases; a byte-sized boolean avoids
all of them.

Signed and unsigned integers are distinct types because their algebraic
structures differ — they have different orderings (signed vs unsigned comparison)
and different semantics for division and right-shift. Keeping them separate
prevents the class of bugs where signed and unsigned operations are silently
mixed (a problem LLVM addresses with `slt`/`ult`, `sdiv`/`udiv` op variants,
but that dgen can prevent at the type level).

**Currently in:** `builtin` (F64, partially), `llvm` (Int, Float)

`Boolean` is new. Currently `llvm.Int<1>` serves this role, leaking the LLVM
representation into higher-level ops. `number.Boolean` is the result type for
comparisons and the input type for branches.

### `memory` — Typed Memory Buffers

Allocation, deallocation, and typed access to multi-dimensional memory.

```dgen
from index import Index
from builtin import Array, Pointer

type Shape<rank: Index>:
    dimensions: Array<Index, rank>

type Reference<shape: Shape, element_type: Type = number.Float64>:
    data: Pointer<Nil>

op allocate(shape: Shape) -> Reference
op deallocate(input: Reference) -> Nil
op load(source: Reference, indices: Index) -> Type
op store(value, destination: Reference, indices: Index) -> Nil
```

**Currently in:** `affine` (Shape, Reference, AllocOp, DeallocOp, LoadOp, StoreOp)

### `index` — Index Type

The `Index` type, which has target-dependent width (currently fixed but may
vary when targeting different platforms).

```dgen
type Index:
    layout Int
```

`Index` implements `CommutativeRing` and `TotalOrder`, so it gets `algebra.add`,
`algebra.multiply`, `algebra.less_than`, etc. for free — no dedicated index
arithmetic ops needed. The current `builtin.add_index`, `builtin.subtract_index`,
and `builtin.equal_index` are replaced by `algebra.add`, `algebra.subtract`, and
`algebra.equal` on `Index` values.

The `index` dialect exists as a separate home for the type because its width is
target-dependent — a property that may affect lowering decisions (e.g. whether
to use 32-bit or 64-bit arithmetic for indices on a given target).

### `affine` — Polyhedral Analysis (future)

Not implemented now. When added, this dialect would sit above `memory` +
`control_flow` and add affine maps to loop bounds and memory access indices,
enabling dependence analysis, loop fusion, tiling, and interchange. The current
`affine` dialect's ops would already have moved to `memory` and `control_flow`;
the new `affine` dialect would wrap them with analysis-friendly structure.

### Ops that stay in their current dialect

**`builtin`:** `ConstantOp`, `PackOp`, `ChainOp`, basic
types (`Nil`, `String`, `Byte`, `Array`, `Pointer`, `Span`, `Tuple`,
`TypeTag`). These are dialect-independent infrastructure.

**`llvm`:** Target-specific ops only — `alloca`, `gep`, `load` (untyped pointer
load), `store` (untyped pointer store), `call` (extern call by string name),
`Ptr` type. Everything else moves up.

**`toy`:** Unchanged. Tensor ops, transpose, reshape, print, etc. These lower to
`memory` + `control_flow` + `algebra`.

**`actor`:** Unchanged. Pipeline and actor ops lower to `control_flow` + `memory`.

## Migration Map

| Current op/type | New location |
|----------------|--------------|
| `builtin.FunctionOp` | `function.function` |
| `builtin.CallOp` | `function.call` |
| `builtin.Function` type | `function.Function` |
| `builtin.IfOp` | `control_flow.if` |
| `builtin.Index` type | `index.Index` |
| `builtin.F64` type | stays `builtin` (layout type) |
| `builtin.add_index` | `index.add` |
| `builtin.subtract_index` | `index.subtract` |
| `builtin.equal_index` | `index.equal` |
| `builtin.ChainOp` | stays `builtin` |
| `builtin.ConstantOp` | stays `builtin` |
| `builtin.PackOp` | stays `builtin` |
| `affine.ForOp` | `control_flow.for` |
| `affine.AllocOp` | `memory.allocate` |
| `affine.DeallocOp` | `memory.deallocate` |
| `affine.LoadOp` | `memory.load` |
| `affine.StoreOp` | `memory.store` |
| `affine.Shape` type | `memory.Shape` |
| `affine.Reference` type | `memory.Reference` |
| `affine.MulFOp` | `algebra.multiply` on Float64 |
| `affine.AddFOp` | `algebra.add` on Float64 |
| `affine.PrintMemrefOp` | toy-specific or `io.print` |
| `llvm.LabelOp` | `goto.label` |
| `llvm.BrOp` | `goto.branch` |
| `llvm.CondBrOp` | `goto.conditional_branch` |
| `llvm.Int` type | `number.SignedInteger` / `number.UnsignedInteger` |
| `llvm.Float` type | `number.Float64` (was unparameterized, now named explicitly) |
| `llvm.Label` type | `goto.Label` |
| `llvm.FaddOp` | lowering target for `algebra.add` |
| `llvm.FmulOp` | lowering target for `algebra.multiply` |
| `llvm.AddOp` | lowering target for `algebra.add` |
| `llvm.SubOp` | lowering target for `algebra.subtract` |
| `llvm.MulOp` | lowering target for `algebra.multiply` |
| `llvm.IcmpOp` | lowering target for `algebra.less_than`, `algebra.equal`, etc. |
| `llvm.FcmpOp` | lowering target for `algebra.less_than`, `algebra.equal`, etc. |
| `llvm.ZextOp` | lowering target for `algebra.cast` |
| (new) `llvm.and` | lowering target for `algebra.meet` |
| (new) `llvm.or` | lowering target for `algebra.join` |
| (new) `llvm.xor` | lowering target for `algebra.symmetric_difference` |
| (new) `llvm.shl` | lowering target for `algebra.shift_left` |
| (new) `llvm.ashr` / `llvm.lshr` | lowering target for `algebra.shift_right` |
| `llvm.Void` type | removed (use `builtin.Nil`) |
| `llvm.AllocaOp` | stays `llvm` |
| `llvm.GepOp` | stays `llvm` |
| `llvm.LoadOp` | stays `llvm` (untyped pointer load) |
| `llvm.StoreOp` | stays `llvm` (untyped pointer store) |
| `llvm.CallOp` | stays `llvm` (extern by string) |

## New Lowering Pipeline

```
toy
  → memory + control_flow + algebra     (pass 1: toy_to_structured)
    → goto + memory + algebra           (pass 2: control_flow_to_goto)
      → goto + llvm + algebra           (pass 3: memory_to_pointers)
        → goto + llvm                   (pass 4: algebra_to_llvm)
          → LLVM IR text → JIT          (codegen)
```

Each pass lowers one dialect while leaving others in place. `algebra.add` and
`memory.load` survive through pass 2 (control flow lowering) unchanged — they
just end up inside `goto.label` bodies. This is the key to the decomposition:
the control flow pass doesn't need to know anything about memory or arithmetic.

### Pass 1: toy → structured (existing `toy_to_affine`, renamed)

No real change in logic. `toy.transpose` → nested `control_flow.for` +
`memory.load` + `memory.store` with swapped indices. `toy.mul` → nested for +
load + `algebra.multiply` + store. Same lowering patterns, cleaner op names.

### Pass 2: control_flow → goto (the hard part, isolated)

The only pass that deals with `%self`, captures, and label/branch structure. It
takes a `control_flow.for` whose body contains arbitrary ops (`memory.load`,
`algebra.add`, etc.) and wraps them in the `goto.label` / `goto.branch`
structure with header, body, and exit blocks.

Crucially, this pass does NOT touch the ops inside the loop body. A
`memory.load` stays as `memory.load` — it just lives inside a `goto.label`
body block now. The pass only handles `control_flow.for` and `control_flow.if`.

What this pass does NOT need (compared to current `affine_to_llvm.py`):
- No `value_map` for arithmetic ops
- No `alloc_shapes` for index linearization
- No `_lower_load`, `_lower_store`, `_lower_print`, `_lower_nonzero_count`
- No type-directed arithmetic dispatch

What it DOES need:
- `%self` block parameters on loop headers
- Captures for outer-scope values referenced by body/exit blocks
- Loop counter for fresh label names

### Pass 3: memory → pointers (linearization, allocation)

Runs after control flow is lowered. Walks into `goto.label` bodies and replaces
memory ops with raw pointer operations:

- `memory.allocate(shape)` → `llvm.call<"malloc">` with size computed from shape
- `memory.load(reference, [%i, %j])` → `llvm.gep` + `llvm.load` with index
  linearization: `%offset = %i * stride + %j`
- `memory.store(value, reference, [%i, %j])` → `llvm.gep` + `llvm.store`
- `memory.deallocate` → no-op (or `llvm.call<"free">`)

This pass needs shape information for linearization but knows nothing about
control flow structure. It is a straightforward op-by-op rewrite — each memory
op maps to a fixed pattern of LLVM ops. No state tracking beyond the current
op's shape.

### Pass 4: algebra → llvm (type-directed dispatch)

Resolves polymorphic algebra ops to concrete LLVM arithmetic:

- `algebra.add` on `Float64` → `llvm.fadd`
- `algebra.multiply` on `Float64` → `llvm.fmul`
- `algebra.add` on `SignedInteger<64>` → `llvm.add`
- `algebra.subtract` on `SignedInteger<64>` → `llvm.sub`
- `algebra.less_than` on `SignedInteger<64>` → `llvm.icmp<"slt">`
- `algebra.less_than` on `UnsignedInteger<64>` → `llvm.icmp<"ult">`
- `algebra.meet` on `SignedInteger<64>` → `llvm.and`
- `algebra.shift_left` → `llvm.shl`

This is a stateless pattern match — inspect the operand type, emit the
corresponding LLVM op. No context, no shape tracking, no control flow awareness.

### Pass ordering

Pass 2 (control flow) must run first — it creates the label/branch structure
that subsequent passes operate inside. Passes 3 and 4 (memory and algebra) are
independent of each other and can run in either order. They operate on different
op types and don't interact.

### How this simplifies the current code

The current `affine_to_llvm.py` is ~350 lines handling 15+ op types, maintaining
`value_map`, `alloc_shapes`, `_header_selfs`, and `_seen` simultaneously. The
`_lower_for` method builds label structures while also resolving memory ops and
arithmetic — because there's only one intermediate dialect for all of it.

With the decomposition:

- **Pass 2** (~100 lines): Only `ForOp` and `IfOp`. Needs captures and `%self`
  but nothing else.
- **Pass 3** (~80 lines): Only memory ops. Needs shapes for linearization but
  no control flow logic.
- **Pass 4** (~30 lines): A type switch. Stateless.

Each pass is independently understandable, testable, and modifiable. A change to
linearization strategy doesn't touch control flow. A new arithmetic type doesn't
touch memory layout. The concerns that are currently entangled in one 350-line
pass become three focused passes that can evolve independently.

### What changes in dgen core

Nothing fundamental. The pass infrastructure already supports multi-dialect IR
and partial lowering (`allow_unregistered_ops = True`). Each pass lowers one
dialect while leaving others in place.

The codegen emits `goto` ops directly as LLVM IR basic blocks (labels, branches,
phi nodes) — `goto` ops don't lower to `llvm` ops, they're emitted by the
codegen alongside `llvm` ops. This matches the current behavior where the codegen
already handles labels and branches separately from arithmetic ops.

## Implementation Order

This is a large refactor. Incremental approach:

1. Define `number` dialect (Boolean, SignedInteger, UnsignedInteger, Float64). Small, no op changes.
2. Define `goto` dialect. Move label/branch ops from `llvm`. Update codegen.
3. Define `memory` dialect. Move Shape, Reference, alloc/load/store from `affine`.
4. Define `index` dialect. Move Index type from `builtin`. Arithmetic comes from `algebra`.
5. Define `control_flow` dialect. Move ForOp from `affine`, IfOp from `builtin`.
6. Define `function` dialect. Move FunctionOp, CallOp from `builtin`. Add `recursive`.
7. Define `algebra` dialect with trait hierarchy. Replace monomorphic arithmetic
   ops. This is the most complex step — the trait system needs design work.
8. Decompose `affine_to_llvm.py` into passes 2, 3, 4. This is the payoff — each
   new pass is a fraction of the complexity.
9. Clean up `affine` (now empty) and slim down `llvm` and `builtin`.

Steps 1–6 are mechanical moves. Step 7 is the design-heavy one. Step 8 is
where the simplification becomes concrete.
