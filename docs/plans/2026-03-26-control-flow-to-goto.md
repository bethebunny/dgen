# ControlFlowToGoto: labels with bodies, flat exit continuation

## Problem

`StructuredToLLVM` is a ~360-line monolithic pass that handles control flow,
memory, and algebra lowering simultaneously using a value_map approach that
rebuilds the entire IR. The pass decomposition goal is three focused passes:
ControlFlowToGoto, MemoryToLLVM, AlgebraToLLVM.

Previous attempts at ControlFlowToGoto failed because:
- The value_map clone approach was fragile and reimplemented IR infrastructure
- The flat label (no body blocks) approach required complex scheduling to assign
  ops to basic blocks, and couldn't naturally express the entry→inner loop path
- Block splitting (moving post-loop ops into an exit label) fought the use-def
  graph model

## Design

The key insight: header and body labels need body blocks (they have phis and
back-edges), but the exit label can be a **marker** with an empty body. Post-loop
code stays in the parent scope and falls through after the exit block in LLVM.

### Input IR

```
%alloc = memory.alloc(...)
%loop  = control_flow.for<0, 4>([]) body(%i):
    %x  = memory.load(%a, [%i])
    %st = memory.store(%x, %alloc, [%i])
%result = chain(%alloc, %loop)
%print  = memory.print_memref(%result)
```

### Output IR

```
%alloc = memory.alloc(...)
%header = goto.label() body<%self>(%iv):
    %cmp   = algebra.less_than(%iv, 4)
    %cond  = goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])
%body = goto.label() body(%j) captures(%self, %alloc):
    %x    = memory.load(%a, [%j])
    %st   = memory.store(%x, %alloc, [%j])
    %next = algebra.add(%j, 1)
    %back = goto.branch<%self>([%next])
    %term = chain(%st, %back)
%exit = goto.label() body():
    ()
%entry_br   = goto.branch<%header>([0])
%after_loop = chain(%exit, %entry_br)
%result     = chain(%alloc, %after_loop)    ← was chain(%alloc, %loop)
%print      = memory.print_memref(%result)
```

The ForOp is replaced by `%after_loop = chain(%exit, %entry_br)` via
`rewriter.replace_uses`. Post-loop ops (`%result`, `%print`) stay in function
scope. LLVM falls through after the exit block, so they emit after exit
naturally. No block splitting needed.

### Why captures work

The body label captures `%self` (for back-edge) and any outer values referenced
by body ops (like `%alloc`). The captures are computed by walking the body
block's ops and collecting values not defined within the block.

### Why the header depends on body and exit

The header's conditional branch targets `%body` and `%exit` (as parameters).
`walk_ops` follows parameters, so the header label's body block reaches
`%body` and `%exit`. This makes the entire loop structure reachable from the
entry branch.

### Nested loops

`Pass._run_block` recurses into nested blocks for unhandled ops. When the outer
ForOp is lowered, the inner ForOp is now inside the body label's body block. The
pass recurses into it and the `@lowering_for(ForOp)` handler fires again. No
special convergence loop needed.

## Implementation

### ControlFlowToGoto pass

A `Pass` subclass with `@lowering_for(control_flow.ForOp)`:

1. Create header, body, exit LabelOps (all with body blocks)
2. Header body: `%self` BlockParameter, IV BlockArgument, comparison, cond_br
3. Body body:
   - New body_iv BlockArgument
   - `Rewriter(for_op.body).replace_uses(old_iv, body_iv)` to remap IV
   - Append back-edge: `chain(original_body_result, goto.branch<%self>([iv+1]))`
   - Captures: computed from body ops' references to outer values
4. Exit body: empty (`result=Nil()`)
5. Entry branch: `goto.branch<%header>([lo])`
6. Replace: `rewriter.replace_uses(for_op, chain(exit, entry_br))`

### Captures computation

```python
def compute_captures(block: Block) -> list[Value]:
    """Values referenced by block ops but not defined within the block."""
    defined = set(block.args) | set(block.parameters) | set(block.ops)
    captures = []
    for op in block.ops:
        for _, v in op.operands:
            if isinstance(v, (Op, BlockArgument, BlockParameter)) and v not in defined:
                if v not in captures:
                    captures.append(v)
    return captures
```

### MemoryToLLVM pass

A `Pass` subclass with `@lowering_for` handlers for each memory op. Uses
`Rewriter.replace_uses`. Walks into label bodies via `Pass._run_block`.

### Pipeline

```
ToyToStructured → ControlFlowToGoto → MemoryToLLVM → [codegen: AlgebraToLLVM + BuiltinToLLVM]
```

### What doesn't change

- goto dialect (LabelOp keeps body block)
- codegen (`_collect_labels` + `claimed` + fall-through works as-is)
- AlgebraToLLVM (already done, already on main)
- BuiltinToLLVM IfOp lowering (same pattern: then/else labels with bodies,
  merge label with empty body, flat continuation — but this is a follow-up)

## Testing

- All existing end-to-end, staging, and codegen tests should pass
- The decomposed pipeline (`ControlFlowToGoto + MemoryToLLVM`) replaces
  `StructuredToLLVM` as the default in `toy_compiler`
