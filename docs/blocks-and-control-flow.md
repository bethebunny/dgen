# Blocks and Control Flow

## Closed-Block Invariant

**Blocks are closed.** An op inside a block may only reference values defined in that same block: local ops, block arguments, block parameters, or declared captures. Any value from an enclosing scope must appear in the block's `captures` list.

```
# Valid: %ptr declared as capture
%body = goto.label([]) body(%i: Index) captures(%ptr):
    %val = memory.load(%ptr)
    %one = 1
    %next = algebra.add(%i, %one)

# Invalid: %ptr not in scope
%body = goto.label([]) body(%i: Index):
    %val = memory.load(%ptr)   ← closed-block violation
```

The verifier (`dgen/ir/verification.py`) checks this invariant: `verify_closed_blocks` walks every reachable block and confirms that all `BlockArgument`/`BlockParameter` values found during traversal belong to the current block.

## Three Kinds of Block Inputs

### args -- Runtime values

`BlockArgument` values are runtime inputs. Branch ops pass values to target block args positionally. Codegen emits phi nodes for these.

```python
loop_iv = BlockArgument(name="i", type=Index())
block = Block(result=..., args=[loop_iv])
```

In IR: `body(%i: index.Index)`

### parameters -- Compile-time structural values

`BlockParameter` values are bound once at IR construction time. They encode structural references like `%self` (back-edges) and `%exit` (loop exit targets). They do not vary at runtime.

```python
self_param = BlockParameter(name="self", type=goto.Label())
exit_param = BlockParameter(name="exit", type=goto.Label())
block = Block(result=..., parameters=[self_param, exit_param])
```

In IR: `body<%self: goto.Label, %exit: goto.Label>(%i: index.Index)`

### captures -- Outer-scope dependencies

Captures are outer-scope values referenced directly. No phi, no copy -- just a declared dependency. The walk stops at captures (they are boundaries in `transitive_dependencies`).

```python
block = Block(result=..., captures=[outer_ptr, loop_bound])
```

In IR: `body(%i: index.Index) captures(%ptr, %bound)`

## Sea-of-Nodes Within Blocks

Within a single block, the use-def graph **is** the execution model. There is no implicit ordering between ops that are not connected by data dependencies.

- **Pure ops** may execute in any order
- **Side-effecting ops** must be chained to the block result via `ChainOp`
- **All ops must be reachable** from `block.result` via `block.ops` -- unreachable ops are dead

### ChainOp

`ChainOp(lhs, rhs)` returns `lhs`'s value with a data dependency on `rhs`. This injects side effects into the use-def graph:

```
%store = memory.store(%val, %ptr)    # side effect, returns Nil
%result = chain(%useful_value, %store)  # %store executes, %useful_value passes through
```

The chain spine defines the schedule for side effects within a block.

## transitive_dependencies

`transitive_dependencies(root, stop)` walks the use-def graph from `root` in topological order. The dependency edges it follows are:

- Operands (runtime SSA values)
- Parameters (compile-time values)
- Types (the `op.type` value)
- Block captures (outer-scope deps)
- Block argument/parameter types

It does **not** descend into nested block bodies. Each block is its own walk scope; captures are boundaries -- the walk visits them but doesn't traverse past them.

### block.ops

`block.ops` is derived, not stored. It walks `transitive_dependencies(block.result, stop=block.captures)` and yields only `Op` instances:

```python
@property
def ops(self) -> Iterator[Op]:
    return (v for v in self.values if isinstance(v, Op))
```

## goto Dialect: Unstructured Control Flow

The `goto` dialect provides labels, branches, and conditional branches for lowered control flow.

### Labels as Values

A `goto.label` is an expression block -- it runs when control reaches it in use-def order. Its `initial_arguments` provide first-iteration values for its block args.

A `goto.region` is similar but has fall-through semantics: it executes inline at its use-def position. Every region body has exactly two parameters: `(%self, %exit)`.

### %self for Back-Edges

`%self` is a `BlockParameter` that enables back-edges without creating use-def cycles. A `goto.branch<%self>([next_values])` branches back to the enclosing region/label header:

```
%header = goto.region([%lo]) header<%self: goto.Label, %exit: goto.Label>(%iv: index.Index):
    %cmp = algebra.less_than(%iv, %hi)
    %body = goto.label([]) body(%jv: index.Index) captures(%self):
        %next = algebra.add(%jv, index.Index(1))
        %_ : Nil = goto.branch<%self>([%next])  # back-edge
    %_ : Nil = goto.conditional_branch<%body, %exit>(%cmp, [%iv], [])
```

### %exit for Loop Exits

`%exit` is the second parameter of every region body. Codegen emits it as a fall-through LLVM basic block after the region body. Branching to `%exit` exits the loop.

## control_flow Dialect: Structured Sugar

Higher-level ops that lower to `goto` (loops) or are emitted directly by codegen (if/else).

### ForOp / WhileOp

`control_flow.for` and `control_flow.while` are lowered to `goto.region` + `goto.label` by the `ControlFlowToGoto` pass. The lowering creates:

- A **region** as the loop header (falls through from use-def position)
- A **label** as the loop body (entered only via conditional branch)
- `%self` parameter for the back-edge
- `%exit` parameter for the loop exit
- `ChainOp` to ensure the loop increment runs after the body

### IfOp

`control_flow.if` is lowered to `goto.region` with two `goto.label` blocks (then/else). The region's block arg serves as the merge phi:

```
goto.region([]) if<%self, %exit>(%result: Type):
    goto.label([]) if_then(...) captures(%self):
        <then body>
        goto.branch<%self>([then_result])
    goto.label([]) if_else(...) captures(%self):
        <else body>
        goto.branch<%self>([else_result])
    goto.conditional_branch<%then, %else>(%cond, [], [])
    chain(%result, %cond_br)
```

Codegen emits this as: entry block (dispatch) -> then/else blocks -> merge block (phi).

> **Not yet implemented:** `func.recursive` for recursive functions. Currently, recursive calls work via callback thunks and global symbol registration, but there is no dedicated structural op for recursion.

## Key Files

| File | Role |
|------|------|
| `dgen/block.py` | `Block`, `BlockArgument`, `BlockParameter` |
| `dgen/ir/traversal.py` | `transitive_dependencies`, `all_values`, `inline_block` |
| `dgen/passes/control_flow_to_goto.py` | `ControlFlowToGoto` pass |
| `dgen/ir/verification.py` | `verify_closed_blocks`, `verify_dag` |
| `dgen/dialects/goto.dgen` | goto dialect spec |
| `dgen/dialects/control_flow.dgen` | control_flow dialect spec |
