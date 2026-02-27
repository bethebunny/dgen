"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import dgen
from dgen import codegen
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.type import Memory
from dgen.value import Constant


def _trace_dependencies(target: dgen.Value, func: builtin.FuncOp) -> list[dgen.Op]:
    """Backward-walk from target, return all needed ops in topological order."""
    needed: set[int] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if id(val) in needed:
            continue
        needed.add(id(val))
        if isinstance(val, dgen.Op):
            for _, operand in val.operands:
                worklist.append(operand)
    return [op for op in func.body.ops if id(op) in needed]


def _is_stage0_evaluable(target: dgen.Value) -> bool:
    """True if target's dependencies can be evaluated without runtime values.

    Returns False if any value in the dependency tree is a BlockArgument
    (function parameter), meaning it depends on runtime input.
    """
    visited: set[int] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if id(val) in visited:
            continue
        visited.add(id(val))
        if isinstance(val, BlockArgument):
            return False
        if isinstance(val, dgen.Op):
            for _, operand in val.operands:
                worklist.append(operand)
    return True


def _make_memories(block_args: list[BlockArgument], python_args: list) -> list[Memory]:
    """Convert Python args to Memory objects using block argument types."""
    return [
        Memory.from_value(param.type, arg)
        for arg, param in zip(python_args, block_args)
    ]


def _jit_evaluate(
    subgraph: list[dgen.Op],
    target: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
    *,
    block_args: list[BlockArgument] | None = None,
    args: list | None = None,
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=ops, args=list(block_args or [])),
        type=builtin.Function(result=target.type),
    )
    module = builtin.Module(functions=[func])
    lowered = lower(module)
    exe = codegen.compile(lowered)
    memories: list[Memory] = []
    if block_args and args:
        memories = _make_memories(block_args, args)
    return exe.run(*memories)


def _resolve_comptime_field(
    func: builtin.FuncOp,
    op: dgen.Op,
    field_name: str,
    value: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
) -> None:
    """Resolve a single Constant field: JIT the dependency subgraph, patch with ConstantOp."""
    subgraph = _trace_dependencies(value, func)
    stage1 = not _is_stage0_evaluable(value)
    result = _jit_evaluate(
        subgraph,
        value,
        lower,
        block_args=func.body.args if stage1 else None,
        args=args,
    )
    if stage1:
        result = int(result)  # type: ignore[arg-type]
        subgraph_ids = {id(o) for o in subgraph}
        func.body.ops[:] = [o for o in func.body.ops if id(o) not in subgraph_ids]
    const_op = builtin.ConstantOp(value=result, type=value.type)
    idx = func.body.ops.index(op)
    func.body.ops.insert(idx, const_op)
    setattr(op, field_name, const_op)


def _resolve_constant_ops(func: builtin.FuncOp) -> None:
    """Replace ops that implement resolve_constant() with ConstantOps.

    After shape inference runs, ops like DimSizeOp may be resolvable to
    constants because their input types are now concrete. This function
    finds such ops and replaces them, patching any Constant field references.
    """
    for i, op in enumerate(func.body.ops):
        resolver = getattr(op, "resolve_constant", None)
        if resolver is None:
            continue
        val = resolver()
        if val is None:
            continue
        const = builtin.ConstantOp(value=val, type=op.type)
        func.body.ops[i] = const
        for other in func.body.ops:
            for param, _ in other.__params__:
                if getattr(other, param) is op:
                    setattr(other, param, const)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_all_comptime(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
) -> builtin.Module:
    """Deepcopy + resolve all Constant fields. Returns pre-lowered module."""
    module = deepcopy(module)
    changed = True
    while changed:
        changed = False
        func = module.functions[0]
        for op in list(func.body.ops):
            for field_name, _ in op.__params__:
                value = getattr(op, field_name)
                if not isinstance(value, dgen.Value):
                    continue  # already a literal constant, no resolution needed
                if isinstance(value, Constant):
                    continue  # already a resolved Constant (ConstantOp or inline)
                _resolve_comptime_field(func, op, field_name, value, lower, args)
                module = infer(module)
                _resolve_constant_ops(module.functions[0])
                changed = True
                break
            if changed:
                break
    return module


def compile_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
) -> codegen.Executable:
    """Stage-resolve, infer, lower, and compile to an Executable."""
    resolved = _resolve_all_comptime(module, infer=infer, lower=lower, args=args)
    func = resolved.functions[0]
    typed = infer(resolved)
    lowered = lower(typed)
    exe = codegen.compile(lowered)
    # Override with pre-lowered types so Memory.from_value works correctly
    if func.body.args:
        exe.input_types = [arg.type for arg in func.body.args]
    return exe


def compile_and_run_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: list | None = None,
) -> object:
    """Full staged compilation pipeline: resolve, compile, and run."""
    exe = compile_staged(module, infer=infer, lower=lower, args=args)
    memories = [Memory.from_value(t, a) for t, a in zip(exe.input_types, args or [])]
    return exe.run(*memories)
