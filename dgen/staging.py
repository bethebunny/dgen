"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from copy import deepcopy
from typing import Sequence

import dgen
from dgen import codegen
from dgen.block import BlockArgument
from dgen.codegen import _ctype, _llvm_type
from dgen.dialects import builtin, llvm
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


def _make_memories(
    block_args: Sequence[BlockArgument], python_args: Sequence
) -> list[Memory]:
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
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = builtin.FuncOp(
        name="main",
        body=dgen.Block(ops=ops, args=list(block_args)),
        type=builtin.Function(result=target.type),
    )
    module = builtin.Module(functions=[func])
    lowered = lower(module)
    exe = codegen.compile(lowered)
    memories = _make_memories(block_args, args)
    raw = exe.run(*memories)

    # For pointer-type results, copy data back while buffers are still alive
    layout = target.type.__layout__
    if _ctype(layout) is ctypes.c_void_p and isinstance(raw, int) and raw != 0:
        buf = (ctypes.c_char * layout.byte_size).from_address(raw)
        return Memory(target.type, bytearray(buf))
    return raw


def _resolve_comptime_field(
    func: builtin.FuncOp,
    op: dgen.Op,
    field_name: str,
    value: dgen.Value,
    lower: Callable[[builtin.Module], builtin.Module],
    *,
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> None:
    """Resolve a single Constant field: JIT the dependency subgraph, patch with ConstantOp."""
    subgraph = _trace_dependencies(value, func)
    result = _jit_evaluate(
        subgraph,
        value,
        lower,
        block_args=block_args,
        args=args,
    )
    if block_args:
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
                if not _is_stage0_evaluable(value):
                    continue  # stage-1: needs runtime args, skip
                _resolve_comptime_field(func, op, field_name, value, lower)
                module = infer(module)
                _resolve_constant_ops(module.functions[0])
                changed = True
                break
            if changed:
                break
    return module


def _has_unresolved_params(func: builtin.FuncOp) -> bool:
    """Check if any ops have unresolved __params__ fields."""
    for op in func.body.ops:
        for field_name, _ in op.__params__:
            value = getattr(op, field_name)
            if isinstance(value, dgen.Value) and not isinstance(value, Constant):
                return True
    return False


def _raw_to_python(raw: object, ty: dgen.Type) -> object:
    """Convert a raw ctypes callback value to a Python value for ConstantOp."""
    layout = ty.__layout__
    if _ctype(layout) is ctypes.c_void_p:
        assert isinstance(raw, int)
        buf = (ctypes.c_char * layout.byte_size).from_address(raw)
        if isinstance(ty, builtin.String):
            return bytes(buf).decode("utf-8")
        return bytes(buf)
    return raw


def _compile_with_callbacks(
    resolved: builtin.Module,
    func: builtin.FuncOp,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
) -> codegen.Executable:
    """Build a stage-1 thunk that calls a host callback for stage-2 JIT.

    The compiled function passes all its arguments to a host callback.
    The callback resolves __params__ values, then JIT-compiles and executes
    the stage-2 code with the remaining runtime arguments.
    """
    # Identify which block args are used as __params__ vs runtime operands
    param_arg_ids: set[int] = set()
    for op in func.body.ops:
        for field_name, _ in op.__params__:
            value = getattr(op, field_name)
            if isinstance(value, BlockArgument) and not isinstance(value, Constant):
                param_arg_ids.add(id(value))

    param_indices: list[int] = []
    runtime_indices: list[int] = []
    for i, arg in enumerate(func.body.args):
        if id(arg) in param_arg_ids:
            param_indices.append(i)
        else:
            runtime_indices.append(i)

    # Build mapping: param arg name → callback arg index
    param_name_to_cb_idx: dict[str | None, int] = {
        func.body.args[i].name: i for i in param_indices
    }

    # Build stage-2 template: remove param block args, keep ops as-is
    stage2_template = deepcopy(resolved)
    stage2_func = stage2_template.functions[0]
    param_arg_names = set(param_name_to_cb_idx.keys())
    stage2_func.body.args[:] = [
        arg for arg in stage2_func.body.args if arg.name not in param_arg_names
    ]

    # Derive callback LLVM signature from original function
    assert func.name is not None
    callback_name = f"_stage2_{func.name}"
    orig_param_types = [arg.type for arg in func.body.args]
    orig_llvm_types = [_llvm_type(t.__layout__) for t in orig_param_types]
    if isinstance(func.type.result, builtin.Nil):
        ret_llvm = "void"
        result_ctype: type[ctypes._CData] | None = None
    else:
        ret_llvm = _llvm_type(func.type.result.__layout__)
        result_ctype = _ctype(func.type.result.__layout__)
    extern_decl = f"declare {ret_llvm} @{callback_name}({', '.join(orig_llvm_types)})"

    # Build ctypes callback type
    param_ctypes = [_ctype(t.__layout__) for t in orig_param_types]
    cb_type = ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    # Capture template + pipeline in closure
    def _callback(*raw_args: object) -> object:
        template = deepcopy(stage2_template)
        s2_func = template.functions[0]

        # Patch __params__ fields with constants from callback args
        for op in list(s2_func.body.ops):
            for field_name, _ in op.__params__:
                value = getattr(op, field_name)
                if not isinstance(value, BlockArgument):
                    continue
                if isinstance(value, Constant):
                    continue
                if value.name not in param_name_to_cb_idx:
                    continue
                cb_idx = param_name_to_cb_idx[value.name]
                raw = raw_args[cb_idx]
                python_val = _raw_to_python(raw, value.type)
                const = builtin.ConstantOp(value=python_val, type=value.type)
                idx = s2_func.body.ops.index(op)
                s2_func.body.ops.insert(idx, const)
                setattr(op, field_name, const)

        # Compile and run stage-2
        typed = infer(template)
        lowered = lower(typed)
        exe = codegen.compile(lowered)
        runtime_args = [raw_args[i] for i in runtime_indices]
        return exe.run(*runtime_args)

    callback_func = cb_type(_callback)

    # Register callback with llvmlite
    codegen._ensure_initialized()
    import llvmlite.binding as _llvm_binding

    _llvm_binding.add_symbol(
        callback_name,
        ctypes.cast(callback_func, ctypes.c_void_p).value,
    )

    # Build stage-1 thunk: call callback with all original params, return result
    thunk_args = [BlockArgument(name=arg.name, type=arg.type) for arg in func.body.args]
    call_op = llvm.CallOp(
        callee=builtin.string_constant(callback_name),
        args=thunk_args,
        type=func.type.result,
    )
    if isinstance(func.type.result, builtin.Nil):
        ret_op = builtin.ReturnOp()
    else:
        ret_op = builtin.ReturnOp(value=call_op)

    thunk_func = builtin.FuncOp(
        name=func.name,
        body=dgen.Block(ops=[call_op, ret_op], args=thunk_args),
        type=builtin.Function(result=func.type.result),
    )
    thunk_module = builtin.Module(functions=[thunk_func])

    exe = codegen.compile(thunk_module, externs=[extern_decl])
    exe.host_refs.append(callback_func)  # prevent GC
    return exe


def compile_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
) -> codegen.Executable:
    """Stage-resolve, infer, lower, and compile to an Executable.

    If all __params__ are resolvable at compile time (stage-0), compiles directly.
    If some __params__ depend on runtime values, builds a callback-based executable
    that JIT-compiles stage-2 code when runtime values become available.
    """
    resolved = _resolve_all_comptime(module, infer=infer, lower=lower)
    func = resolved.functions[0]

    if _has_unresolved_params(func):
        return _compile_with_callbacks(resolved, func, infer=infer, lower=lower)

    typed = infer(resolved)
    lowered = lower(typed)
    return codegen.compile(lowered)


def compile_and_run_staged(
    module: builtin.Module,
    *,
    infer: Callable[[builtin.Module], builtin.Module],
    lower: Callable[[builtin.Module], builtin.Module],
    args: Sequence = (),
) -> object:
    """Full staged compilation pipeline: resolve, compile, and run."""
    resolved = _resolve_all_comptime(module, infer=infer, lower=lower)
    # Stage-1: resolve comptime fields that depend on runtime args
    if args:
        changed = True
        while changed:
            changed = False
            func = resolved.functions[0]
            for op in list(func.body.ops):
                for field_name, _ in op.__params__:
                    value = getattr(op, field_name)
                    if not isinstance(value, dgen.Value):
                        continue
                    if isinstance(value, Constant):
                        continue
                    _resolve_comptime_field(
                        func,
                        op,
                        field_name,
                        value,
                        lower,
                        block_args=func.body.args,
                        args=args,
                    )
                    resolved = infer(resolved)
                    _resolve_constant_ops(resolved.functions[0])
                    changed = True
                    break
                if changed:
                    break
    typed = infer(resolved)
    lowered = lower(typed)
    exe = codegen.compile(lowered)
    return exe.run(*args)
