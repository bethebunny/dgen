"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from copy import deepcopy

import dgen
from dgen import codegen
from dgen.block import BlockArgument
from dgen.codegen import _ctype, _llvm_type
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, String
from dgen.module import ConstantOp, Function, Module
from dgen.type import Memory
from dgen.type import Constant


def _trace_dependencies(target: dgen.Value, func: FunctionOp) -> list[dgen.Op]:
    """Backward-walk from target, return all needed ops in topological order."""
    needed: set[int] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if id(val) in needed:
            continue
        needed.add(id(val))
        if isinstance(val, dgen.Op):
            for _, param in val.parameters:
                worklist.append(param)
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
            for _, param in val.parameters:
                worklist.append(param)
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
    lower: Callable[[Module], Module],
    *,
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    ops = list(subgraph) + [builtin.ReturnOp(value=target)]
    func = FunctionOp(
        name="main",
        body=dgen.Block(ops=ops, args=list(block_args)),
        type=Function(result=target.type),
    )
    module = Module(functions=[func])
    lowered = lower(module)
    exe = codegen.compile(lowered)
    memories = _make_memories(block_args, args)
    raw = exe.run(*memories)

    # Convert result back to Python while JIT buffers are still alive
    return _raw_to_json(raw, target.type)


def _resolve_comptime_field(
    func: FunctionOp,
    op: dgen.Op,
    field_name: str,
    value: dgen.Value,
    lower: Callable[[Module], Module],
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
        func.body.ops = [o for o in func.body.ops if id(o) not in subgraph_ids]
    const_op = ConstantOp(value=result, type=value.type)
    idx = func.body.ops.index(op)
    func.body.ops.insert(idx, const_op)
    setattr(op, field_name, const_op)


def _resolve_constant_ops(func: FunctionOp) -> None:
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
        const = ConstantOp(value=val, type=op.type)
        func.body.ops[i] = const
        for other in func.body.ops:
            for param_name, param_value in other.parameters:
                if param_value is op:
                    setattr(other, param_name, const)


# ---------------------------------------------------------------------------
# Stage computation
# ---------------------------------------------------------------------------


def _field_values(op: dgen.Op, fields: dgen.type.Fields) -> list[dgen.Value]:
    """Get all Value inputs from a set of fields, flattening list-valued ones."""
    result: list[dgen.Value] = []
    for name, _ in fields:
        val = getattr(op, name)
        if isinstance(val, list):
            result.extend(v for v in val if isinstance(v, dgen.Value))
        elif isinstance(val, dgen.Value):
            result.append(val)
    return result


def compute_stages(func: FunctionOp) -> dict[int, int]:
    """Assign a stage number to every Value in a function.

    Base cases:
      - Constant / ConstantOp: stage 0
      - BlockArgument: stage 1

    For ops::

        stage = max((
            *(1 + stage(p) for p in __params__),
            *(stage(v) for v in __operands__),
        ))

    Returns a dict mapping ``id(value) → stage_number``.
    """
    stages: dict[int, int] = {}

    def _stage(value: dgen.Value) -> int:
        vid = id(value)
        if vid in stages:
            return stages[vid]
        if isinstance(value, Constant):
            stages[vid] = 0
            return 0
        if isinstance(value, BlockArgument):
            stages[vid] = 1
            return 1
        assert isinstance(value, dgen.Op)
        parts: list[int] = []
        for pv in _field_values(value, value.__params__):
            parts.append(1 + _stage(pv))
        for ov in _field_values(value, value.__operands__):
            parts.append(_stage(ov))
        result = max(parts, default=0)
        stages[vid] = result
        return result

    for arg in func.body.args:
        stages[id(arg)] = 1
    for op in func.body.ops:
        _stage(op)
    return stages


def _unresolved_boundaries(
    func: FunctionOp,
    stages: dict[int, int],
) -> list[tuple[int, dgen.Op, str, dgen.Value]]:
    """Find ops with unresolved __params__, sorted by stage number.

    Returns ``(stage, op, field_name, param_value)`` tuples for every
    ``__params__`` field that is a non-Constant Value.
    """
    boundaries: list[tuple[int, dgen.Op, str, dgen.Value]] = []
    for op in func.body.ops:
        for field_name, value in op.parameters:
            if isinstance(value, dgen.Value) and not isinstance(value, Constant):
                boundaries.append((stages.get(id(op), 0), op, field_name, value))
    boundaries.sort(key=lambda t: t[0])
    return boundaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_all_comptime(
    module: Module,
    *,
    infer: Callable[[Module], Module],
    lower: Callable[[Module], Module],
) -> Module:
    """Deepcopy + resolve all Constant fields in stage order.

    Computes stage numbers, then processes unresolved ``__params__``
    boundaries from lowest stage first.  Stops when remaining boundaries
    depend on runtime values (BlockArguments).
    """
    module = deepcopy(module)
    while True:
        func = module.functions[0]
        stages = compute_stages(func)
        boundaries = _unresolved_boundaries(func, stages)
        if not boundaries:
            break
        _stage_num, op, field_name, value = boundaries[0]
        if not _is_stage0_evaluable(value):
            break  # remaining boundaries need runtime args
        _resolve_comptime_field(func, op, field_name, value, lower)
        module = infer(module)
        _resolve_constant_ops(module.functions[0])
    return module


def _raw_to_json(raw: object, ty: dgen.Type) -> object:
    """Convert a raw ctypes callback value to a Python value.

    Scalars (int, float) pass through. Pointer types are read from memory
    via Memory.from_raw().to_json().
    """
    layout = ty.__layout__
    if _ctype(layout) is ctypes.c_void_p:
        assert isinstance(raw, int)
        return Memory.from_raw(ty, raw).to_json()
    return raw


def _compile_with_callbacks(
    resolved: Module,
    func: FunctionOp,
    *,
    infer: Callable[[Module], Module],
    lower: Callable[[Module], Module],
) -> codegen.Executable:
    """Build a stage-1 thunk that calls a host callback for stage-2 JIT.

    The compiled function passes all its arguments to a host callback.
    The callback resolves all remaining __params__ values (using the full
    stage-1 resolution loop), then JIT-compiles and executes stage-2.
    """
    stage2_template = resolved

    # Derive callback LLVM signature from original function
    assert func.name is not None
    callback_name = f"_stage2_{func.name}"
    orig_types = [arg.type for arg in func.body.args]
    orig_llvm_types = [_llvm_type(t.__layout__) for t in orig_types]
    if isinstance(func.type.result, builtin.Nil):
        ret_llvm = "void"
        result_ctype: type[ctypes._CData] | None = None
    else:
        ret_llvm = _llvm_type(func.type.result.__layout__)
        result_ctype = _ctype(func.type.result.__layout__)
    extern_decl = f"declare {ret_llvm} @{callback_name}({', '.join(orig_llvm_types)})"

    # Build ctypes callback type
    param_ctypes = [_ctype(t.__layout__) for t in orig_types]
    cb_type = ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    # Capture template + pipeline in closure
    def _callback(*raw_args: object) -> object:
        # Convert raw ctypes values to Python values
        python_args = [
            _raw_to_json(raw_args[i], orig_types[i]) for i in range(len(orig_types))
        ]

        # Deep-copy template and resolve all __params__ in stage order
        template = deepcopy(stage2_template)
        while True:
            s2_func = template.functions[0]
            stages = compute_stages(s2_func)
            boundaries = _unresolved_boundaries(s2_func, stages)
            if not boundaries:
                break
            _stage_num, op, field_name, value = boundaries[0]
            _resolve_comptime_field(
                s2_func,
                op,
                field_name,
                value,
                lower,
                block_args=s2_func.body.args,
                args=python_args,
            )
            template = infer(template)
            _resolve_constant_ops(template.functions[0])

        # Compile and run stage-2
        typed = infer(template)
        lowered = lower(typed)
        exe = codegen.compile(lowered)
        return exe.run(*python_args)

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
        callee=String().constant(callback_name),
        args=thunk_args,
        type=func.type.result,
    )
    if isinstance(func.type.result, builtin.Nil):
        ret_op = builtin.ReturnOp()
    else:
        ret_op = builtin.ReturnOp(value=call_op)

    thunk_func = FunctionOp(
        name=func.name,
        body=dgen.Block(ops=[call_op, ret_op], args=thunk_args),
        type=Function(result=func.type.result),
    )
    thunk_module = Module(functions=[thunk_func])

    exe = codegen.compile(thunk_module, externs=[extern_decl])
    exe.host_refs.append(callback_func)  # prevent GC
    return exe


def compile_staged(
    module: Module,
    *,
    infer: Callable[[Module], Module],
    lower: Callable[[Module], Module],
) -> codegen.Executable:
    """Stage-resolve, infer, lower, and compile to an Executable.

    If all __params__ are resolvable at compile time (stage-0), compiles directly.
    If some __params__ depend on runtime values, builds a callback-based executable
    that JIT-compiles stage-2 code when runtime values become available.
    """
    resolved = _resolve_all_comptime(module, infer=infer, lower=lower)
    func = resolved.functions[0]

    stages = compute_stages(func)
    if _unresolved_boundaries(func, stages):
        return _compile_with_callbacks(resolved, func, infer=infer, lower=lower)

    typed = infer(resolved)
    lowered = lower(typed)
    return codegen.compile(lowered)


def compile_and_run_staged(
    module: Module,
    *,
    infer: Callable[[Module], Module],
    lower: Callable[[Module], Module],
    args: Sequence = (),
) -> object:
    """Full staged compilation pipeline: resolve, compile, and run."""
    exe = compile_staged(module, infer=infer, lower=lower)
    return exe.run(*args)
