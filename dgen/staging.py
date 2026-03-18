"""Staging evaluator: JIT-compile comptime expressions to resolve dependent types."""

from __future__ import annotations

import ctypes
import itertools
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Iterator, TypeVar

import dgen
from dgen import codegen
from dgen.block import BlockArgument
from dgen.codegen import Executable, _ctype, _llvm_type
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, String
from dgen.module import ConstantOp, Module, PackOp
from dgen.type import Constant, Memory

if TYPE_CHECKING:
    from dgen.compiler import Compiler

T = TypeVar("T")


def _walk_inputs(op: dgen.Op) -> Iterator[dgen.Value]:
    """Append all parameter and operand Values to worklist, flattening lists."""
    for _, val in itertools.chain(op.parameters, op.operands):
        if isinstance(val, list):
            # TODO: remove this case once we push through Tuple types
            yield from (v for v in val if isinstance(v, dgen.Value))
        elif isinstance(val, dgen.Value):
            yield val


def _trace_dependencies(target: dgen.Value, func: FunctionOp) -> list[dgen.Op]:
    """Backward-walk from target, return all needed ops in topological order."""
    needed: set[dgen.Value] = set()
    worklist = [target]
    while worklist:
        val = worklist.pop()
        if val in needed:
            continue
        needed.add(val)
        if isinstance(val, dgen.Op):
            worklist.extend(_walk_inputs(val))
    return [op for op in func.body.ops if op in needed]


def _make_memories(
    block_args: Sequence[BlockArgument], python_args: Sequence
) -> list[Memory]:
    """Convert Python args to Memory objects using block argument types."""
    return [
        Memory.from_value(param.type, arg)
        for arg, param in zip(python_args, block_args)
    ]


def _extern_declarations(subgraph: list[dgen.Op]) -> list[str]:
    """Generate LLVM extern declarations for function calls in a subgraph."""
    externs: list[str] = []
    seen: set[str] = set()
    for op in subgraph:
        if not isinstance(op, builtin.CallOp):
            continue
        callee_name = op.callee.name
        if callee_name is None or callee_name in seen:
            continue
        seen.add(callee_name)
        # Derive return type from CallOp's result type
        result_type = dgen.type.type_constant(op.type)
        if isinstance(result_type, builtin.Nil):
            ret_llvm = "void"
        else:
            ret_llvm = _llvm_type(result_type.__layout__)
        # Derive param types from the call args
        if isinstance(op.args, PackOp):
            arg_values = op.args.values
        else:
            arg_values = [op.args]
        param_types = [
            _llvm_type(dgen.type.type_constant(arg.type).__layout__)
            for arg in arg_values
        ]
        externs.append(f"declare {ret_llvm} @{callee_name}({', '.join(param_types)})")
    return externs


def _jit_evaluate(
    subgraph: list[dgen.Op],
    target: dgen.Value,
    lower: Callable[[Module], Module],
    *,
    block_args: Sequence[BlockArgument] = (),
    args: Sequence = (),
) -> object:
    """Build a mini-module from the subgraph, lower via the caller's pipeline, JIT."""
    assert target.ready
    externs = _extern_declarations(subgraph)
    ops = list(subgraph)
    func = FunctionOp(
        name="main",
        body=dgen.Block(result=target, args=list(block_args)),
        result=target.type,
    )
    module = Module(functions=[func])
    lowered = lower(module)
    exe = codegen.compile(lowered, externs=externs)
    memories = _make_memories(block_args, args)
    raw = exe.run(*memories)

    # Convert result back to Python while JIT buffers are still alive
    return _raw_to_json(raw, dgen.type.type_constant(target.type))


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
    const_type = value.type
    if isinstance(result, dict) and "tag" in result:
        const_type = dgen.type.TypeType()
    const_op = ConstantOp(value=result, type=const_type)
    setattr(op, field_name, const_op)


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


def compute_stages(func: FunctionOp) -> dict[dgen.Value, int]:
    """Assign a stage number to every Value in a function.

    Stage 0 means the value can be evaluated at compile time (no runtime
    dependencies). Stage 1+ means it depends on runtime values.

    Base cases:
      - Constant / ConstantOp: stage 0
      - BlockArgument: stage 1

    For ops, params bump the stage only when they depend on runtime::

        stage = max((
            *((1 + stage(p) if stage(p) > 0 else stage(p)) for p in __params__),
            *(stage(v) for v in __operands__),
        ))

    Returns a dict mapping ``value → stage_number``.
    """
    stages: dict[dgen.Value, int] = {}

    def _stage(value: dgen.Value) -> int:
        if value in stages:
            return stages[value]
        if isinstance(value, (Constant, dgen.Type, FunctionOp)):
            stages[value] = 0
            return 0
        if isinstance(value, BlockArgument):
            stages[value] = 1
            return 1
        if not isinstance(value, dgen.Op):
            # Forward reference to a module-level entity (e.g. function name)
            stages[value] = 0
            return 0
        parts: list[int] = []
        for pv in _field_values(value, value.__params__):
            s = _stage(pv)
            # +1 only when the param depends on runtime values (stage > 0).
            # Stage-0 params (all-constant deps) don't bump the stage —
            # the op is still evaluable at compile time.
            parts.append(1 + s if s > 0 else s)
        for ov in _field_values(value, value.__operands__):
            parts.append(_stage(ov))
        # Unresolved type ref counts as a param boundary
        if isinstance(value.type, dgen.Value) and not isinstance(
            value.type, (Constant, dgen.Type)
        ):
            s = _stage(value.type)
            parts.append(1 + s if s > 0 else s)
        result = max(parts, default=0)
        stages[value] = result
        return result

    for arg in func.body.args:
        stages[arg] = 1
    for op in func.body.ops:
        _stage(op)
    return stages


def _unresolved_boundaries(
    func: FunctionOp,
    stages: dict[dgen.Value, int],
) -> list[tuple[int, dgen.Op, str, dgen.Value]]:
    """Find ops with unresolved __params__, sorted by stage number.

    Returns ``(stage, op, field_name, param_value)`` tuples for every
    ``__params__`` field that is a non-Constant Value.
    """
    boundaries: list[tuple[int, dgen.Op, str, dgen.Value]] = []
    for op in func.body.ops:
        for field_name, value in op.parameters:
            if isinstance(value, (dgen.Op, BlockArgument)) and not isinstance(
                value, (Constant, dgen.Type, FunctionOp)
            ):
                boundaries.append((stages.get(op, 0), op, field_name, value))
        # Also check op.type — if it's an unresolved SSA ref (Value, not Type)
        if isinstance(op.type, dgen.Value) and not isinstance(
            op.type, (Constant, dgen.Type)
        ):
            boundaries.append((stages.get(op, 0), op, "type", op.type))
    boundaries.sort(key=lambda t: t[0])
    return boundaries


# ---------------------------------------------------------------------------
# IfOp specialization
# ---------------------------------------------------------------------------


def _specialize_ifs(
    func: FunctionOp,
    lower: Callable[[Module], Module],
    block_args: Sequence[BlockArgument],
    args: Sequence,
) -> None:
    """Evaluate IfOp conditions with runtime args and inline the taken branch.

    After specialization, nested block ops are flattened into func.body.ops
    and the IfOp is removed. References to the IfOp result are rewired to
    the branch's return value.
    """
    replacements: dict[dgen.Value, dgen.Value] = {}
    new_ops: list[dgen.Op] = []

    for op in func.body.ops:
        if not isinstance(op, builtin.IfOp):
            new_ops.append(op)
            continue

        # Evaluate the condition
        cond = replacements.get(op.cond, op.cond)
        subgraph = _trace_dependencies(cond, func)
        cond_result = _jit_evaluate(
            subgraph,
            cond,
            lower,
            block_args=block_args,
            args=args,
        )

        # Pick the taken branch
        branch = op.then_body if cond_result else op.else_body

        # Inline branch ops; block.result is the terminator value
        new_ops.extend(branch.ops)
        val = branch.result
        replacements[op] = replacements.get(val, val)

    # Apply replacements to all inlined ops
    for op in new_ops:
        for fname, fval in op.operands:
            if isinstance(fval, dgen.Value):
                mapped = replacements.get(fval)
                if mapped is not None:
                    setattr(op, fname, mapped)
        for fname, fval in op.parameters:
            if isinstance(fval, dgen.Value):
                mapped = replacements.get(fval)
                if mapped is not None:
                    setattr(op, fname, mapped)

    func.body.result = new_ops[-1]


def _has_nested_boundaries(func: FunctionOp) -> bool:
    """True if any op in nested blocks has unresolved __params__."""
    for op in func.body.ops:
        for _, block in op.blocks:
            for nested_op in block.ops:
                for _, value in nested_op.parameters:
                    if isinstance(value, (dgen.Op, BlockArgument)) and not isinstance(
                        value, (Constant, dgen.Type, FunctionOp)
                    ):
                        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _raw_to_json(raw: object, ty: dgen.Type) -> object:
    """Convert a raw ctypes callback value to a Python value.

    Scalars (int, float) pass through. Pointer types are read from memory
    via Memory.from_raw().to_json().

    TypeType values use the self-describing TypeValue layout which reads
    through the pointer and resolves the tag to determine the full Record.
    """
    layout = ty.__layout__
    if _ctype(layout) is ctypes.c_void_p:
        assert isinstance(raw, int)
        return Memory.from_raw(ty, raw).to_json()
    return raw


def resolve_stage0(
    module: Module,
    lower: Callable[[Module], Module],
) -> Module:
    """Deepcopy + resolve all stage-0 comptime boundaries.

    Iteratively resolves the lowest-stage boundary whose value can be
    evaluated without runtime values (no BlockArgument dependencies).
    Stops when only runtime-dependent boundaries remain.
    """
    module = deepcopy(module)
    while True:
        resolved_any = False
        for func in module.functions:
            stages = compute_stages(func)
            boundaries = _unresolved_boundaries(func, stages)
            if not boundaries:
                continue
            _stage_num, op, field_name, value = boundaries[0]
            if stages.get(value, 0) != 0:
                continue
            _resolve_comptime_field(func, op, field_name, value, lower)
            resolved_any = True
            break  # re-iterate from the start after each resolution
        if not resolved_any:
            break
    return module


def _resolve_with_runtime_args(
    func: FunctionOp,
    lower: Callable[[Module], Module],
    block_args: Sequence[BlockArgument],
    python_args: Sequence,
) -> None:
    """Resolve all boundaries in a function using runtime argument values.

    Unlike ``resolve_stage0``, this resolves boundaries at any stage
    by passing runtime args to the JIT evaluator. Used inside callback
    thunks when runtime values become available.
    """
    while True:
        stages = compute_stages(func)
        boundaries = _unresolved_boundaries(func, stages)
        if not boundaries:
            break
        _stage_num, op, field_name, value = boundaries[0]
        _resolve_comptime_field(
            func,
            op,
            field_name,
            value,
            lower,
            block_args=block_args,
            args=python_args,
        )


def _build_callback_thunk(
    resolved: Module,
    func: FunctionOp,
    compiler: Compiler[T],
) -> Executable:
    """Build a stage-1 thunk that calls a host callback for stage-2 JIT.

    The compiled function passes all its arguments to a host callback.
    The callback resolves all remaining __params__ values (using the full
    stage-1 resolution loop), then JIT-compiles and executes stage-2.
    """
    stage2_template = resolved

    # Derive callback LLVM signature from original function
    assert func.name is not None
    callback_name = f"_stage2_{func.name}"
    orig_types = [dgen.type.type_constant(arg.type) for arg in func.body.args]
    orig_llvm_types = [_llvm_type(t.__layout__) for t in orig_types]
    result_type = dgen.type.type_constant(func.result)
    if isinstance(result_type, builtin.Nil):
        ret_llvm = "void"
        result_ctype: type[ctypes._CData] | None = None
    else:
        ret_llvm = _llvm_type(result_type.__layout__)
        result_ctype = _ctype(result_type.__layout__)
    extern_decl = f"declare {ret_llvm} @{callback_name}({', '.join(orig_llvm_types)})"

    # Build ctypes callback type
    param_ctypes = [_ctype(t.__layout__) for t in orig_types]
    cb_type = ctypes.CFUNCTYPE(result_ctype, *param_ctypes)

    # Capture template + compiler in closure
    func_name = func.name
    callback_host_refs: list[object] = []  # Keep JIT data alive across calls

    def _callback(*raw_args: object) -> object:
        # Convert raw ctypes values to Python values
        python_args = [
            _raw_to_json(raw_args[i], orig_types[i]) for i in range(len(orig_types))
        ]

        # Deep-copy template and resolve all __params__ with runtime args
        template = deepcopy(stage2_template)
        s2_func = next(f for f in template.functions if f.name == func_name)

        # Specialize IfOps: evaluate conditions, inline taken branches
        _specialize_ifs(s2_func, compiler.run, s2_func.body.args, python_args)

        # Resolve all remaining boundaries with runtime args
        _resolve_with_runtime_args(
            s2_func, compiler.run, s2_func.body.args, python_args
        )

        # Compile the fully-resolved function through the full pipeline
        func_module = Module(functions=[s2_func])
        result = compiler.compile(func_module)
        assert isinstance(result, Executable)
        callback_host_refs.extend(result.host_refs)  # Keep memory alive
        return result.run(*python_args)

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
    pack = PackOp(values=thunk_args, type=builtin.List(element_type=builtin.Index()))
    call_op = llvm.CallOp(
        callee=String().constant(callback_name),
        args=pack,
        type=result_type,
    )
    thunk_func = FunctionOp(
        name=func.name,
        body=dgen.Block(result=call_op, args=thunk_args),
        result=result_type,
    )
    thunk_module = Module(functions=[thunk_func])

    exe = codegen.compile(thunk_module, externs=[extern_decl])
    exe.host_refs.append(callback_func)  # prevent GC
    exe.host_refs.append(callback_host_refs)  # prevent GC of callback results
    return exe


def compile_module(module: Module, compiler: Compiler[T]) -> T:
    """Full staged compilation: resolve stage-0, run passes, exit.

    If all __params__ are resolvable at compile time (stage-0), resolves them
    and proceeds through the normal pass pipeline.

    If some __params__ depend on runtime values (stage-1+), builds callback-
    based thunks that JIT-compile when runtime values become available.

    For multi-function modules where multiple functions have unresolved
    boundaries, each is compiled as a callback thunk and registered as a
    global symbol so cross-function calls (including recursion) work.
    """
    resolved = resolve_stage0(module, compiler.run)

    # Find all functions with unresolved boundaries (including nested blocks)
    unresolved_funcs: list[FunctionOp] = []
    for func in resolved.functions:
        stages = compute_stages(func)
        if _unresolved_boundaries(func, stages) or _has_nested_boundaries(func):
            unresolved_funcs.append(func)

    if not unresolved_funcs:
        lowered = compiler.run(resolved)
        return compiler.exit.run(lowered)

    # Compile each unresolved function as a callback thunk
    # and register it as a global symbol for cross-function calls.
    # Process non-entry functions first so their symbols are available
    # when the entry function's callback fires.
    import llvmlite.binding as _llvm_binding

    entry = resolved.functions[0]
    ordered = [f for f in unresolved_funcs if f is not entry] + [
        f for f in unresolved_funcs if f is entry
    ]

    all_host_refs: list[object] = []
    entry_exe: Executable | None = None

    for func in ordered:
        exe = _build_callback_thunk(resolved, func, compiler)
        all_host_refs.extend(exe.host_refs)

        # JIT the thunk and register its address as a global symbol
        engine = codegen._jit_engine(exe)
        assert func.name is not None
        func_ptr = engine.get_function_address(func.name)
        _llvm_binding.add_symbol(func.name, func_ptr)
        all_host_refs.append(engine)  # keep JIT engine alive

        if func is entry:
            entry_exe = exe

    if entry_exe is not None:
        entry_exe.host_refs = all_host_refs
        return entry_exe  # type: ignore[return-value]

    # Entry point has no unresolved boundaries — compile normally
    # but keep callback thunks alive
    lowered = compiler.run(resolved)
    result = compiler.exit.run(lowered)
    assert isinstance(result, Executable)
    result.host_refs.extend(all_host_refs)
    return result
