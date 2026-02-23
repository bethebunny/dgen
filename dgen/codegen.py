"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import ctypes
import sys
from io import StringIO

import dgen
from dgen.asm.formatting import SlotTracker, format_float
from dgen.dialects import builtin, llvm
from dgen.layout import Layout, Memory

# ---------------------------------------------------------------------------
# struct.format → LLVM / ctypes mapping
# ---------------------------------------------------------------------------

_FMT_LLVM = {"q": "i64", "d": "double"}
_FMT_CTYPE = {"q": ctypes.c_int64, "d": ctypes.c_double}


def _llvm_type(layout: Layout) -> str:
    """Derive LLVM type from a layout's struct format."""
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_LLVM.get(fmt, "ptr")


def _ctype(layout: Layout) -> type[ctypes._SimpleCData]:  # type: ignore[type-arg]
    """Derive ctypes type from a layout's struct format."""
    fmt = layout.struct.format.lstrip("=@<>!")
    return _FMT_CTYPE.get(fmt, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# Runtime: print_memref callback
# ---------------------------------------------------------------------------


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
def _print_memref(ptr: int, size: int) -> None:
    arr = (ctypes.c_double * size).from_address(ptr)
    print(", ".join(f"{arr[i]:g}" for i in range(size)))


# ---------------------------------------------------------------------------
# LLVM initialization
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        import llvmlite.binding as _llvm

        _llvm.initialize_native_target()
        _llvm.initialize_native_asmprinter()
        _llvm.add_symbol(
            "print_memref", ctypes.cast(_print_memref, ctypes.c_void_p).value
        )
        _initialized = True


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def emit_llvm_ir(module: builtin.Module, host_buffers: list | None = None) -> str:
    """Emit valid LLVM IR text that llvmlite can parse."""
    if host_buffers is None:
        host_buffers = []
    lines: list[str] = ["declare void @print_memref(ptr, i64)", ""]
    for func in module.functions:
        lines.extend(_emit_func(func, host_buffers))
    return "\n".join(lines)


def _emit_func(f: builtin.FuncOp, host_buffers: list) -> list[str]:
    # Build a SlotTracker so all ops get stable names
    tracker = SlotTracker()
    # Register block args (function parameters) first
    for arg in f.body.args:
        tracker.get_name(arg)
    builtin._register_ops(tracker, f.body.ops)

    # Pre-scan: build constants and types maps (keyed by id)
    constants: dict[int, str] = {}  # id(op) -> typed literal
    types: dict[int, str] = {}  # id(op) -> LLVM type

    # Register block arg types
    for arg in f.body.args:
        types[id(arg)] = _llvm_type(arg.type.__layout__)

    for op in f.body.ops:
        vid = id(op)
        if isinstance(op, builtin.ConstantOp):
            layout = op.type.__layout__
            if isinstance(op.value, list):
                mem = Memory(op.type)
                mem.pack(*op.value)
                host_buffers.append(mem)
                constants[vid] = f"ptr inttoptr (i64 {mem.address} to ptr)"
                types[vid] = "ptr"
            else:
                lt = _llvm_type(layout)
                val_str = (
                    format_float(op.value)
                    if isinstance(op.value, float)
                    else str(op.value)
                )
                constants[vid] = f"{lt} {val_str}"
                types[vid] = lt
        elif isinstance(op, llvm.AllocaOp):
            types[vid] = "ptr"
        elif isinstance(op, llvm.GepOp):
            types[vid] = "ptr"
        elif isinstance(op, llvm.LoadOp):
            types[vid] = "double"
        elif isinstance(op, (llvm.FAddOp, llvm.FMulOp)):
            types[vid] = "double"
        elif isinstance(op, (llvm.AddOp, llvm.MulOp)):
            types[vid] = "i64"
        elif isinstance(op, llvm.FcmpOp):
            types[vid] = "i1"
        elif isinstance(op, llvm.ZextOp):
            types[vid] = "i64"
        elif isinstance(op, llvm.IcmpOp):
            types[vid] = "i1"
        elif isinstance(op, llvm.PhiOp):
            first_val = op.values[0]
            types[vid] = types.get(id(first_val), "i64")

    def typed_ref(val: dgen.Value) -> str:
        """'type value' — e.g. 'double 1.0' or 'ptr %v3'."""
        vid = id(val)
        if vid in constants:
            return constants[vid]
        return f"{types.get(vid, 'i64')} %{tracker.get_name(val)}"

    def bare_ref(val: dgen.Value) -> str:
        """Just the value — e.g. '1.0' or '%v3'."""
        vid = id(val)
        if vid in constants:
            return constants[vid].split(" ", 1)[1]
        return f"%{tracker.get_name(val)}"

    func_name = tracker.get_name(f) if f.name is not None else f.name
    # Derive LLVM return type from function signature
    result_type = f.type.result
    if isinstance(result_type, builtin.Nil):
        llvm_ret = "void"
    else:
        llvm_ret = _llvm_type(result_type.__layout__)
    # Build parameter list from block args
    params = []
    for arg in f.body.args:
        name = tracker.get_name(arg)
        ty = types.get(id(arg), "i64")
        params.append(f"{ty} %{name}")
    param_str = ", ".join(params)
    lines = [f"define {llvm_ret} @{func_name}({param_str}) {{", "entry:"]

    for op in f.body.ops:
        if isinstance(op, builtin.ConstantOp):
            continue

        name = tracker.get_name(op)

        if isinstance(op, llvm.LabelOp):
            lines.append(f"{op.label_name}:")
        elif isinstance(op, llvm.AllocaOp):
            lines.append(f"  %{name} = alloca double, i64 {op.elem_count}")
        elif isinstance(op, llvm.GepOp):
            lines.append(
                f"  %{name} = getelementptr double, ptr {bare_ref(op.base)},"
                f" {typed_ref(op.index)}"
            )
        elif isinstance(op, llvm.LoadOp):
            lines.append(f"  %{name} = load double, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.StoreOp):
            lines.append(f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}")
        elif isinstance(op, llvm.FAddOp):
            lines.append(
                f"  %{name} = fadd double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.FMulOp):
            lines.append(
                f"  %{name} = fmul double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.AddOp):
            lines.append(f"  %{name} = add i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.MulOp):
            lines.append(f"  %{name} = mul i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}")
        elif isinstance(op, llvm.FcmpOp):
            lines.append(
                f"  %{name} = fcmp {op.pred} double {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.ZextOp):
            lines.append(f"  %{name} = zext i1 {bare_ref(op.input)} to i64")
        elif isinstance(op, llvm.IcmpOp):
            lines.append(
                f"  %{name} = icmp {op.pred} i64 {bare_ref(op.lhs)}, {bare_ref(op.rhs)}"
            )
        elif isinstance(op, llvm.BrOp):
            lines.append(f"  br label %{op.dest}")
        elif isinstance(op, llvm.CondBrOp):
            lines.append(
                f"  br i1 %{tracker.get_name(op.cond)}, label %{op.true_dest}, label %{op.false_dest}"
            )
        elif isinstance(op, llvm.PhiOp):
            ty = types.get(id(op), "i64")
            pairs = ", ".join(
                f"[ {bare_ref(v)}, %{lab} ]" for v, lab in zip(op.values, op.labels)
            )
            lines.append(f"  %{name} = phi {ty} {pairs}")
        elif isinstance(op, llvm.CallOp):
            if op.callee == "print_memref" and len(op.args) == 2:
                a = f"{typed_ref(op.args[0])}, {typed_ref(op.args[1])}"
                lines.append(f"  call void @print_memref({a})")
            else:
                a = ", ".join(typed_ref(arg) for arg in op.args)
                lines.append(f"  call void @{op.callee}({a})")
        elif isinstance(op, builtin.ReturnOp):
            if op.value is not None:
                lines.append(f"  ret {typed_ref(op.value)}")
            else:
                lines.append("  ret void")

    lines.append("}")
    return lines


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------


def _func_param_ctypes(func: builtin.FuncOp) -> list:
    """Map function block arg types to ctypes parameter types."""
    return [_ctype(arg.type.__layout__) for arg in func.body.args]


def compile_and_run(
    ll_module: builtin.Module,
    capture_output: bool = False,
    args: list | None = None,
) -> str | None:
    """Emit LLVM IR, JIT-compile, and execute the module's main function."""
    import llvmlite.binding as _llvm

    _ensure_initialized()
    host_buffers: list = []
    ir_text = emit_llvm_ir(ll_module, host_buffers)

    mod = _llvm.parse_assembly(ir_text)
    mod.verify()

    target = _llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = _llvm.create_mcjit_compiler(mod, target_machine)

    main_func = ll_module.functions[0]
    param_types = _func_param_ctypes(main_func)
    func_ptr = engine.get_function_address(main_func.name)
    cfunc = ctypes.CFUNCTYPE(None, *param_types)(func_ptr)
    call_args = args if args else []

    if capture_output:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            cfunc(*call_args)
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
    else:
        cfunc(*call_args)
        return None


def jit_eval(
    ll_module: builtin.Module, return_layout: Layout, args: list | None = None
) -> object:
    """JIT-compile a module and return the result of its main function."""
    import llvmlite.binding as _llvm

    _ensure_initialized()
    host_buffers: list = []
    ir_text = emit_llvm_ir(ll_module, host_buffers)

    mod = _llvm.parse_assembly(ir_text)
    mod.verify()

    target = _llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = _llvm.create_mcjit_compiler(mod, target_machine)

    main_func = ll_module.functions[0]
    param_types = _func_param_ctypes(main_func)
    func_ptr = engine.get_function_address(main_func.name)
    cfunc = ctypes.CFUNCTYPE(_ctype(return_layout), *param_types)(func_ptr)
    call_args = args if args else []
    return cfunc(*call_args)
