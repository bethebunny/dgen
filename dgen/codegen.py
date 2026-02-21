"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import ctypes
import sys
from io import StringIO

import dgen
from dgen.asm.formatting import SlotTracker, format_float
from dgen.dialects import builtin, llvm
from dgen.layout import Float64, Int

# ---------------------------------------------------------------------------
# Runtime: print_memref callback
# ---------------------------------------------------------------------------


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
def _print_memref(ptr, size):
    arr = (ctypes.c_double * size).from_address(ptr)
    print(", ".join(f"{arr[i]:g}" for i in range(size)))


# ---------------------------------------------------------------------------
# LLVM initialization
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_initialized():
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
    builtin._register_ops(tracker, f.body.ops)

    # Pre-scan: build constants and types maps (keyed by id)
    constants: dict[int, str] = {}  # id(op) -> typed literal
    types: dict[int, str] = {}  # id(op) -> LLVM type

    for op in f.body.ops:
        vid = id(op)
        if isinstance(op, builtin.ConstantOp):
            if isinstance(op.value, list):
                layout = op.type.__layout__  # type: ignore[union-attr]
                elem = layout.element
                ctype = ctypes.c_double if isinstance(elem, Float64) else ctypes.c_int64
                values = op.value
                assert isinstance(values, list)
                buf = (ctype * layout.count)(*values)
                host_buffers.append(buf)
                addr = ctypes.addressof(buf)
                constants[vid] = f"ptr inttoptr (i64 {addr} to ptr)"
                types[vid] = "ptr"
            elif isinstance(op.type, builtin.F64Type):
                assert isinstance(op.value, (int, float))
                constants[vid] = f"double {format_float(op.value)}"
                types[vid] = "double"
            elif isinstance(op.type, builtin.IndexType):
                constants[vid] = f"i64 {op.value}"
                types[vid] = "i64"
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
    lines = [f"define void @{func_name}() {{", "entry:"]

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
                f"[ {bare_ref(v)}, %{l} ]" for v, l in zip(op.values, op.labels)
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


def compile_and_run(
    ll_module: builtin.Module, capture_output: bool = False
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

    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(None)(func_ptr)

    if capture_output:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            cfunc()
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
    else:
        cfunc()
        return None
